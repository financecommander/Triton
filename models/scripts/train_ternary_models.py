"""
Training Script for Ternary Neural Networks

Trains ResNet-18 and MobileNetV2 with ternary quantization on ImageNet or CIFAR-10.
Uses the Triton GPU backend for optimized ternary operations.

Enhanced features:
- Early stopping with configurable patience
- TensorBoard and CSV logging
- Advanced data augmentation (CutMix, MixUp, AutoAugment, RandAugment)
- Label smoothing
- Complete checkpoint resumption (including scheduler state)
- Best model tracking by validation accuracy

Usage:
    # Standard CIFAR-10 training
    python train_ternary_models.py --model resnet18 --dataset cifar10 --batch_size 128 --epochs 500
    
    # Resume from checkpoint with early stopping
    python train_ternary_models.py --model resnet18 --dataset cifar10 --resume ./checkpoints/checkpoint_epoch_76.pth --epochs 500 --early_stopping_patience 40
    
    # With advanced augmentation and label smoothing
    python train_ternary_models.py --model resnet18 --dataset cifar10 --cutmix --mixup --label_smoothing 0.1 --epochs 500
"""

import argparse
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.resnet18.ternary_resnet18 import ternary_resnet18, quantize_model_weights
from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

try:
    from torchvision.transforms import autoaugment, RandAugment
    AUTOAUGMENT_AVAILABLE = True
except ImportError:
    AUTOAUGMENT_AVAILABLE = False
    print("Warning: AutoAugment not available. Update torchvision: pip install torchvision>=0.11.0")


class CutMix:
    """CutMix augmentation."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        return images, labels, labels[rand_index], lam

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class MixUp:
    """MixUp augmentation."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        
        return mixed_images, labels, labels[rand_index], lam


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        
        # One-hot encode target
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class EarlyStopping:
    """Early stopping callback."""
    def __init__(self, patience=40, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


def get_dataset(dataset_name: str, train: bool = True, use_autoaugment=False, use_randaugment=False):
    """Get dataset with appropriate transforms."""
    if dataset_name.lower() == 'cifar10':
        if train:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            
            # Add AutoAugment or RandAugment if requested and available
            if use_autoaugment and AUTOAUGMENT_AVAILABLE:
                transform_list.append(autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.CIFAR10))
            elif use_randaugment and AUTOAUGMENT_AVAILABLE:
                transform_list.append(RandAugment(num_ops=2, magnitude=9))
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name.lower() == 'imagenet':
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Note: ImageNet requires manual download
        dataset = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='train' if train else 'val', transform=transform
        )
        num_classes = 1000

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, num_classes


def get_model(model_name: str, num_classes: int):
    """Get the specified ternary model."""
    if model_name.lower() == 'resnet18':
        model = ternary_resnet18(num_classes=num_classes)
    elif model_name.lower() == 'mobilenetv2':
        model = ternary_mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval=100, 
                use_cutmix=False, use_mixup=False, cutmix_alpha=1.0, mixup_alpha=1.0, writer=None, global_step=0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    
    cutmix_fn = CutMix(alpha=cutmix_alpha) if use_cutmix else None
    mixup_fn = MixUp(alpha=mixup_alpha) if use_mixup else None

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply CutMix or MixUp
        mixed = False
        if use_cutmix and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_fn((inputs, targets))
            mixed = True
        elif use_mixup and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_fn((inputs, targets))
            mixed = True

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        if mixed:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        # Quantize weights periodically during training
        if batch_idx % 100 == 0:
            quantize_model_weights(model)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if mixed:
            # For mixed samples, count as correct if prediction matches either label
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {batch_loss:.4f}, '
                  f'Acc: {batch_acc:.2f}%')
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('train/batch_loss', batch_loss, global_step + batch_idx)
                writer.add_scalar('train/batch_acc', batch_acc, global_step + batch_idx)

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    new_global_step = global_step + len(train_loader)

    return epoch_loss, epoch_acc, epoch_time, new_global_step


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, filename, early_stopping=None):
    """Save model checkpoint with complete state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'accuracy': accuracy
    }
    if early_stopping is not None:
        checkpoint['early_stopping'] = {
            'counter': early_stopping.counter,
            'best_score': early_stopping.best_score,
            'patience': early_stopping.patience
        }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train Ternary Neural Networks')
    
    # Model and dataset
    parser.add_argument('--model', type=str, choices=['resnet18', 'mobilenetv2'],
                       default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'],
                       default='cifar10', help='Dataset')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # Learning rate scheduling
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'none'],
                       default='cosine', help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
    
    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.0, 
                       help='Label smoothing factor (0.0 to 0.2, default: 0.0)')
    
    # Data augmentation
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha parameter')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='MixUp alpha parameter')
    parser.add_argument('--autoaugment', action='store_true', help='Use AutoAugment')
    parser.add_argument('--randaugment', action='store_true', help='Use RandAugment')
    
    # Checkpointing and resuming
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume from checkpoint (path to .pth file)')
    parser.add_argument('--save_freq', type=int, default=10, 
                       help='Save checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', 
                       help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=40,
                       help='Early stopping patience (epochs)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--csv_log', type=str, default=None,
                       help='Path to CSV log file (default: auto-generated)')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval in batches')
    
    # Other
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_subdir = f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(os.path.join(args.log_dir, log_subdir))
        print(f"TensorBoard logging to: {os.path.join(args.log_dir, log_subdir)}")
    
    # Initialize CSV logging
    csv_file = None
    csv_writer = None
    if args.csv_log:
        csv_path = args.csv_log
    else:
        csv_path = os.path.join(args.log_dir, f"{args.model}_{args.dataset}_metrics.csv")
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                        'lr', 'epoch_time', 'best_acc'])
    print(f"CSV logging to: {csv_path}")

    # Get dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, num_classes = get_dataset(
        args.dataset, train=True, 
        use_autoaugment=args.autoaugment,
        use_randaugment=args.randaugment
    )
    val_dataset, _ = get_dataset(args.dataset, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Get model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, num_classes)
    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"Using label smoothing: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    print(f"Optimizer: SGD(lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"Scheduler: CosineAnnealingLR(T_max={args.epochs})")
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(f"Scheduler: StepLR(step_size={args.step_size}, gamma={args.gamma})")
    else:
        print("No learning rate scheduler")

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
        print(f"Early stopping enabled with patience={args.early_stopping_patience}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    global_step = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state if available
            if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Restored scheduler state")
            
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('accuracy', 0.0)
            
            # Restore early stopping state if available
            if early_stopping is not None and 'early_stopping' in checkpoint:
                es_state = checkpoint['early_stopping']
                early_stopping.counter = es_state.get('counter', 0)
                early_stopping.best_score = es_state.get('best_score', None)
                print(f"Restored early stopping state (counter={early_stopping.counter})")
            
            print(f"Resumed from epoch {start_epoch} with best acc: {best_acc:.2f}%")
            global_step = start_epoch * len(train_loader)
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")

    # Print training configuration
    print("\n" + "="*80)
    print("Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset} (classes: {num_classes})")
    print(f"  Epochs: {start_epoch} -> {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial LR: {args.lr}")
    print(f"  Augmentations: CutMix={args.cutmix}, MixUp={args.mixup}, " 
          f"AutoAugment={args.autoaugment}, RandAugment={args.randaugment}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Early stopping: {args.early_stopping}")
    print("="*80 + "\n")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, epoch_time, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            log_interval=args.log_interval,
            use_cutmix=args.cutmix, use_mixup=args.mixup,
            cutmix_alpha=args.cutmix_alpha, mixup_alpha=args.mixup_alpha,
            writer=writer, global_step=global_step
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
        else:
            new_lr = current_lr

        # Print results
        print(f'\n{"="*80}')
        print(f'Epoch {epoch + 1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f} -> {new_lr:.6f}')
        print(f'  Time: {epoch_time:.2f}s')
        
        # Update best accuracy
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print(f'  *** New best accuracy: {best_acc:.2f}% ***')
        
        print(f'{"="*80}\n')

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('train/epoch_acc', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            writer.add_scalar('learning_rate', current_lr, epoch)
            writer.add_scalar('best_acc', best_acc, epoch)

        # Log to CSV
        if csv_writer is not None:
            csv_writer.writerow([
                epoch + 1, train_loss, train_acc, val_loss, val_acc,
                current_lr, epoch_time, best_acc
            ])
            csv_file.flush()

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_name = f"{args.model}_{args.dataset}_epoch_{epoch + 1}.pth"
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, 
                          checkpoint_path, early_stopping)

        # Save best model
        if is_best:
            best_checkpoint = os.path.join(args.checkpoint_dir, 
                                          f"{args.model}_{args.dataset}_best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, 
                          best_checkpoint, early_stopping)

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_acc):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_acc:.2f}%")
                print(f"No improvement for {early_stopping.patience} epochs")
                break

    # Close logging
    if writer is not None:
        writer.close()
    if csv_file is not None:
        csv_file.close()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total epochs: {epoch + 1}")
    print("="*80)


if __name__ == "__main__":
    main()