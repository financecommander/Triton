"""
Quantization-Aware Training (QAT) Script
Trains neural networks with quantization simulation for optimal post-training performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import argparse
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import Triton-compiled models
try:
    from models.resnet18.ternary_resnet18 import ternary_resnet18
    from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
except ImportError:
    print("Warning: Could not import Triton models. Using mock models.")


class QATTrainer:
    """Quantization-Aware Training orchestrator"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # QAT setup
        if args.qat_mode:
            self._setup_qat()
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
    def _create_model(self):
        """Create model based on args"""
        if self.args.model == 'resnet18':
            model = ternary_resnet18(num_classes=self.args.num_classes)
        elif self.args.model == 'mobilenetv2':
            model = ternary_mobilenet_v2(num_classes=self.args.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")
        
        # Load pretrained if specified
        if self.args.pretrained:
            checkpoint = torch.load(self.args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded pretrained model from {self.args.pretrained}")
        
        return model
    
    def _setup_qat(self):
        """Setup Quantization-Aware Training"""
        print("Setting up QAT...")
        
        # Add quantization stubs
        self.model.quant = QuantStub()
        self.model.dequant = DeQuantStub()
        
        # Configure quantization
        if self.args.backend == 'fbgemm':
            # x86 CPU
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        else:
            # ARM/Mobile
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        
        # Prepare for QAT
        self.model.train()
        prepare_qat(self.model, inplace=True)
        
        print(f"QAT configured with backend: {self.args.backend}")
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders"""
        if self.args.dataset == 'cifar10':
            # CIFAR-10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            valset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_val
            )
            
        elif self.args.dataset == 'imagenet':
            # ImageNet
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            trainset = torchvision.datasets.ImageFolder(
                root=f'{self.args.data_path}/train', transform=transform_train
            )
            valset = torchvision.datasets.ImageFolder(
                root=f'{self.args.data_path}/val', transform=transform_val
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        train_loader = DataLoader(
            trainset, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            valset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.workers, pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.args.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.args.scheduler == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.args.milestones,
                gamma=0.1
            )
        elif self.args.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.min_lr
            )
        elif self.args.scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Freeze batch norm stats after warmup
        if self.args.qat_mode and epoch >= self.args.freeze_bn_epoch:
            self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress
            if batch_idx % self.args.print_freq == 0:
                acc = 100. * correct / total
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.3f} | Acc: {acc:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        print(f'\nValidation: Loss: {val_loss:.3f} | Acc: {val_acc:.2f}%\n')
        
        return val_loss, val_acc
    
    def train(self):
        """Full training loop"""
        print(f"Starting QAT training for {self.args.epochs} epochs...")
        print(f"Model: {self.args.model}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Device: {self.device}")
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['lr'].append(current_lr)
            
            # Save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_acc': best_acc,
                'optimizer': self.optimizer.state_dict(),
                'metrics': self.metrics
            }, is_best)
            
            # Convert to quantized model periodically
            if self.args.qat_mode and epoch % 10 == 0:
                self._test_quantized_model(epoch)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"{'='*60}")
        
        # Final quantized model
        if self.args.qat_mode:
            self._convert_and_save_quantized()
        
        # Save metrics
        self._save_metrics()
        
        # Plot results
        self._plot_metrics()
    
    def _test_quantized_model(self, epoch):
        """Test quantized model accuracy"""
        print(f"\nTesting quantized model at epoch {epoch}...")
        
        # Convert to quantized
        self.model.eval()
        quantized_model = convert(self.model.cpu(), inplace=False)
        quantized_model.to(self.device)
        
        # Test
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = quantized_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        quant_acc = 100. * correct / total
        print(f"Quantized model accuracy: {quant_acc:.2f}%")
    
    def _convert_and_save_quantized(self):
        """Convert to fully quantized model and save"""
        print("\nConverting to quantized model...")
        
        self.model.eval()
        self.model.to('cpu')
        
        quantized_model = convert(self.model, inplace=False)
        
        # Save
        save_path = self.save_dir / 'quantized_model.pth'
        torch.save(quantized_model.state_dict(), save_path)
        print(f"Quantized model saved to {save_path}")
        
        # Model size
        model_size = save_path.stat().st_size / (1024 * 1024)
        print(f"Quantized model size: {model_size:.2f} MB")
    
    def _save_checkpoint(self, state, is_best):
        """Save checkpoint"""
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{state['epoch']}.pth"
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'model_best.pth'
            torch.save(state, best_path)
            print(f"✓ Best model saved: {state['best_acc']:.2f}%")
    
    def _save_metrics(self):
        """Save training metrics"""
        metrics_path = self.save_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    def _plot_metrics(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.metrics['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.metrics['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Accuracy difference
        acc_diff = [t - v for t, v in zip(self.metrics['train_acc'], self.metrics['val_acc'])]
        axes[1, 1].plot(acc_diff)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Training curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Quantization-Aware Training')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenetv2'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet'],
                        help='Dataset name')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='multistep',
                        choices=['multistep', 'cosine', 'exponential'],
                        help='LR scheduler')
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75],
                        help='LR decay milestones')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Gradient clipping (0 to disable)')
    
    # QAT
    parser.add_argument('--qat-mode', action='store_true',
                        help='Enable Quantization-Aware Training')
    parser.add_argument('--backend', type=str, default='fbgemm',
                        choices=['fbgemm', 'qnnpack'],
                        help='Quantization backend')
    parser.add_argument('--freeze-bn-epoch', type=int, default=50,
                        help='Freeze BN stats at this epoch')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--print-freq', type=int, default=50,
                        help='Print frequency')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Train
    trainer = QATTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
