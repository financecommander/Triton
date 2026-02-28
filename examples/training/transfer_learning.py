#!/usr/bin/env python3
"""
Transfer Learning Script
=========================
Fine-tune pretrained models on custom datasets with various strategies.

Features:
- Load pretrained models (ImageNet, custom)
- Flexible layer freezing/unfreezing strategies
- Discriminative learning rates
- Progressive unfreezing
- Custom dataset support
- Advanced learning rate schedulers
- Comprehensive monitoring and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import argparse
import logging
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from models.resnet18.ternary_resnet18 import ternary_resnet18
    from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
except ImportError:
    logging.warning("Could not import Triton models. Using torchvision models.")
    def ternary_resnet18(num_classes=1000):
        return torchvision.models.resnet18(num_classes=num_classes)
    def ternary_mobilenet_v2(num_classes=1000):
        return torchvision.models.mobilenet_v2(num_classes=num_classes)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataset(Dataset):
    """Custom dataset for transfer learning"""
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load images and labels
        self.samples = []
        self.classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = self.data_path / class_name
            for img_path in class_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        logging.info(f'Loaded {len(self.samples)} samples from {len(self.classes)} classes')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TransferLearningModel(nn.Module):
    """Wrapper for transfer learning with flexible architecture modification"""
    
    def __init__(self, base_model: str, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.base_model_name = base_model
        
        # Load base model
        if base_model == 'resnet18':
            if pretrained:
                self.backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            else:
                self.backbone = ternary_resnet18(num_classes=1000)
            
            # Replace classifier
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif base_model == 'mobilenetv2':
            if pretrained:
                self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            else:
                self.backbone = ternary_mobilenet_v2(num_classes=1000)
            
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        elif base_model == 'resnet50':
            self.backbone = torchvision.models.resnet50(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        elif base_model == 'efficientnet_b0':
            self.backbone = torchvision.models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {base_model}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        if self.base_model_name in ['resnet18', 'resnet50']:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.base_model_name in ['mobilenetv2', 'efficientnet_b0']:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        logging.info('Backbone frozen, classifier trainable')
    
    def unfreeze_backbone(self, layers: Optional[List[str]] = None):
        """Unfreeze backbone layers"""
        if layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            logging.info('All layers unfrozen')
        else:
            # Unfreeze specific layers
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layers):
                    param.requires_grad = True
            logging.info(f'Unfrozen layers: {layers}')
    
    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """Get parameter groups for discriminative learning rates"""
        if self.base_model_name in ['resnet18', 'resnet50']:
            groups = [
                list(self.backbone.layer1.parameters()),
                list(self.backbone.layer2.parameters()),
                list(self.backbone.layer3.parameters()),
                list(self.backbone.layer4.parameters()),
                list(self.backbone.fc.parameters())
            ]
        elif self.base_model_name == 'mobilenetv2':
            features = list(self.backbone.features.parameters())
            mid = len(features) // 2
            groups = [
                features[:mid],
                features[mid:],
                list(self.backbone.classifier.parameters())
            ]
        else:
            # Default: split into 3 groups
            params = list(self.backbone.parameters())
            n = len(params)
            groups = [
                params[:n//3],
                params[n//3:2*n//3],
                params[2*n//3:]
            ]
        
        return groups


def get_transforms(args) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms"""
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms
    train_transforms = [
        transforms.RandomResizedCrop(args.img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    if args.color_jitter:
        train_transforms.append(
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        )
    
    if args.auto_augment:
        train_transforms.append(transforms.AutoAugment())
    
    train_transforms.extend([
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return transforms.Compose(train_transforms), val_transforms


def get_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders"""
    
    train_transform, val_transform = get_transforms(args)
    
    # Load datasets
    if args.dataset_format == 'imagefolder':
        train_dataset = ImageFolder(
            root=os.path.join(args.data_path, 'train'),
            transform=train_transform
        )
        val_dataset = ImageFolder(
            root=os.path.join(args.data_path, 'val'),
            transform=val_transform
        )
    else:
        train_dataset = CustomDataset(
            data_path=os.path.join(args.data_path, 'train'),
            transform=train_transform
        )
        val_dataset = CustomDataset(
            data_path=os.path.join(args.data_path, 'val'),
            transform=val_transform
        )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    return train_loader, val_loader


def get_optimizer(model: TransferLearningModel, args) -> optim.Optimizer:
    """Get optimizer with optional discriminative learning rates"""
    
    if args.discriminative_lr:
        # Discriminative learning rates: lower layers get smaller LR
        layer_groups = model.get_layer_groups()
        params = []
        
        for i, group in enumerate(layer_groups):
            # Learning rate decreases for earlier layers
            lr_mult = args.lr_decay ** (len(layer_groups) - i - 1)
            params.append({
                'params': group,
                'lr': args.lr * lr_mult
            })
        
        logging.info(f'Using discriminative LR with {len(layer_groups)} groups')
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            params,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    return optimizer


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion, optimizer,
               scaler: Optional[GradScaler], epoch: int, args) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Measure accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100. * correct / targets.size(0)
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        top1.update(accuracy, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if i % args.print_freq == 0:
            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc {top1.val:.2f} ({top1.avg:.2f})'
            )
    
    return {
        'loss': losses.avg,
        'accuracy': top1.avg,
        'batch_time': batch_time.avg
    }


def validate(model: nn.Module, val_loader: DataLoader, criterion, args) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / targets.size(0)
            
            losses.update(loss.item(), images.size(0))
            top1.update(accuracy, images.size(0))
    
    logging.info(f'Validation: Loss {losses.avg:.4f} Acc {top1.avg:.2f}')
    
    return {
        'loss': losses.avg,
        'accuracy': top1.avg
    }


def progressive_unfreezing(model: TransferLearningModel, epoch: int, 
                          unfreeze_schedule: List[Tuple[int, List[str]]]):
    """Progressively unfreeze layers based on schedule"""
    for unfreeze_epoch, layers in unfreeze_schedule:
        if epoch == unfreeze_epoch:
            model.unfreeze_backbone(layers if layers else None)
            logging.info(f'Epoch {epoch}: Unfroze layers {layers if layers else "all"}')


def save_checkpoint(state: dict, is_best: bool, save_dir: Path, 
                   filename: str = 'checkpoint.pth'):
    """Save checkpoint"""
    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = save_dir / 'model_best.pth'
        torch.save(state, best_path)


def load_pretrained_weights(model: nn.Module, weights_path: str):
    """Load custom pretrained weights"""
    logging.info(f'Loading pretrained weights from {weights_path}')
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle different naming conventions
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        # Remove module. prefix if present
        k = k.replace('module.', '')
        
        # Only load matching keys
        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict[k] = v
        else:
            logging.debug(f'Skipping {k}: shape mismatch or not in model')
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    logging.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} layers')


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--dataset-format', type=str, default='imagefolder',
                       choices=['imagefolder', 'custom'], help='Dataset format')
    parser.add_argument('--num-classes', type=int, required=True, 
                       help='Number of classes in target dataset')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'mobilenetv2', 'efficientnet_b0'],
                       help='Base model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--pretrained-path', type=str, default='',
                       help='Path to custom pretrained weights')
    
    # Transfer learning strategy
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone initially')
    parser.add_argument('--progressive-unfreeze', action='store_true',
                       help='Use progressive unfreezing')
    parser.add_argument('--unfreeze-epoch', type=int, default=10,
                       help='Epoch to unfreeze all layers (if progressive unfreezing)')
    parser.add_argument('--discriminative-lr', action='store_true',
                       help='Use discriminative learning rates')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                       help='LR decay factor for earlier layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamw'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='Gradient clipping')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'onecycle'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # Augmentation
    parser.add_argument('--color-jitter', action='store_true', help='Use color jitter')
    parser.add_argument('--auto-augment', action='store_true', help='Use AutoAugment')
    
    # Optimization
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='./checkpoints_transfer',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume')
    
    # Logging
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    
    args = parser.parse_args()
    
    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = save_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f'Starting transfer learning with args: {args}')
    
    # Create model
    model = TransferLearningModel(
        base_model=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    
    # Load custom pretrained weights if provided
    if args.pretrained_path:
        load_pretrained_weights(model, args.pretrained_path)
    
    model = model.cuda()
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(args)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    optimizer = get_optimizer(model, args)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, 
            steps_per_epoch=len(train_loader)
        )
    
    # AMP scaler
    scaler = GradScaler() if args.amp else None
    
    # Progressive unfreezing schedule
    unfreeze_schedule = []
    if args.progressive_unfreeze:
        # Unfreeze in stages
        unfreeze_schedule = [
            (args.unfreeze_epoch // 2, ['layer4', 'classifier']),  # Unfreeze last layers
            (args.unfreeze_epoch, None)  # Unfreeze all
        ]
    
    # Training loop
    best_acc = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        # Progressive unfreezing
        if unfreeze_schedule:
            progressive_unfreezing(model, epoch, unfreeze_schedule)
        
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                    scaler, epoch, args)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, args)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['accuracy'])
        elif args.scheduler != 'onecycle':
            if epoch >= args.warmup_epochs:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_acc
        best_acc = max(val_metrics['accuracy'], best_acc)
        
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Save training history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump({'train': train_history, 'val': val_history}, f, indent=2)
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'args': vars(args)
        }, is_best, save_dir, 'checkpoint_latest.pth')
        
        logging.info(f'Epoch {epoch}: Train Acc {train_metrics["accuracy"]:.2f}% '
                    f'Val Acc {val_metrics["accuracy"]:.2f}% Best {best_acc:.2f}%')
    
    logging.info(f'Training complete! Best Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
