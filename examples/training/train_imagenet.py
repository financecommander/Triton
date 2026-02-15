#!/usr/bin/env python3
"""
Production-Ready ImageNet Training Script
==========================================
Comprehensive training pipeline with distributed training, mixed precision,
advanced augmentations, checkpoint resumption, and model exports.

Features:
- Multi-GPU distributed training (DDP)
- Mixed precision training (AMP)
- Advanced augmentations (RandAugment, MixUp, CutMix)
- Automatic checkpoint resumption
- Model export to ONNX/TorchScript
- Comprehensive logging and metrics
- EMA (Exponential Moving Average) support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment

import argparse
import time
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from models.resnet18.ternary_resnet18 import ternary_resnet18
    from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
except ImportError:
    logging.warning("Could not import Triton models. Using torchvision models as fallback.")
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


class MixUp:
    """MixUp data augmentation"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_batch, targets_a, targets_b, lam


class CutMix:
    """CutMix data augmentation"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch.size(), lam)
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        
        targets_a, targets_b = targets, targets[index]
        return batch, targets_a, targets_b, lam
    
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


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = model
        self.decay = decay
        self.ema = {name: param.clone().detach() 
                    for name, param in model.named_parameters() if param.requires_grad}
    
    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.ema:
                    self.ema[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        backup = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.ema:
                backup[name] = param.data.clone()
                param.data.copy_(self.ema[name])
        return backup
    
    def restore(self, backup):
        """Restore original parameters"""
        for name, param in self.module.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', 
                           world_size=world_size, rank=rank)
    dist.barrier()
    
    return rank, world_size, gpu


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloaders(args, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    """Create ImageNet data loaders with augmentations"""
    
    # Training augmentations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9) if args.randaugment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'train'),
        transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'val'),
        transform=val_transforms
    )
    
    # Samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                          rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, 
                                        rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    return train_loader, val_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion, optimizer, 
                scaler: Optional[GradScaler], epoch: int, args, 
                ema: Optional[ModelEMA] = None, rank: int = 0) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    
    mixup = MixUp(args.mixup_alpha) if args.mixup else None
    cutmix = CutMix(args.cutmix_alpha) if args.cutmix else None
    
    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Apply MixUp or CutMix
        use_mixup = mixup and np.random.rand() < args.mix_prob
        use_cutmix = cutmix and not use_mixup and np.random.rand() < args.mix_prob
        
        if use_mixup:
            images, targets_a, targets_b, lam = mixup(images, targets)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix(images, targets)
        
        # Mixed precision training
        with autocast(enabled=args.amp):
            outputs = model(images)
            
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
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
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if i % args.print_freq == 0 and rank == 0:
            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg,
        'batch_time': batch_time.avg
    }


def validate(model: nn.Module, val_loader: DataLoader, criterion, args, rank: int = 0) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    
    if rank == 0:
        logging.info(f'Validation: Loss {losses.avg:.4f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Compute top-k accuracy"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state: dict, is_best: bool, save_dir: Path, filename: str = 'checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = save_dir / 'model_best.pth'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer=None, 
                    scheduler=None, scaler=None) -> Tuple[int, float]:
    """Load checkpoint and resume training"""
    if not checkpoint_path.exists():
        return 0, 0.0
    
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc1 = checkpoint['best_acc1']
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    logging.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]} with best acc1: {best_acc1:.2f}')
    return start_epoch, best_acc1


def export_model(model: nn.Module, save_dir: Path, input_shape=(1, 3, 224, 224)):
    """Export model to ONNX and TorchScript"""
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(input_shape)
    
    # Export to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, save_dir / 'model.torchscript.pt')
        logging.info(f'Exported TorchScript model to {save_dir / "model.torchscript.pt"}')
    except Exception as e:
        logging.error(f'Failed to export TorchScript: {e}')
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_dir / 'model.onnx',
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logging.info(f'Exported ONNX model to {save_dir / "model.onnx"}')
    except Exception as e:
        logging.error(f'Failed to export ONNX: {e}')


def main():
    parser = argparse.ArgumentParser(description='ImageNet Training')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'mobilenetv2'], help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='Gradient clipping')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'multistep', 'plateau'], help='LR scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # Augmentation
    parser.add_argument('--randaugment', action='store_true', help='Use RandAugment')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='MixUp alpha')
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, help='CutMix alpha')
    parser.add_argument('--mix-prob', type=float, default=0.5, help='Probability of applying mix')
    
    # Optimization
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--ema', action='store_true', help='Use exponential moving average')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_imagenet', 
                       help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Distributed
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    
    # Export
    parser.add_argument('--export', action='store_true', help='Export model after training')
    
    # Logging
    parser.add_argument('--print-freq', type=int, default=100, help='Print frequency')
    parser.add_argument('--log-file', type=str, default='', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = args.log_file or save_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Distributed training setup
    rank = 0
    world_size = 1
    if args.distributed:
        rank, world_size, gpu = setup_distributed()
        args.batch_size = args.batch_size // world_size
    
    if rank == 0:
        logging.info(f'Starting training with args: {args}')
        logging.info(f'World size: {world_size}, Rank: {rank}')
    
    # Create model
    if args.model == 'resnet18':
        model = ternary_resnet18(num_classes=args.num_classes)
    else:
        model = ternary_mobilenet_v2(num_classes=args.num_classes)
    
    model = model.cuda()
    
    if args.distributed:
        model = DDP(model, device_ids=[gpu])
    
    # Create EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, 
                                                          eta_min=args.min_lr)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs * 0.3), int(args.epochs * 0.6), int(args.epochs * 0.9)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
    
    # AMP scaler
    scaler = GradScaler() if args.amp else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(Path(args.resume), model, optimizer, 
                                                 scheduler, scaler)
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(args, rank, world_size)
    
    # Training loop
    train_history = []
    val_history = []
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                    epoch, args, ema, rank)
        
        # Validate
        if ema is not None:
            backup = ema.apply_shadow()
            val_metrics = validate(model, val_loader, criterion, args, rank)
            ema.restore(backup)
        else:
            val_metrics = validate(model, val_loader, criterion, args, rank)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['top1'])
        elif epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['top1'] > best_acc1
        best_acc1 = max(val_metrics['top1'], best_acc1)
        
        if rank == 0:
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Save training history
            with open(save_dir / 'history.json', 'w') as f:
                json.dump({'train': train_history, 'val': val_history}, f, indent=2)
            
            # Save checkpoint
            if epoch % args.save_freq == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'args': vars(args)
                }, is_best, save_dir, f'checkpoint_epoch_{epoch}.pth')
            
            logging.info(f'Epoch {epoch}: Train Acc@1 {train_metrics["top1"]:.2f} '
                        f'Val Acc@1 {val_metrics["top1"]:.2f} Best Acc@1 {best_acc1:.2f}')
    
    # Export model
    if rank == 0 and args.export:
        model_to_export = model.module if args.distributed else model
        if ema is not None:
            ema.apply_shadow()
        export_model(model_to_export, save_dir)
    
    # Cleanup
    if args.distributed:
        cleanup_distributed()
    
    if rank == 0:
        logging.info(f'Training complete! Best Top-1 Accuracy: {best_acc1:.2f}%')


if __name__ == '__main__':
    main()
