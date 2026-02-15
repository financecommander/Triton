"""
CIFAR-10 Training Script
Production-ready training for ternary/quantized models on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from models.resnet18.ternary_resnet18 import ternary_resnet18
    from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
except ImportError:
    print("Warning: Using mock models")
    def ternary_resnet18(num_classes=10):
        return torchvision.models.resnet18(num_classes=num_classes)
    def ternary_mobilenet_v2(num_classes=10):
        return torchvision.models.mobilenet_v2(num_classes=num_classes)


class CutMix:
    """CutMix data augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch.size(0))
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch.size(), lam)
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        targets_a = targets
        targets_b = targets[rand_index]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        
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


def train_epoch(model, train_loader, criterion, optimizer, device, args, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    cutmix = CutMix(alpha=1.0) if args.cutmix else None
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply CutMix
        if cutmix and np.random.random() < args.cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % args.print_freq == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
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
    
    return running_loss / len(val_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'mobilenetv2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='multistep')
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75])
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--cutmix-prob', type=float, default=0.5)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--save-dir', type=str, default='./checkpoints_cifar10')
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    if args.model == 'resnet18':
        model = ternary_resnet18(num_classes=10)
    else:
        model = ternary_mobilenet_v2(num_classes=10)
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('='*60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, save_dir / 'model_best.pth')
            print(f'✓ Best model saved: {best_acc:.2f}%')
    
    print(f'\nTraining complete! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    import numpy as np
    main()
