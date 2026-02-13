"""
Training Script for Ternary Neural Networks

Trains ResNet-18 and MobileNetV2 with ternary quantization on ImageNet or CIFAR-10.
Uses the Triton GPU backend for optimized ternary operations.

Usage:
    python train_ternary_models.py --model resnet18 --dataset imagenet --batch_size 256
    python train_ternary_models.py --model mobilenetv2 --dataset cifar10 --batch_size 128
"""

import argparse
import os
import time
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


def get_dataset(dataset_name: str, train: bool = True):
    """Get dataset with appropriate transforms."""
    if dataset_name.lower() == 'cifar10':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval=100):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Quantize weights periodically during training
        if batch_idx % 100 == 0:
            quantize_model_weights(model)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, epoch_time


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


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train Ternary Neural Networks')
    parser.add_argument('--model', type=str, choices=['resnet18', 'mobilenetv2'],
                       default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'],
                       default='cifar10', help='Dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, num_classes = get_dataset(args.dataset, train=True)
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

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_acc = 0.0
    print("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Print results
        print(f'Epoch {epoch + 1}/{args.epochs}: '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')

        # Save checkpoint
        checkpoint_name = f"{args.model}_{args.dataset}_epoch_{epoch + 1}_acc_{val_acc:.2f}.pth"
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = os.path.join(args.checkpoint_dir, f"{args.model}_{args.dataset}_best.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_checkpoint)
            print(f"New best accuracy: {best_acc:.2f}%")

    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()