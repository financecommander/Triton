# Tutorial 4: Advanced Training Techniques

Master advanced training techniques for ternary neural networks, including distributed training, mixed precision, learning rate schedules, and specialized optimization strategies for quantized models.

## Learning Objectives

- Implement distributed training for ternary models
- Use mixed precision training effectively
- Apply advanced learning rate schedules
- Implement data augmentation for ternary models
- Set up monitoring and logging infrastructure
- Use early stopping and checkpointing strategies
- Optimize training performance and convergence

## Prerequisites

- Completed [Tutorial 1](01_basic_model.md), [Tutorial 2](02_quantization.md), and [Tutorial 3](03_custom_layers.md)
- Understanding of training dynamics
- Familiarity with PyTorch training loops
- Basic knowledge of distributed systems (for distributed training)

## Training Challenges with Ternary Models

### The Quantization Gap

Ternary models face unique challenges:

```
Challenge 1: Limited representational capacity
  → Solution: Careful initialization and learning rates

Challenge 2: Gradient mismatch (STE approximation)
  → Solution: Adaptive learning rates and warm-up

Challenge 3: Training instability
  → Solution: Gradient clipping and normalization

Challenge 4: Convergence speed
  → Solution: Knowledge distillation and progressive training
```

### Key Principles

1. **Start with FP32**: Train full-precision baseline first
2. **Progressive Quantization**: Gradually introduce ternary constraints
3. **Patience**: Ternary models need more epochs
4. **Monitoring**: Track both FP32 and quantized metrics

## Distributed Training

### Single-Node Multi-GPU Training

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class TernaryTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def setup_distributed(self, rank, world_size):
        """Initialize distributed training environment."""
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU
            rank=rank,
            world_size=world_size
        )
        
    def cleanup_distributed(self):
        """Clean up distributed training."""
        dist.destroy_process_group()
        
    def train_distributed(self, rank, world_size):
        """Training function for each process."""
        print(f"Running on rank {rank}/{world_size}")
        
        # Setup
        self.setup_distributed(rank, world_size)
        
        # Move model to GPU
        device = torch.device(f'cuda:{rank}')
        self.model = self.model.to(device)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False  # Set True if needed
        )
        
        # Create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # DataLoader with sampler
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Set epoch for sampler (ensures different shuffle each epoch)
            train_sampler.set_epoch(epoch)
            
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (important for ternary models)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                if rank == 0 and batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Synchronize before validation
            dist.barrier()
            
            # Validation (only on rank 0)
            if rank == 0:
                self.validate(device)
        
        # Cleanup
        self.cleanup_distributed()

def launch_distributed_training(model, config):
    """Launch distributed training."""
    world_size = torch.cuda.device_count()
    mp.spawn(
        TernaryTrainer(model, config).train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### Multi-Node Training

```python
# For multi-node training, set environment variables:
# MASTER_ADDR=<master-node-ip>
# MASTER_PORT=<port>
# WORLD_SIZE=<total-gpus>
# RANK=<global-rank>

def setup_multinode_distributed():
    """Setup for multi-node training."""
    import os
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

# Launch with torchrun:
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
#          --master_addr=<master-ip> --master_port=29500 \
#          train.py
```

### Gradient Accumulation

For larger effective batch sizes without more memory:

```python
class GradientAccumulationTrainer:
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        
    def train_with_accumulation(self, train_loader, optimizer, criterion):
        """Train with gradient accumulation."""
        self.model.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Normalize loss for accumulation
                loss = loss / self.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    print(f'Batch {batch_idx}, Loss: {loss.item() * self.accumulation_steps:.4f}')
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTernaryTrainer:
    """Mixed precision training for ternary models."""
    
    def __init__(self, model):
        self.model = model
        self.scaler = GradScaler()
        
    def train_amp(self, train_loader, optimizer, criterion, epochs):
        """Train with automatic mixed precision."""
        self.model.train()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                
                # Forward pass with autocast
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                # Optimizer step with scaling
                self.scaler.step(optimizer)
                self.scaler.update()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Manual Mixed Precision

```python
class ManualMixedPrecision:
    """Manual control over precision for different parts."""
    
    def __init__(self, model):
        self.model = model
        self.convert_to_fp16()
        
    def convert_to_fp16(self):
        """Convert appropriate layers to FP16."""
        for name, module in self.model.named_modules():
            # Keep batch norm and layer norm in FP32
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                continue
            
            # Convert other layers to FP16
            if hasattr(module, 'weight') and module.weight is not None:
                module.half()
                
    def forward_mixed(self, x):
        """Forward pass with mixed precision."""
        # Input in FP32
        x = x.float()
        
        # Process through FP16 layers
        for layer in self.model.layers:
            if isinstance(layer, torch.nn.BatchNorm2d):
                x = layer(x.float())  # BN in FP32
            else:
                x = layer(x.half())   # Other layers in FP16
        
        # Output in FP32
        return x.float()
```

### Best Practices for AMP with Ternary Models

```python
class TernaryAMPBestPractices:
    """Best practices for AMP with ternary models."""
    
    @staticmethod
    def setup_optimizer_for_amp(model, lr=0.001):
        """Setup optimizer compatible with AMP."""
        # Separate parameters by type
        ternary_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'ternary' in name or 'quantized' in name:
                ternary_params.append(param)
            else:
                other_params.append(param)
        
        # Use different learning rates
        optimizer = torch.optim.Adam([
            {'params': ternary_params, 'lr': lr * 0.1},  # Lower LR for ternary
            {'params': other_params, 'lr': lr}
        ])
        
        return optimizer
    
    @staticmethod
    def loss_scaling_strategy(loss, scale_factor=1.0):
        """Custom loss scaling for ternary models."""
        # Scale loss to prevent underflow in FP16
        return loss * scale_factor
```

## Learning Rate Schedules

### Cosine Annealing with Warm Restarts

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineScheduler:
    """Cosine annealing with warm restarts."""
    
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6):
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,        # Epochs until first restart
            T_mult=T_mult,  # Factor to increase T_0 after each restart
            eta_min=eta_min # Minimum learning rate
        )
        
    def step(self):
        """Step the scheduler."""
        self.scheduler.step()
        
    def get_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineScheduler(optimizer, T_0=10, T_mult=2)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()
    print(f'Epoch {epoch}, LR: {scheduler.get_lr():.6f}')
```

### One Cycle Policy

```python
from torch.optim.lr_scheduler import OneCycleLR

class OneCycleTrainer:
    """Training with One Cycle learning rate policy."""
    
    def __init__(self, model, train_loader, epochs, max_lr=0.01):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=max_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Calculate total steps
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        
        # Create scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,      # Percentage of cycle spent increasing LR
            anneal_strategy='cos',
            div_factor=25.0,    # Initial LR = max_lr / div_factor
            final_div_factor=1e4  # Final LR = max_lr / final_div_factor
        )
        
    def train(self):
        """Train with One Cycle policy."""
        for epoch in range(self.epochs):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.cuda(), target.cuda()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Step scheduler after each batch
                self.scheduler.step()
                
                if batch_idx % 100 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
```

### Custom Warmup Schedule

```python
class WarmupScheduler:
    """Custom warmup followed by decay."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 base_lr, max_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupScheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=100,
    base_lr=1e-6,
    max_lr=0.001
)

for epoch in range(100):
    lr = scheduler.step()
    train_epoch(model, optimizer)
    print(f'Epoch {epoch}, LR: {lr:.6f}')
```

## Data Augmentation for Ternary Models

### Standard Augmentation

```python
import torchvision.transforms as transforms

class TernaryDataAugmentation:
    """Data augmentation strategies for ternary models."""
    
    @staticmethod
    def get_training_transforms(image_size=32):
        """Get training augmentation pipeline."""
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_test_transforms(image_size=32):
        """Get test transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

### Advanced Augmentation

```python
from torchvision.transforms import autoaugment, RandAugment

class AdvancedAugmentation:
    """Advanced augmentation techniques."""
    
    @staticmethod
    def get_autoaugment():
        """AutoAugment for CIFAR10."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            autoaugment.AutoAugment(
                autoaugment.AutoAugmentPolicy.CIFAR10
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_randaugment(n=2, m=9):
        """RandAugment with n operations and magnitude m."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=n, m=m),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

### Mixup and CutMix

```python
import numpy as np

class MixupCutmix:
    """Mixup and CutMix augmentation."""
    
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        
    def mixup(self, x, y):
        """Apply mixup augmentation."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(self, x, y):
        """Apply CutMix augmentation."""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()
        
        # Get random box
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def __call__(self, x, y):
        """Apply mixup or cutmix randomly."""
        if np.random.rand() > self.prob:
            return x, y, None, 1.0
        
        if np.random.rand() < 0.5:
            return self.mixup(x, y)
        else:
            return self.cutmix(x, y)

# Usage with custom loss
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Criterion for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## Monitoring and Logging

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter
import time

class TensorBoardLogger:
    """TensorBoard logging for ternary model training."""
    
    def __init__(self, log_dir='runs'):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        
    def log_scalar(self, tag, value, step=None):
        """Log scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        """Log multiple scalars."""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_histogram(self, tag, values, step=None):
        """Log histogram of values."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_model_weights(self, model, step=None):
        """Log all model weights as histograms."""
        if step is None:
            step = self.step
            
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_ternary_statistics(self, model, step=None):
        """Log ternary-specific statistics."""
        if step is None:
            step = self.step
            
        for name, param in model.named_parameters():
            if 'ternary' in name or 'quantized' in name:
                # Count distribution of ternary values
                values = param.detach().cpu()
                zeros = (values == 0).sum().item()
                ones = (values == 1).sum().item()
                neg_ones = (values == -1).sum().item()
                total = values.numel()
                
                self.writer.add_scalars(f'ternary_dist/{name}', {
                    'zeros': zeros / total,
                    'ones': ones / total,
                    'neg_ones': neg_ones / total
                }, step)
    
    def log_learning_rates(self, optimizer, step=None):
        """Log learning rates for all parameter groups."""
        if step is None:
            step = self.step
            
        for idx, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(
                f'learning_rate/group_{idx}',
                param_group['lr'],
                step
            )
    
    def close(self):
        """Close the writer."""
        self.writer.close()

# Usage
logger = TensorBoardLogger(log_dir='runs/ternary_training')

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        loss = train_step(model, data, target, optimizer)
        
        # Log metrics
        logger.log_scalar('train/loss', loss, logger.step)
        logger.log_learning_rates(optimizer, logger.step)
        
        if logger.step % 100 == 0:
            logger.log_model_weights(model, logger.step)
            logger.log_ternary_statistics(model, logger.step)
        
        logger.step += 1
    
    # Validation
    val_loss, val_acc = validate(model, val_loader)
    logger.log_scalars('validation', {
        'loss': val_loss,
        'accuracy': val_acc
    }, logger.step)

logger.close()
```

### Weights & Biases Integration

```python
import wandb

class WandBLogger:
    """Weights & Biases logging."""
    
    def __init__(self, project, config):
        wandb.init(project=project, config=config)
        self.config = config
        
    def log(self, metrics, step=None):
        """Log metrics."""
        wandb.log(metrics, step=step)
        
    def watch_model(self, model, log_freq=100):
        """Watch model parameters."""
        wandb.watch(model, log='all', log_freq=log_freq)
        
    def log_ternary_table(self, model):
        """Log ternary weight distribution as table."""
        data = []
        for name, param in model.named_parameters():
            if 'ternary' in name:
                values = param.detach().cpu()
                zeros = (values == 0).float().mean().item()
                ones = (values == 1).float().mean().item()
                neg_ones = (values == -1).float().mean().item()
                
                data.append([name, zeros, ones, neg_ones])
        
        table = wandb.Table(
            columns=['Layer', 'Zeros %', 'Ones %', 'Neg Ones %'],
            data=data
        )
        wandb.log({'ternary_distribution': table})
    
    def finish(self):
        """Finish logging."""
        wandb.finish()

# Usage
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 100,
    'model': 'TernaryResNet',
}

logger = WandBLogger(project='ternary-training', config=config)
logger.watch_model(model, log_freq=100)

for epoch in range(config['epochs']):
    metrics = train_epoch(model, train_loader, optimizer)
    logger.log(metrics, step=epoch)
    
    if epoch % 10 == 0:
        logger.log_ternary_table(model)

logger.finish()
```

## Early Stopping and Checkpointing

### Early Stopping

```python
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, score):
        """Check if should stop."""
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print(f'Early stopping at epoch {epoch}')
        break
```

### Model Checkpointing

```python
import os
from pathlib import Path

class ModelCheckpoint:
    """Save model checkpoints."""
    
    def __init__(self, directory, mode='min', save_best_only=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
        else:
            self.monitor_op = lambda x, y: x > y
    
    def save_checkpoint(self, model, optimizer, epoch, score, 
                       scheduler=None, filename=None):
        """Save checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = self.directory / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        return filepath
    
    def __call__(self, model, optimizer, epoch, score, scheduler=None):
        """Check and save if better."""
        should_save = False
        
        if self.best_score is None:
            self.best_score = score
            should_save = True
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            should_save = True
        
        if should_save or not self.save_best_only:
            # Save checkpoint
            filename = f'checkpoint_epoch_{epoch}.pth'
            self.save_checkpoint(model, optimizer, epoch, score, 
                               scheduler, filename)
            
            # Save best model separately
            if should_save:
                best_path = self.directory / 'best_model.pth'
                torch.save(model.state_dict(), best_path)
                print(f'Saved best model with score: {score:.4f}')
    
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
        """Load checkpoint."""
        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['score']

# Usage
checkpoint = ModelCheckpoint(
    directory='checkpoints/ternary_model',
    mode='min',
    save_best_only=True
)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Save checkpoint
    checkpoint(model, optimizer, epoch, val_loss, scheduler)

# Load best model
checkpoint.load_checkpoint(
    'checkpoints/ternary_model/best_model.pth',
    model
)
```

## Complete Training Pipeline

```python
class CompleteTernaryTrainer:
    """Complete training pipeline with all features."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = self.setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self.setup_scheduler()
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup augmentation
        self.augmentation = MixupCutmix(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha
        )
        
        # Setup logging
        self.logger = TensorBoardLogger(config.log_dir)
        
        # Setup checkpointing
        self.checkpoint = ModelCheckpoint(
            config.checkpoint_dir,
            mode='max',
            save_best_only=True
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode='max'
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
    def setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        ternary_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'ternary' in name or 'quantized' in name:
                ternary_params.append(param)
            else:
                other_params.append(param)
        
        return torch.optim.Adam([
            {'params': ternary_params, 'lr': self.config.lr * 0.1},
            {'params': other_params, 'lr': self.config.lr}
        ], weight_decay=self.config.weight_decay)
    
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        steps_per_epoch = len(self.train_loader)
        total_steps = self.config.epochs * steps_per_epoch
        
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr,
            total_steps=total_steps,
            pct_start=0.3
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.cuda(), target.cuda()
            
            # Apply augmentation
            data, target_a, target_b, lam = self.augmentation(data, target)
            
            self.optimizer.zero_grad()
            
            # Forward with AMP
            with autocast():
                output = self.model(data)
                if target_b is not None:
                    loss = mixup_criterion(
                        self.criterion, output, target_a, target_b, lam
                    )
                else:
                    loss = self.criterion(output, target)
            
            # Backward with scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step scheduler
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            if batch_idx % 100 == 0:
                self.logger.log_scalar('train/loss', loss.item(), self.logger.step)
                self.logger.log_learning_rates(self.optimizer, self.logger.step)
            
            self.logger.step += 1
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.cuda(), target.cuda()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.val_loader), 100.0 * correct / total
    
    def train(self):
        """Complete training loop."""
        print('Starting training...')
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            self.logger.log_scalars('metrics', {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, epoch)
            
            # Checkpoint
            self.checkpoint(self.model, self.optimizer, epoch, 
                          val_acc, self.scheduler)
            
            # Early stopping
            if self.early_stopping(val_acc):
                print(f'Early stopping at epoch {epoch}')
                break
        
        self.logger.close()
        print('Training complete!')
```

## Best Practices Summary

1. **Start Simple**: Begin with FP32 baseline, then quantize
2. **Learning Rates**: Use lower LR for ternary parameters
3. **Gradients**: Always clip gradients (max_norm=1.0)
4. **Monitoring**: Track ternary distribution and convergence
5. **Patience**: Ternary models need more epochs
6. **Augmentation**: Essential for small datasets
7. **Checkpointing**: Save best model based on validation
8. **Distributed**: Use DDP for multi-GPU training

## Exercises

1. **Distributed Training**: Modify the provided code to train on 4 GPUs
2. **Custom Schedule**: Implement a custom warmup + exponential decay schedule
3. **Advanced Augmentation**: Add RandAugment to your training pipeline
4. **Monitoring Dashboard**: Set up TensorBoard or W&B logging
5. **Complete Pipeline**: Combine all techniques in one training script

## Next Steps

- [Tutorial 5: Model Deployment](05_deployment.md)
- [Tutorial 6: Advanced Features](06_advanced_features.md)
- [API Reference](../api/README.md)

## Additional Resources

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
