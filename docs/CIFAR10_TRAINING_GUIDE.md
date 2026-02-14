# CIFAR-10 Training Guide for 500-Epoch Runs

This guide explains how to train ternary neural networks on CIFAR-10 for extended training runs (up to 500 epochs) with advanced features like early stopping, enhanced data augmentation, and comprehensive logging.

## Table of Contents
- [Quick Start](#quick-start)
- [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Advanced Features](#advanced-features)
- [Command-Line Options](#command-line-options)
- [Expected Results](#expected-results)
- [Tips for Long Training](#tips-for-long-training)

## Quick Start

### Fresh Training Run (500 epochs)

```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --cutmix_alpha 1.0 \
    --save_freq 10 \
    --log_interval 100
```

This will:
- Train ResNet-18 on CIFAR-10 for up to 500 epochs
- Use batch size 128 (adjust based on your GPU memory)
- Apply cosine annealing LR schedule (default)
- Enable early stopping (stops if no improvement for 40 epochs)
- Use label smoothing (0.1) to prevent overfitting
- Apply CutMix augmentation for better regularization
- Save checkpoints every 10 epochs
- Log metrics to TensorBoard and CSV

## Resuming from Checkpoint

If you're at epoch 76 and want to continue to 500 epochs:

```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume ./checkpoints/resnet18_cifar10_epoch_76.pth \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 10
```

**Important Notes:**
- The script will automatically resume from epoch 77
- Optimizer state, scheduler state, and early stopping counters are all restored
- Best accuracy tracking continues from the checkpoint
- Learning rate will continue from where it left off with cosine annealing

## Advanced Features

### 1. Early Stopping

Automatically stops training if validation accuracy doesn't improve for N epochs:

```bash
--early_stopping \
--early_stopping_patience 40
```

This prevents wasting compute time if the model plateaus.

### 2. Data Augmentation

#### CutMix
Randomly cuts and pastes patches between images. Excellent for regularization:

```bash
--cutmix \
--cutmix_alpha 1.0
```

#### MixUp
Blends two images together. Can be used with or without CutMix:

```bash
--mixup \
--mixup_alpha 1.0
```

#### AutoAugment (CIFAR-10 Policy)
Advanced augmentation policy learned from data:

```bash
--autoaugment
```

#### RandAugment
Random augmentation with configurable strength:

```bash
--randaugment
```

**Recommendation:** For CIFAR-10, use either `--cutmix` or both `--cutmix --mixup`. Avoid combining with AutoAugment/RandAugment initially.

### 3. Label Smoothing

Prevents overconfident predictions and improves generalization:

```bash
--label_smoothing 0.1
```

Typical values: 0.05 to 0.15. Start with 0.1.

### 4. Learning Rate Scheduling

#### Cosine Annealing (Default)
Smoothly reduces LR following a cosine curve:

```bash
--scheduler cosine
```

This is the recommended scheduler for long training runs.

#### Step Decay
Reduces LR by a factor at fixed intervals:

```bash
--scheduler step \
--step_size 150 \
--gamma 0.1
```

#### No Scheduler
Keep learning rate constant (not recommended for 500 epochs):

```bash
--scheduler none
```

### 5. Logging

#### TensorBoard
Automatically enabled if tensorboard is installed:

```bash
# View logs with:
tensorboard --logdir ./logs
```

Metrics logged:
- Train/validation loss and accuracy (per epoch and batch)
- Learning rate schedule
- Best accuracy over time

#### CSV Logging
All metrics are automatically saved to CSV:

```bash
--csv_log ./results/my_experiment.csv
```

Default location: `./logs/{model}_{dataset}_metrics.csv`

### 6. Checkpointing

#### Periodic Checkpoints
Save every N epochs:

```bash
--save_freq 10
```

#### Best Model
Automatically saved to `{model}_{dataset}_best.pth` when validation accuracy improves.

#### Checkpoint Contents
Each checkpoint includes:
- Model state (weights)
- Optimizer state
- Scheduler state
- Current epoch
- Best accuracy so far
- Early stopping counter

## Command-Line Options

### Model and Dataset
```
--model {resnet18, mobilenetv2}    Model architecture (default: resnet18)
--dataset {cifar10, imagenet}      Dataset (default: cifar10)
```

### Training Hyperparameters
```
--batch_size INT                   Batch size (default: 128)
--epochs INT                       Total epochs (default: 100)
--lr FLOAT                         Initial learning rate (default: 0.1)
--momentum FLOAT                   SGD momentum (default: 0.9)
--weight_decay FLOAT               Weight decay (default: 5e-4)
```

### Learning Rate Scheduling
```
--scheduler {cosine, step, none}   LR scheduler (default: cosine)
--step_size INT                    Step size for StepLR (default: 30)
--gamma FLOAT                      Gamma for StepLR (default: 0.1)
```

### Regularization
```
--label_smoothing FLOAT            Label smoothing (default: 0.0)
```

### Data Augmentation
```
--cutmix                           Enable CutMix
--cutmix_alpha FLOAT               CutMix alpha (default: 1.0)
--mixup                            Enable MixUp
--mixup_alpha FLOAT                MixUp alpha (default: 1.0)
--autoaugment                      Enable AutoAugment
--randaugment                      Enable RandAugment
```

### Checkpointing
```
--checkpoint_dir PATH              Checkpoint directory (default: ./checkpoints)
--resume PATH                      Resume from checkpoint
--save_freq INT                    Save every N epochs (default: 10)
```

### Early Stopping
```
--early_stopping                   Enable early stopping
--early_stopping_patience INT      Patience in epochs (default: 40)
```

### Logging
```
--log_dir PATH                     TensorBoard log directory (default: ./logs)
--csv_log PATH                     CSV log file path
--log_interval INT                 Batch logging interval (default: 100)
```

### Other
```
--workers INT                      Data loading workers (default: 4)
--seed INT                         Random seed for reproducibility
```

## Expected Results

### ResNet-18 on CIFAR-10

With standard augmentation (RandomCrop + RandomHorizontalFlip):
- **100 epochs**: 85-87% validation accuracy
- **200 epochs**: 87-89% validation accuracy  
- **500 epochs**: 88-90% validation accuracy

With enhanced augmentation (CutMix + Label Smoothing):
- **100 epochs**: 86-88% validation accuracy
- **200 epochs**: 88-90% validation accuracy
- **500 epochs**: 90-92% validation accuracy

**Note:** Ternary networks typically achieve 2-3% lower accuracy than FP32 models but with 32x memory reduction.

### Training Time Estimates

On a modern GPU (e.g., RTX 3090):
- **Epoch time**: ~60-70 seconds (batch_size=128)
- **100 epochs**: ~1.7 hours
- **500 epochs**: ~8.5 hours

Adjust batch size based on your GPU memory:
- 4GB VRAM: batch_size=64
- 8GB VRAM: batch_size=128  
- 12GB+ VRAM: batch_size=256

## Tips for Long Training

### 1. Start with Conservative Settings

For your first 500-epoch run:
```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix
```

### 2. Monitor Training Progress

Check TensorBoard regularly:
```bash
tensorboard --logdir ./logs --port 6006
```

Look for:
- **Overfitting**: Train acc >> Val acc → Increase regularization
- **Underfitting**: Both accuracies plateau early → Increase model capacity or LR
- **Instability**: Large fluctuations → Reduce LR or increase batch size

### 3. Adjust Regularization

If you see overfitting (train acc - val acc > 5%):
- Increase `--label_smoothing` to 0.15
- Add `--mixup` if not already using it
- Increase `--weight_decay` to 1e-3
- Try `--autoaugment` or `--randaugment`

### 4. Fine-tune Learning Rate

If training stalls:
- The cosine schedule should handle this automatically
- For manual control, reduce `--lr` to 0.05 or 0.01
- Consider warmup (requires code modification)

### 5. Use Early Stopping Wisely

- **patience=40** is good for 500 epochs (8% of total)
- For shorter runs (100-200 epochs), use patience=15-20
- Monitor the early stopping counter in logs

### 6. Checkpoint Management

With `--save_freq 10`, you'll have ~50 checkpoints for 500 epochs. To save space:
- Keep only the best checkpoint and latest N checkpoints
- Delete intermediate checkpoints manually
- Consider increasing `--save_freq` to 20 or 50

### 7. Reproducibility

For reproducible results:
```bash
--seed 42
```

Note: Some non-determinism may remain due to CUDA operations.

## Example Workflow

### Scenario: Resume from Epoch 76, Target 500 Epochs

**Current State:**
- Epoch: 76/300
- Train Acc: 86.21%, Val Acc: 85.54%
- Train Loss: 0.3994, Val Loss: 0.6352
- LR: 0.000850
- Epoch Time: ~61 seconds

**Recommended Command:**

```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume ./checkpoints/resnet18_cifar10_epoch_76.pth \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --cutmix_alpha 1.0 \
    --save_freq 20 \
    --csv_log ./results/500epoch_run.csv
```

**What to Expect:**
1. Training resumes from epoch 77
2. Cosine annealing LR continues from 0.000850
3. With early stopping, training may stop around epoch 350-450 if it plateaus
4. Final validation accuracy: 88-91% (estimated)
5. Total training time: ~7-8 hours on modern GPU

**Monitoring:**
```bash
# Terminal 1: Training
python models/scripts/train_ternary_models.py ...

# Terminal 2: TensorBoard
tensorboard --logdir ./logs

# Terminal 3: Watch metrics in real-time
tail -f ./results/500epoch_run.csv
```

## Troubleshooting

### Training is Too Slow
- Reduce `--workers` if CPU is bottleneck
- Increase `--batch_size` if GPU memory allows
- Check GPU utilization: `nvidia-smi -l 1`

### Out of Memory
- Reduce `--batch_size` (128 → 64 → 32)
- Reduce `--workers` (4 → 2 → 1)
- Use gradient accumulation (requires code modification)

### Validation Accuracy Dropping
- Reduce learning rate manually
- Increase regularization (label smoothing, weight decay)
- Check for data loading errors

### Early Stopping Too Aggressive
- Increase `--early_stopping_patience` (40 → 60 → 80)
- Check if validation accuracy is actually plateauing

## Advanced: Custom Modifications

For advanced users who want to modify the training script:

### Add Gradient Clipping
In `train_epoch()`, after `loss.backward()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Add Learning Rate Warmup
Before the main loop in `main()`:
```python
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
```

### Add Multi-GPU Support
Replace model initialization:
```python
model = torch.nn.DataParallel(get_model(args.model, num_classes))
model = model.to(device)
```

## References

- Original training script: `models/scripts/train_ternary_models.py`
- Model architectures: `models/resnet18/ternary_resnet18.py`
- CutMix paper: https://arxiv.org/abs/1905.04899
- Label Smoothing: https://arxiv.org/abs/1906.02629
- Cosine Annealing: https://arxiv.org/abs/1608.03983

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review TensorBoard logs and CSV metrics
3. Enable debug logging: add `--log_interval 10` for more frequent updates
4. Check GPU memory usage: `nvidia-smi`
