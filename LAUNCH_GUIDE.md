# Quick Launch Guide for CIFAR-10 Training

This guide helps you quickly launch your CIFAR-10 training run.

## Prerequisites

Install required dependencies:
```bash
pip install torch torchvision numpy tensorboard matplotlib seaborn
```

Or use the project install:
```bash
pip install -e .
```

## Quick Launch Options

### Option 1: Interactive Launcher (Recommended)

Use the interactive launcher script:
```bash
./launch_training.sh
```

This script will:
- Check dependencies
- Show available checkpoints
- Present training scenarios
- Guide you through the launch

### Option 2: Direct Command

#### Fresh 500-Epoch Training
```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 10
```

#### Resume from Checkpoint (Epoch 76 → 500)
```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume ./checkpoints/YOUR_CHECKPOINT_EPOCH_76.pth \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 20
```

#### Quick 100-Epoch Test
```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 128 \
    --early_stopping \
    --early_stopping_patience 15 \
    --label_smoothing 0.1 \
    --cutmix
```

### Option 3: Use Example Scripts

Run pre-configured scenarios:
```bash
# View available scenarios
./examples/cifar10_training_examples.sh

# Run a specific scenario (e.g., scenario 2 for resume)
./examples/cifar10_training_examples.sh 2
```

## Monitoring Your Training

### TensorBoard (Real-time Visualization)
In a separate terminal:
```bash
tensorboard --logdir ./logs --port 6006
```
Then open: http://localhost:6006

### CSV Logs (Metrics History)
```bash
# Watch in real-time
tail -f ./results/*.csv

# Or for specific file
tail -f ./logs/resnet18_cifar10_metrics.csv
```

### Check Checkpoints
```bash
# List all checkpoints
ls -lh checkpoints/

# Find best checkpoint
ls -lh checkpoints/*best.pth
```

## Training Progress

Your training will:
1. Download CIFAR-10 dataset (if not already present) → `./data/`
2. Start training from epoch 0 (or resume from checkpoint)
3. Save checkpoints every N epochs → `./checkpoints/`
4. Log metrics to TensorBoard → `./logs/`
5. Save CSV metrics → `./logs/` or custom path
6. Automatically save best model → `./checkpoints/*_best.pth`
7. Stop early if no improvement (if enabled)

## Expected Timeline

For 500-epoch training:
- **Fresh start:** ~8-9 hours on modern GPU (batch_size=128)
- **Resume from epoch 76:** ~7-8 hours remaining
- **With early stopping:** May complete around epoch 350-450

Epoch times vary by hardware:
- Modern GPU (RTX 3090, A100): ~60-70 seconds
- Mid-range GPU (GTX 1080, RTX 2070): ~90-120 seconds
- CPU only: Not recommended (very slow)

## Expected Results

Starting fresh or from early epochs:
- **Epoch 100:** 86-88% validation accuracy
- **Epoch 200:** 88-89% validation accuracy
- **Epoch 500:** 90-92% validation accuracy

With enhanced augmentation (CutMix, label smoothing):
- Additional +1-2% accuracy improvement
- Better generalization (smaller train/val gap)

## Troubleshooting

### Dependencies Not Installed
```bash
# Install all at once
pip install torch torchvision numpy tensorboard matplotlib seaborn

# Or individual packages
pip install torch torchvision
pip install numpy tensorboard
```

### Out of Memory (CUDA)
Reduce batch size:
```bash
--batch_size 64  # or even 32
```

### Training Too Slow
Check GPU usage:
```bash
nvidia-smi  # Should show GPU utilization

# Adjust workers if needed
--workers 2  # reduce if CPU bottleneck
```

### Checkpoint Not Found
```bash
# List available checkpoints
ls -lh checkpoints/

# Use absolute path
--resume /full/path/to/checkpoint.pth
```

### Early Stopping Too Aggressive
Increase patience:
```bash
--early_stopping_patience 60  # or 80
```

## All Command-Line Options

View complete options:
```bash
python models/scripts/train_ternary_models.py --help
```

Key options:
- `--model` - Model architecture (resnet18, mobilenetv2)
- `--dataset` - Dataset (cifar10, imagenet)
- `--epochs` - Total training epochs
- `--batch_size` - Batch size (default: 128)
- `--resume` - Resume from checkpoint
- `--early_stopping` - Enable early stopping
- `--early_stopping_patience` - Patience in epochs (default: 40)
- `--label_smoothing` - Label smoothing factor (0.0-0.2)
- `--cutmix` - Enable CutMix augmentation
- `--mixup` - Enable MixUp augmentation
- `--autoaugment` - Enable AutoAugment
- `--save_freq` - Checkpoint save frequency
- `--csv_log` - CSV log file path
- `--log_interval` - Logging interval (batches)

## Next Steps After Launch

1. **Monitor** - Watch TensorBoard and logs
2. **Verify** - Check first few epochs look reasonable
3. **Wait** - Let it train (hours)
4. **Analyze** - Review results in TensorBoard
5. **Export** - Export best model if desired

## Documentation

For more details, see:
- **START_HERE.md** - Complete quick-start guide
- **docs/CIFAR10_TRAINING_GUIDE.md** - Comprehensive training guide
- **docs/QUICK_START_CIFAR10.md** - Command reference
- **IMPLEMENTATION_CIFAR10.md** - Technical details

## Support

If you encounter issues:
1. Check error messages in terminal
2. Review TensorBoard for anomalies
3. Check GPU memory with `nvidia-smi`
4. Verify dependencies are installed
5. Consult documentation above

---

**Ready to train? Run `./launch_training.sh` or use one of the commands above!** 🚀
