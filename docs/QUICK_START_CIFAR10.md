# CIFAR-10 Training Quick Reference

## Your Current Situation
- Epoch: 76/300
- Train Acc: 86.21%, Val Acc: 85.54%
- Train Loss: 0.3994, Val Loss: 0.6352
- LR: 0.000850
- Epoch Time: ~61 seconds

## Resume Training to 500 Epochs

### Option 1: With All Enhancements (Recommended)
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

### Option 2: Minimal Changes (Safe)
```bash
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume ./checkpoints/YOUR_CHECKPOINT_EPOCH_76.pth \
    --early_stopping \
    --early_stopping_patience 40
```

## What You Get

### New Features Added ✨
1. **Early Stopping** - Stops training if no improvement for 40 epochs
2. **CutMix** - Advanced data augmentation for better regularization
3. **Label Smoothing** - Prevents overconfident predictions
4. **TensorBoard Logging** - Real-time visualization of training
5. **CSV Logging** - Complete metrics history
6. **Enhanced Checkpointing** - Saves scheduler state, early stopping state
7. **MixUp** - Optional additional augmentation
8. **AutoAugment/RandAugment** - Optional advanced augmentation

### File Structure
```
Triton/
├── checkpoints/              # Model checkpoints
│   ├── resnet18_cifar10_epoch_76.pth
│   ├── resnet18_cifar10_epoch_86.pth
│   ├── resnet18_cifar10_best.pth
│   └── ...
├── logs/                     # Training logs
│   ├── resnet18_cifar10_*/  # TensorBoard logs
│   └── resnet18_cifar10_metrics.csv
└── models/scripts/
    └── train_ternary_models.py  # Enhanced training script
```

## Monitoring Training

### TensorBoard (Real-time Visualization)
```bash
tensorboard --logdir ./logs --port 6006
```
Then open: http://localhost:6006

### CSV Logs (For Analysis)
```bash
# View metrics
cat ./logs/resnet18_cifar10_metrics.csv

# Watch in real-time
tail -f ./logs/resnet18_cifar10_metrics.csv
```

## Expected Timeline

From epoch 76 to 500:
- **Remaining epochs**: 424
- **Estimated time**: ~7.2 hours (at 61s/epoch)
- **With early stopping**: May stop around epoch 350-450
- **Expected final accuracy**: 88-91%

## Troubleshooting

### If Training Starts from Epoch 0
❌ Problem: Checkpoint not loaded correctly
✅ Solution: Check the path in `--resume` argument

### If Overfitting Occurs (Train >> Val Acc)
❌ Problem: Train acc much higher than val acc
✅ Solution: Add stronger regularization:
```bash
--label_smoothing 0.15 \
--cutmix \
--mixup \
--weight_decay 1e-3
```

### If Training is Too Slow
❌ Problem: Each epoch takes > 90 seconds
✅ Solution: Check GPU usage with `nvidia-smi` and:
```bash
--batch_size 256  # If you have GPU memory
--workers 2       # If CPU is bottleneck
```

### If Out of Memory
❌ Problem: CUDA out of memory error
✅ Solution: Reduce batch size:
```bash
--batch_size 64   # or even 32
```

## Quick Commands

### Check GPU Status
```bash
nvidia-smi
```

### List Available Checkpoints
```bash
ls -lh checkpoints/
```

### View Last 10 Training Epochs
```bash
tail -20 logs/resnet18_cifar10_metrics.csv
```

### Kill Training (if needed)
```bash
# Press Ctrl+C in terminal
# Or find process ID:
ps aux | grep train_ternary
kill <PID>
```

## Next Steps After Training

1. **Check Best Accuracy**
   ```bash
   grep "best" logs/resnet18_cifar10_metrics.csv | tail -1
   ```

2. **Load Best Model**
   ```python
   import torch
   checkpoint = torch.load('checkpoints/resnet18_cifar10_best.pth')
   print(f"Best accuracy: {checkpoint['accuracy']:.2f}%")
   ```

3. **Export Model** (Optional)
   ```bash
   python models/scripts/publish_model.py \
       --model resnet18 \
       --checkpoint checkpoints/resnet18_cifar10_best.pth \
       --export-onnx
   ```

## Key Parameters Explained

| Parameter | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `--epochs` | Total training epochs | 500 |
| `--early_stopping_patience` | Stop if no improvement for N epochs | 40 (8% of total) |
| `--label_smoothing` | Regularization strength | 0.1 |
| `--cutmix` | Data augmentation | Enable |
| `--weight_decay` | L2 regularization | 5e-4 |
| `--save_freq` | Checkpoint frequency | 10-20 |
| `--batch_size` | Samples per batch | 128 |

## Full Documentation

For complete details, see:
- **Training Guide**: `docs/CIFAR10_TRAINING_GUIDE.md`
- **Example Scripts**: `examples/cifar10_training_examples.sh`
- **Training Script**: `models/scripts/train_ternary_models.py`

## Support

If you encounter issues:
1. Check training logs for error messages
2. Review TensorBoard plots for anomalies
3. Verify checkpoint files exist and are valid
4. Check GPU memory with `nvidia-smi`

## Realistic Expectations

### ResNet-18 Ternary on CIFAR-10
- **Your Current**: 85.54% val acc at epoch 76
- **Expected at 200**: 87-89% val acc
- **Expected at 500**: 88-91% val acc
- **With best augmentation**: 90-92% val acc

Note: Ternary networks are ~2-3% less accurate than FP32 but use 32x less memory!
