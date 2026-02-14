# Model Training and Publishing Scripts

This directory contains scripts for training, benchmarking, packaging, and publishing ternary neural network models.

## Scripts

### train_ternary_models.py

**Enhanced CIFAR-10 and ImageNet training script with production features.**

Train ResNet-18 or MobileNetV2 with ternary quantization, supporting:
- ✨ Early stopping with configurable patience
- ✨ Advanced data augmentation (CutMix, MixUp, AutoAugment, RandAugment)
- ✨ Label smoothing for better generalization
- ✨ TensorBoard and CSV logging
- ✨ Complete checkpoint resumption (model, optimizer, scheduler, early stopping state)
- ✨ Multiple LR schedulers (cosine annealing, step decay)

**Quick Start:**
```bash
# Fresh 500-epoch CIFAR-10 training
python train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix

# Resume from checkpoint
python train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --resume ./checkpoints/resnet18_cifar10_epoch_76.pth \
    --early_stopping \
    --label_smoothing 0.1 \
    --cutmix

# View all options
python train_ternary_models.py --help
```

**Documentation:**
- Complete guide: `docs/CIFAR10_TRAINING_GUIDE.md`
- Quick reference: `docs/QUICK_START_CIFAR10.md`
- Examples: `examples/cifar10_training_examples.sh`

**Features:**

| Feature | Description | Flag |
|---------|-------------|------|
| Early Stopping | Stop training if no improvement | `--early_stopping --early_stopping_patience 40` |
| CutMix | Cut and paste image patches | `--cutmix --cutmix_alpha 1.0` |
| MixUp | Blend images together | `--mixup --mixup_alpha 1.0` |
| AutoAugment | CIFAR-10 learned policy | `--autoaugment` |
| RandAugment | Random augmentation | `--randaugment` |
| Label Smoothing | Prevent overconfidence | `--label_smoothing 0.1` |
| TensorBoard | Real-time visualization | Automatic if installed |
| CSV Logging | Metrics history | `--csv_log path/to/file.csv` |

**Expected Results:**
- ResNet-18 CIFAR-10: 88-91% validation accuracy @ 500 epochs
- Training time: ~8 hours on modern GPU (batch_size=128)
- With early stopping: May complete around epoch 350-450

---

### benchmark_ternary_models.py

Benchmark trained ternary models for inference speed, memory usage, and accuracy.

**Usage:**
```bash
python benchmark_ternary_models.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18_cifar10_best.pth \
    --dataset cifar10 \
    --batch_size 128
```

---

### package_ternary_models.py

Package trained models with metadata for distribution.

**Usage:**
```bash
python package_ternary_models.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18_cifar10_best.pth \
    --output ./packages/
```

---

### publish_model.py

Publish trained models to ONNX, Hugging Face Hub, and GitHub Releases.

**Usage:**
```bash
# Export to ONNX
python publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --export-onnx

# Publish to Hugging Face Hub
python publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --hf-repo username/ternary-resnet18

# Create GitHub Release
python publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --github-release v1.0.0 \
    --github-repo username/Triton
```

See `docs/EXPORT_GUIDE.md` for detailed documentation.

---

## Common Workflows

### 1. Train a Model
```bash
# Start training
python train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --early_stopping \
    --cutmix \
    --label_smoothing 0.1

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

### 2. Resume Training
```bash
python train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --resume ./checkpoints/resnet18_cifar10_epoch_76.pth \
    --early_stopping
```

### 3. Benchmark Model
```bash
python benchmark_ternary_models.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18_cifar10_best.pth \
    --dataset cifar10
```

### 4. Export and Publish
```bash
# Export to ONNX
python publish_model.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18_cifar10_best.pth \
    --export-onnx

# Publish to Hugging Face
python publish_model.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18_cifar10_best.pth \
    --hf-repo username/ternary-resnet18
```

## Requirements

```bash
# Training
pip install torch torchvision numpy tensorboard

# Publishing (optional)
pip install onnx huggingface_hub PyGithub

# Or install all
pip install -e ".[export]"
```

## Directory Structure

```
models/scripts/
├── README.md                      # This file
├── train_ternary_models.py        # Enhanced training script ⭐
├── benchmark_ternary_models.py    # Model benchmarking
├── package_ternary_models.py      # Model packaging
└── publish_model.py               # Model publishing

Related:
├── docs/CIFAR10_TRAINING_GUIDE.md       # Complete training guide
├── docs/QUICK_START_CIFAR10.md          # Quick reference
├── examples/cifar10_training_examples.sh # Ready-to-run examples
└── IMPLEMENTATION_CIFAR10.md            # Implementation summary
```

## Support

For training-related questions:
- See `docs/CIFAR10_TRAINING_GUIDE.md` for comprehensive documentation
- See `docs/QUICK_START_CIFAR10.md` for quick commands
- Run `examples/cifar10_training_examples.sh` for ready-to-use scenarios

For export/publishing questions:
- See `docs/EXPORT_GUIDE.md`

## Recent Enhancements (2026-02-14)

The training script has been significantly enhanced with:
- Early stopping with configurable patience (--early_stopping)
- Advanced data augmentation (CutMix, MixUp, AutoAugment, RandAugment)
- Label smoothing for regularization (--label_smoothing)
- TensorBoard and CSV logging
- Complete checkpoint state preservation
- Support for 500+ epoch training runs

See `IMPLEMENTATION_CIFAR10.md` for full details.
