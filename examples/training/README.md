# Training Scripts

Production-ready training scripts for neural network training with Triton DSL models.

## Overview

This directory contains comprehensive training scripts with support for:
- **ImageNet Training**: Full-scale distributed training with advanced augmentations
- **Transfer Learning**: Fine-tuning pretrained models on custom datasets
- **CIFAR-10 Training**: Quick experiments with ternary/quantized models
- **Quantization-Aware Training**: QAT for optimal quantized model performance

## Quick Start

### Basic CIFAR-10 Training

```bash
# Train a ternary ResNet18 on CIFAR-10
python train_cifar10.py \
    --model resnet18 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --save-dir ./checkpoints
```

### ImageNet Training

```bash
# Single GPU
python train_imagenet.py \
    --data-path /path/to/imagenet \
    --model resnet18 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.1 \
    --amp \
    --randaugment

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train_imagenet.py \
    --data-path /path/to/imagenet \
    --model resnet18 \
    --distributed \
    --batch-size 64 \
    --amp \
    --mixup \
    --cutmix \
    --ema
```

### Transfer Learning

```bash
# Fine-tune on custom dataset
python transfer_learning.py \
    --data-path ./my_dataset \
    --num-classes 10 \
    --model resnet18 \
    --pretrained \
    --freeze-backbone \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer adam
```

## Scripts

### 1. train_imagenet.py

Production-ready ImageNet training with distributed training, mixed precision, and advanced augmentations.

**Key Features:**
- Multi-GPU distributed training (DDP)
- Mixed precision training (AMP)
- Advanced augmentations: RandAugment, MixUp, CutMix
- Exponential Moving Average (EMA)
- Automatic checkpoint resumption
- Model export to ONNX/TorchScript
- Comprehensive logging and metrics

**Usage:**

```bash
# Full training with all features
python train_imagenet.py \
    --data-path /path/to/imagenet \
    --model resnet18 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.1 \
    --scheduler cosine \
    --warmup-epochs 5 \
    --amp \
    --randaugment \
    --mixup --mixup-alpha 0.2 \
    --cutmix --cutmix-alpha 1.0 \
    --mix-prob 0.5 \
    --ema --ema-decay 0.9999 \
    --label-smoothing 0.1 \
    --weight-decay 1e-4 \
    --grad-clip 5.0 \
    --save-dir ./checkpoints_imagenet \
    --export
```

**Distributed Training:**

```bash
# 4 GPUs, 64 batch size per GPU = 256 total
torchrun --nproc_per_node=4 train_imagenet.py \
    --distributed \
    --data-path /path/to/imagenet \
    --batch-size 64 \
    --lr 0.4 \
    [other args...]

# 8 GPUs across 2 nodes
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_imagenet.py \
    --distributed \
    [args...]
```

**Resume Training:**

```bash
python train_imagenet.py \
    --resume ./checkpoints_imagenet/checkpoint_epoch_50.pth \
    [other args...]
```

### 2. transfer_learning.py

Fine-tune pretrained models on custom datasets with flexible strategies.

**Key Features:**
- Load pretrained models (ImageNet or custom)
- Layer freezing/unfreezing strategies
- Progressive unfreezing
- Discriminative learning rates
- Multiple optimizer choices (SGD, Adam, AdamW)
- Advanced schedulers (Cosine, OneCycle, Plateau)
- Custom dataset support

**Usage:**

```bash
# Basic transfer learning
python transfer_learning.py \
    --data-path ./my_dataset \
    --num-classes 10 \
    --model resnet18 \
    --pretrained \
    --freeze-backbone \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer adam \
    --scheduler cosine

# Progressive unfreezing
python transfer_learning.py \
    --data-path ./my_dataset \
    --num-classes 10 \
    --model resnet50 \
    --pretrained \
    --freeze-backbone \
    --progressive-unfreeze \
    --unfreeze-epoch 15 \
    --epochs 40 \
    --batch-size 32 \
    --lr 0.001

# Discriminative learning rates
python transfer_learning.py \
    --data-path ./my_dataset \
    --num-classes 10 \
    --model resnet18 \
    --pretrained \
    --discriminative-lr \
    --lr 0.001 \
    --lr-decay 0.1 \
    --epochs 30

# With custom pretrained weights
python transfer_learning.py \
    --data-path ./my_dataset \
    --num-classes 10 \
    --model resnet18 \
    --pretrained-path ./my_pretrained_model.pth \
    --freeze-backbone \
    --epochs 30
```

**Dataset Format:**

Your dataset should follow this structure:

```
my_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── classN/
│       └── ...
└── val/
    ├── class1/
    ├── class2/
    └── classN/
```

### 3. train_cifar10.py

Quick training and experimentation on CIFAR-10 with ternary/quantized models.

**Features:**
- Ternary ResNet18 and MobileNetV2
- CutMix augmentation
- Multiple scheduler options
- Label smoothing
- Gradient clipping

**Usage:**

```bash
# Basic training
python train_cifar10.py \
    --model resnet18 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1

# With CutMix and label smoothing
python train_cifar10.py \
    --model mobilenetv2 \
    --epochs 100 \
    --cutmix \
    --cutmix-prob 0.5 \
    --label-smoothing 0.1 \
    --scheduler cosine
```

### 4. Quantization-Aware Training

See [../quantization/qat_training.py](../quantization/qat_training.py) for QAT training.

```bash
# Symlink for convenience
python quantization_aware_training.py [args...]
```

## Training Strategies

### Transfer Learning Strategies

#### 1. Feature Extraction (Frozen Backbone)

Train only the final classifier, keeping backbone frozen:

```bash
python transfer_learning.py \
    --pretrained \
    --freeze-backbone \
    --epochs 20 \
    --lr 0.001
```

**Best for:**
- Small datasets (< 1000 images)
- Similar domain to pretraining
- Fast training needed

#### 2. Fine-Tuning (Unfrozen Backbone)

Train all layers with small learning rate:

```bash
python transfer_learning.py \
    --pretrained \
    --epochs 50 \
    --lr 0.0001 \
    --weight-decay 1e-4
```

**Best for:**
- Medium datasets (1000-10000 images)
- Some domain shift from pretraining
- Better accuracy needed

#### 3. Progressive Unfreezing

Gradually unfreeze layers during training:

```bash
python transfer_learning.py \
    --pretrained \
    --freeze-backbone \
    --progressive-unfreeze \
    --unfreeze-epoch 15 \
    --epochs 40 \
    --lr 0.001
```

**Best for:**
- Preventing overfitting
- Large domain shift
- Balanced training

#### 4. Discriminative Learning Rates

Different learning rates for different layers:

```bash
python transfer_learning.py \
    --pretrained \
    --discriminative-lr \
    --lr 0.001 \
    --lr-decay 0.1 \
    --epochs 30
```

**Best for:**
- Fine-grained control
- Preventing catastrophic forgetting
- Advanced users

### Augmentation Strategies

#### Light Augmentation
For small models or quick experiments:
```bash
--color-jitter
```

#### Medium Augmentation
Balanced approach:
```bash
--randaugment
--mixup --mixup-alpha 0.2
```

#### Heavy Augmentation
Maximum regularization:
```bash
--randaugment
--mixup --mixup-alpha 0.2
--cutmix --cutmix-alpha 1.0
--mix-prob 0.5
--label-smoothing 0.1
```

### Learning Rate Scheduling

#### Cosine Annealing
Smooth decay, good default:
```bash
--scheduler cosine --warmup-epochs 5 --min-lr 1e-6
```

#### OneCycle
Fast convergence:
```bash
--scheduler onecycle
```

#### ReduceLROnPlateau
Adaptive to validation performance:
```bash
--scheduler plateau
```

#### MultiStep
Step-wise decay:
```bash
--scheduler multistep --milestones 30 60 90
```

## Best Practices

### 1. Learning Rate Selection

**Rule of thumb:**
- ImageNet (SGD): 0.1 * (batch_size / 256)
- Transfer learning (Adam): 0.001 for frozen, 0.0001 for unfrozen
- Fine-tuning: 10-100x smaller than original training

**Finding optimal LR:**
```bash
# LR range test
python transfer_learning.py \
    --lr 1e-6 \
    --scheduler onecycle \
    --epochs 5
# Monitor loss curve and select LR at steepest descent
```

### 2. Batch Size Selection

**Guidelines:**
- Small datasets: 16-32
- Medium datasets: 32-128
- Large datasets: 128-512
- Distributed: 32-128 per GPU

**Memory constraints:**
- Use `--amp` for 40-50% memory reduction
- Gradient accumulation for effective larger batches

### 3. Data Augmentation

**Progressive augmentation strategy:**

Phase 1 (Epochs 0-10): Light
```bash
--color-jitter
```

Phase 2 (Epochs 10-30): Medium
```bash
--randaugment
```

Phase 3 (Epochs 30+): Heavy
```bash
--randaugment --mixup --cutmix
```

### 4. Avoiding Overfitting

**Techniques (in order of effectiveness):**

1. More data (always best)
2. Data augmentation
3. Weight decay: `--weight-decay 1e-4`
4. Dropout (model-dependent)
5. Label smoothing: `--label-smoothing 0.1`
6. Early stopping
7. Reduce model size

**Detection:**
- Training acc > 95% but validation acc < 80%
- Train/val loss divergence
- Solution: Increase regularization

### 5. Avoiding Underfitting

**Symptoms:**
- Both train and val accuracy low
- Loss not decreasing

**Solutions:**
1. Increase model capacity
2. Train longer
3. Increase learning rate
4. Reduce regularization
5. Check data quality

### 6. Mixed Precision Training

Always use `--amp` for:
- 40-50% faster training
- 50% less memory
- Same or better accuracy
- Free speedup on modern GPUs

```bash
--amp
```

### 7. Distributed Training

**Linear scaling rule:**
```bash
# 4 GPUs, scale batch size and LR
--batch-size 64    # 256 total (4 * 64)
--lr 0.4           # 4x base LR (0.1 * 4)
```

**Warmup for large batches:**
```bash
--warmup-epochs 5
```

## Hyperparameter Tuning

### Systematic Approach

1. **Baseline** (default parameters)
2. **Learning rate** (most important)
3. **Batch size** (affects LR)
4. **Augmentation** (prevents overfitting)
5. **Regularization** (weight decay, label smoothing)
6. **Scheduler** (cosine, onecycle, etc.)
7. **Architecture** (only if needed)

### Grid Search Example

```bash
# Test learning rates
for lr in 0.1 0.01 0.001 0.0001; do
    python transfer_learning.py \
        --data-path ./data \
        --lr $lr \
        --save-dir ./results/lr_${lr}
done

# Test schedulers
for sched in cosine onecycle plateau; do
    python transfer_learning.py \
        --data-path ./data \
        --scheduler $sched \
        --save-dir ./results/sched_${sched}
done
```

### Recommended Starting Points

**Small dataset (< 1000 images):**
```bash
python transfer_learning.py \
    --pretrained \
    --freeze-backbone \
    --batch-size 16 \
    --lr 0.001 \
    --optimizer adam \
    --epochs 50 \
    --color-jitter \
    --weight-decay 1e-4
```

**Medium dataset (1000-10000 images):**
```bash
python transfer_learning.py \
    --pretrained \
    --progressive-unfreeze \
    --unfreeze-epoch 20 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer adamw \
    --epochs 60 \
    --auto-augment \
    --label-smoothing 0.1
```

**Large dataset (10000+ images):**
```bash
python train_imagenet.py \
    --batch-size 128 \
    --lr 0.1 \
    --optimizer sgd \
    --epochs 100 \
    --scheduler cosine \
    --warmup-epochs 5 \
    --randaugment \
    --mixup \
    --cutmix \
    --amp \
    --ema
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solutions:**
```bash
# Reduce batch size
--batch-size 16  # instead of 32

# Use mixed precision
--amp

# Reduce image size
--img-size 224  # instead of 299

# Reduce workers
--workers 2
```

#### 2. NaN Loss

**Causes & Solutions:**
- **High learning rate**: Reduce by 10x
- **Mixed precision instability**: Remove `--amp` or use `--grad-clip 1.0`
- **Bad initialization**: Check pretrained weights
- **Data issues**: Check for NaN/Inf in data

#### 3. Slow Convergence

**Solutions:**
```bash
# Increase learning rate
--lr 0.01  # instead of 0.001

# Use warmup
--warmup-epochs 5

# Better optimizer
--optimizer adamw

# Better scheduler
--scheduler onecycle
```

#### 4. Poor Validation Accuracy

**Overfitting:**
```bash
--weight-decay 1e-4
--label-smoothing 0.1
--auto-augment
# More data or reduce model size
```

**Underfitting:**
```bash
# Increase capacity
--model resnet50  # instead of resnet18

# Train longer
--epochs 100  # instead of 50

# Increase LR
--lr 0.001  # instead of 0.0001
```

#### 5. Distributed Training Issues

**NCCL errors:**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If using non-InfiniBand
```

**Slow distributed training:**
```bash
# Check network bandwidth
# Increase batch size per GPU
--batch-size 64  # instead of 32

# Use persistent workers
--workers 4
```

#### 6. Data Loading Bottleneck

**Symptoms:** GPU utilization < 90%

**Solutions:**
```bash
# Increase workers
--workers 8

# Use pin_memory (automatic)
# Check disk I/O (SSD recommended)
```

### Performance Optimization

**Checklist:**
- ✅ Use `--amp` for mixed precision
- ✅ Use `--workers 4-8` for data loading
- ✅ Set batch size to maximize GPU memory
- ✅ Use pin_memory (automatic)
- ✅ Use persistent_workers (automatic)
- ✅ Use distributed training for multiple GPUs
- ✅ Benchmark different model architectures

## Monitoring Training

### Key Metrics to Track

1. **Training loss** - Should decrease smoothly
2. **Validation loss** - Should decrease and stay close to training loss
3. **Training accuracy** - Should increase
4. **Validation accuracy** - Should increase and be close to training accuracy
5. **Learning rate** - Check scheduler is working
6. **Batch time** - Monitor training speed

### Using Training History

All scripts save `history.json` with training metrics:

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('checkpoints/history.json') as f:
    history = json.load(f)

# Plot training curves
plt.plot([h['loss'] for h in history['train']], label='Train Loss')
plt.plot([h['loss'] for h in history['val']], label='Val Loss')
plt.legend()
plt.savefig('loss_curve.png')

plt.figure()
plt.plot([h['accuracy'] for h in history['train']], label='Train Acc')
plt.plot([h['accuracy'] for h in history['val']], label='Val Acc')
plt.legend()
plt.savefig('accuracy_curve.png')
```

## Advanced Topics

### Custom Models

Add your model to the scripts:

```python
# In train_imagenet.py or transfer_learning.py
from my_models import my_custom_model

# Add to model creation section
if args.model == 'custom':
    model = my_custom_model(num_classes=args.num_classes)
```

### Custom Datasets

Implement `Dataset` class:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Load your data
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

### Custom Augmentations

```python
from torchvision.transforms import Compose

my_transforms = Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # Add custom transforms
    MyCustomTransform(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Ensemble Training

Train multiple models with different random seeds:

```bash
for seed in 42 123 456 789 999; do
    python train_imagenet.py \
        --data-path ./data \
        --save-dir ./checkpoints/seed_${seed} \
        --seed ${seed}
done

# Ensemble predictions
python ensemble_predict.py \
    --checkpoints ./checkpoints/seed_*/model_best.pth
```

## Export and Deployment

### Export Trained Models

```bash
# Automatically export after training
python train_imagenet.py \
    [training args...] \
    --export

# Models saved as:
# - checkpoints/model.torchscript.pt
# - checkpoints/model.onnx
```

### Load Exported Models

```python
# TorchScript
model = torch.jit.load('checkpoints/model.torchscript.pt')
output = model(input_tensor)

# ONNX
import onnxruntime as ort
session = ort.InferenceSession('checkpoints/model.onnx')
output = session.run(None, {'input': input_array})
```

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the main [README](../../README.md)
- Review [examples documentation](../README.md)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
