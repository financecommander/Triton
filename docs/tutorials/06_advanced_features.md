# Tutorial 6: Advanced Features

Explore cutting-edge features for ternary neural networks, including mixed precision techniques, knowledge distillation, neural architecture search, hardware-specific optimizations, and advanced compression methods.

## Learning Objectives

- Implement mixed precision ternary models
- Apply knowledge distillation techniques
- Use neural architecture search for ternary models
- Optimize for specific hardware architectures
- Apply advanced pruning and compression
- Combine multiple optimization techniques
- Design state-of-the-art ternary architectures

## Prerequisites

- Completed all previous tutorials
- Strong understanding of neural network theory
- Familiarity with optimization techniques
- Experience with PyTorch internals

## Mixed Precision Ternary Models

### Hybrid Precision Architecture

```python
import torch
import torch.nn as nn

class MixedPrecisionTernaryModel(nn.Module):
    """
    Model with mixed precision:
    - Input/Output layers: FP32
    - Middle layers: Ternary
    - Batch norms: FP32
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # First layer: FP32 (preserve input information)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Middle layers: Ternary (compression)
        self.ternary_layers = nn.ModuleList([
            TernaryConv2d(64, 128, kernel_size=3, padding=1),
            TernaryConv2d(128, 256, kernel_size=3, padding=1),
            TernaryConv2d(256, 512, kernel_size=3, padding=1),
        ])
        
        # Batch norms: FP32 (stable training)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512),
        ])
        
        # Output layer: FP32 (preserve precision)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # FP32 first layer
        x = torch.relu(self.bn1(self.conv1(x)))
        
        # Ternary middle layers
        for ternary_layer, bn in zip(self.ternary_layers, self.batch_norms):
            x = ternary_layer(x)
            x = torch.relu(bn(x))
            x = torch.max_pool2d(x, 2)
        
        # Global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # FP32 output layer
        x = self.fc(x)
        
        return x

class TernaryConv2d(nn.Module):
    """Ternary convolutional layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # Full-precision weights for training
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        
        # Scaling factor for ternary weights
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        
    def forward(self, x):
        # Ternarize weights
        ternary_weight = self.ternarize(self.weight)
        
        # Scale by learned alpha
        scaled_weight = ternary_weight * self.alpha
        
        # Convolution
        return torch.nn.functional.conv2d(
            x, scaled_weight,
            stride=self.stride,
            padding=self.padding
        )
    
    @staticmethod
    def ternarize(weight):
        """Ternarize weights using threshold."""
        threshold = 0.7 * weight.abs().mean()
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1
        ternary[weight < -threshold] = -1
        return ternary

# Usage
model = MixedPrecisionTernaryModel(num_classes=10)
print(model)
```

### Adaptive Precision Selection

```python
class AdaptivePrecisionModel(nn.Module):
    """
    Automatically select precision per layer based on sensitivity.
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Define layers with different precisions
        self.layers = nn.ModuleList()
        self.precisions = []  # Track precision of each layer
        
        # Build network
        channels = [3, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            # Analyze sensitivity and decide precision
            precision = self.select_precision(i, len(channels) - 1)
            self.precisions.append(precision)
            
            if precision == 'fp32':
                layer = nn.Conv2d(channels[i], channels[i+1], 3, padding=1)
            elif precision == 'ternary':
                layer = TernaryConv2d(channels[i], channels[i+1], 3, padding=1)
            else:  # binary
                layer = BinaryConv2d(channels[i], channels[i+1], 3, padding=1)
            
            self.layers.append(layer)
        
        self.fc = nn.Linear(512, num_classes)
    
    def select_precision(self, layer_idx, total_layers):
        """
        Select precision based on layer position.
        Early layers: FP32 (more important)
        Middle layers: Ternary (compression)
        Late layers: Can be more aggressive
        """
        position = layer_idx / total_layers
        
        if position < 0.2:  # First 20%
            return 'fp32'
        elif position < 0.8:  # Middle 60%
            return 'ternary'
        else:  # Last 20%
            return 'ternary'
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = torch.max_pool2d(x, 2)
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def print_precision_summary(self):
        """Print precision usage summary."""
        print('Layer Precision Summary:')
        print('=' * 50)
        for i, (precision, layer) in enumerate(zip(self.precisions, self.layers)):
            print(f'Layer {i}: {precision} - {layer.__class__.__name__}')
        print('=' * 50)

# Usage
model = AdaptivePrecisionModel()
model.print_precision_summary()
```

### Per-Channel Quantization

```python
class PerChannelTernary(nn.Module):
    """Ternary layer with per-channel thresholds."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        
        # Per-channel scaling factors
        self.alpha = nn.Parameter(torch.ones(out_channels))
        
        # Per-channel thresholds (learnable)
        self.threshold = nn.Parameter(torch.ones(out_channels) * 0.33)
        
    def forward(self, x):
        # Ternarize with per-channel thresholds
        ternary_weight = self.ternarize_per_channel(
            self.weight, 
            self.threshold
        )
        
        # Apply per-channel scaling
        scaled_weight = ternary_weight * self.alpha.view(-1, 1, 1, 1)
        
        return torch.nn.functional.conv2d(x, scaled_weight, padding=1)
    
    @staticmethod
    def ternarize_per_channel(weight, threshold):
        """Ternarize with different threshold per output channel."""
        ternary = torch.zeros_like(weight)
        
        for i in range(weight.size(0)):
            channel_weight = weight[i]
            channel_threshold = threshold[i]
            
            ternary[i][channel_weight > channel_threshold] = 1
            ternary[i][channel_weight < -channel_threshold] = -1
        
        return ternary
```

## Knowledge Distillation

### Teacher-Student Framework

```python
import torch.nn.functional as F

class KnowledgeDistillation:
    """
    Distill knowledge from full-precision teacher to ternary student.
    """
    
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        """
        Args:
            teacher_model: Pre-trained full-precision model
            student_model: Ternary model to train
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss.
        
        Loss = α * KL(teacher || student) + (1-α) * CrossEntropy(student, labels)
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence
        distillation_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, distillation_loss, hard_loss
    
    def train_step(self, x, labels, optimizer):
        """Single training step with distillation."""
        self.student.train()
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Student forward
        student_logits = self.student(x)
        
        # Compute loss
        loss, dist_loss, hard_loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            labels
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'distillation_loss': dist_loss.item(),
            'hard_loss': hard_loss.item()
        }

# Usage
teacher = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
student = MixedPrecisionTernaryModel()

distiller = KnowledgeDistillation(
    teacher_model=teacher,
    student_model=student,
    temperature=4.0,
    alpha=0.7
)

# Training loop
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        losses = distiller.train_step(data, target, optimizer)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}')
            print(f'Total Loss: {losses["total_loss"]:.4f}')
            print(f'Distillation Loss: {losses["distillation_loss"]:.4f}')
            print(f'Hard Loss: {losses["hard_loss"]:.4f}')
```

### Feature-Based Distillation

```python
class FeatureDistillation(nn.Module):
    """Distill intermediate features from teacher to student."""
    
    def __init__(self, teacher, student, feature_layers):
        """
        Args:
            teacher: Teacher model
            student: Student model
            feature_layers: List of layer names to match
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.feature_layers = feature_layers
        
        # Adaptation layers to match dimensions
        self.adaptors = nn.ModuleDict()
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def add_adaptor(self, layer_name, student_dim, teacher_dim):
        """Add adaptation layer to match feature dimensions."""
        if student_dim != teacher_dim:
            self.adaptors[layer_name] = nn.Conv2d(
                student_dim, teacher_dim, kernel_size=1
            )
    
    def extract_features(self, model, x, layer_names):
        """Extract intermediate features."""
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        output = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features, output
    
    def compute_feature_loss(self, student_features, teacher_features):
        """Compute feature matching loss."""
        loss = 0.0
        
        for layer_name in self.feature_layers:
            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]
            
            # Apply adaptor if needed
            if layer_name in self.adaptors:
                student_feat = self.adaptors[layer_name](student_feat)
            
            # L2 loss
            loss += F.mse_loss(student_feat, teacher_feat)
        
        return loss / len(self.feature_layers)
    
    def forward(self, x, labels):
        """Forward with feature distillation."""
        # Extract features
        with torch.no_grad():
            teacher_features, teacher_output = self.extract_features(
                self.teacher, x, self.feature_layers
            )
        
        student_features, student_output = self.extract_features(
            self.student, x, self.feature_layers
        )
        
        # Feature loss
        feature_loss = self.compute_feature_loss(student_features, teacher_features)
        
        # Classification loss
        cls_loss = F.cross_entropy(student_output, labels)
        
        # Total loss
        total_loss = cls_loss + 0.1 * feature_loss
        
        return total_loss, cls_loss, feature_loss
```

## Neural Architecture Search (NAS)

### Differentiable Architecture Search for Ternary Models

```python
class TernarySearchSpace(nn.Module):
    """Searchable space for ternary architectures."""
    
    def __init__(self, num_layers=8, num_operations=4):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_operations = num_operations
        
        # Architecture parameters (learnable)
        self.arch_params = nn.Parameter(
            torch.randn(num_layers, num_operations)
        )
        
        # Operations to choose from
        self.operations = nn.ModuleList([
            self.create_operations(i) for i in range(num_layers)
        ])
    
    def create_operations(self, layer_idx):
        """Create candidate operations for a layer."""
        channels = 64 * (2 ** (layer_idx // 2))
        
        ops = nn.ModuleList([
            # Option 1: Ternary 3x3 conv
            TernaryConv2d(channels, channels, 3, padding=1),
            
            # Option 2: Ternary 5x5 conv
            TernaryConv2d(channels, channels, 5, padding=2),
            
            # Option 3: Separable ternary conv
            nn.Sequential(
                TernaryConv2d(channels, channels, 3, padding=1),
                TernaryConv2d(channels, channels, 1)
            ),
            
            # Option 4: Skip connection
            nn.Identity()
        ])
        
        return ops
    
    def forward(self, x):
        """Forward with mixed operations."""
        for layer_idx in range(self.num_layers):
            # Get architecture weights for this layer
            weights = F.softmax(self.arch_params[layer_idx], dim=0)
            
            # Compute weighted sum of operations
            layer_output = sum(
                w * op(x)
                for w, op in zip(weights, self.operations[layer_idx])
            )
            
            x = torch.relu(layer_output)
        
        return x
    
    def get_best_architecture(self):
        """Get the best architecture after search."""
        best_ops = []
        
        for layer_idx in range(self.num_layers):
            weights = F.softmax(self.arch_params[layer_idx], dim=0)
            best_op_idx = weights.argmax().item()
            best_ops.append(best_op_idx)
        
        return best_ops
    
    def print_architecture(self):
        """Print discovered architecture."""
        best_ops = self.get_best_architecture()
        op_names = ['Ternary 3x3', 'Ternary 5x5', 'Separable', 'Skip']
        
        print('Discovered Architecture:')
        print('=' * 50)
        for i, op_idx in enumerate(best_ops):
            print(f'Layer {i}: {op_names[op_idx]}')
        print('=' * 50)

# Architecture search training
def train_architecture_search(search_space, train_loader, val_loader, epochs=50):
    """Train with architecture search."""
    
    # Separate optimizers for weights and architecture
    model_optimizer = torch.optim.Adam(
        search_space.operations.parameters(),
        lr=0.001
    )
    
    arch_optimizer = torch.optim.Adam(
        [search_space.arch_params],
        lr=0.003
    )
    
    for epoch in range(epochs):
        # Train model weights
        search_space.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            
            model_optimizer.zero_grad()
            output = search_space(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            model_optimizer.step()
        
        # Update architecture on validation set
        search_space.eval()
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            
            arch_optimizer.zero_grad()
            output = search_space(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            arch_optimizer.step()
            break  # Use one batch
        
        if epoch % 10 == 0:
            search_space.print_architecture()

# Usage
search_space = TernarySearchSpace(num_layers=8, num_operations=4)
train_architecture_search(search_space, train_loader, val_loader)
```

### Evolutionary Architecture Search

```python
import random
import copy

class EvolutionaryTernaryNAS:
    """Evolutionary search for ternary architectures."""
    
    def __init__(self, population_size=20, num_generations=50):
        self.population_size = population_size
        self.num_generations = num_generations
        
    def random_architecture(self, num_layers=8):
        """Generate random architecture."""
        ops = ['ternary_3x3', 'ternary_5x5', 'separable', 'skip']
        return [random.choice(ops) for _ in range(num_layers)]
    
    def mutate(self, architecture, mutation_rate=0.2):
        """Mutate architecture."""
        ops = ['ternary_3x3', 'ternary_5x5', 'separable', 'skip']
        mutated = architecture.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(ops)
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Crossover two architectures."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def evaluate_architecture(self, architecture, train_loader, val_loader):
        """Evaluate architecture fitness."""
        # Build model from architecture
        model = self.build_model(architecture).cuda()
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for a few epochs
        for epoch in range(5):
            model.train()
            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def build_model(self, architecture):
        """Build model from architecture description."""
        # Implement model builder based on architecture
        layers = []
        for op in architecture:
            if op == 'ternary_3x3':
                layers.append(TernaryConv2d(64, 64, 3, padding=1))
            elif op == 'ternary_5x5':
                layers.append(TernaryConv2d(64, 64, 5, padding=2))
            # ... etc
        
        return nn.Sequential(*layers)
    
    def search(self, train_loader, val_loader):
        """Run evolutionary search."""
        # Initialize population
        population = [
            self.random_architecture() for _ in range(self.population_size)
        ]
        
        best_architecture = None
        best_fitness = 0.0
        
        for generation in range(self.num_generations):
            print(f'Generation {generation + 1}/{self.num_generations}')
            
            # Evaluate population
            fitness_scores = []
            for arch in population:
                fitness = self.evaluate_architecture(
                    arch, train_loader, val_loader
                )
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_architecture = arch.copy()
            
            print(f'Best fitness: {best_fitness:.4f}')
            
            # Selection
            sorted_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True
            )
            
            # Keep top 50%
            survivors = [population[i] for i in sorted_indices[:self.population_size // 2]]
            
            # Generate new population
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                # Crossover
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_architecture, best_fitness

# Usage
nas = EvolutionaryTernaryNAS(population_size=20, num_generations=50)
best_arch, best_fitness = nas.search(train_loader, val_loader)
print(f'Best architecture: {best_arch}')
print(f'Best fitness: {best_fitness:.4f}')
```

## Hardware-Specific Optimization

### CUDA Kernel for Ternary Operations

```python
from torch.utils.cpp_extension import load_inline

# CUDA kernel for optimized ternary matrix multiplication
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ternary_matmul_kernel(
    const float* input,
    const int8_t* ternary_weight,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (out_idx < out_features) {
        float sum = 0.0f;
        
        for (int i = 0; i < in_features; i++) {
            int input_idx = batch_idx * in_features + i;
            int weight_idx = out_idx * in_features + i;
            
            int8_t w = ternary_weight[weight_idx];
            
            // Zero-skipping optimization
            if (w != 0) {
                sum += input[input_idx] * w;
            }
        }
        
        int output_idx = batch_idx * out_features + out_idx;
        output[output_idx] = sum;
    }
}

torch::Tensor ternary_matmul_cuda(
    torch::Tensor input,
    torch::Tensor ternary_weight
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = ternary_weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 blocks(batch_size, (out_features + 255) / 256);
    dim3 threads(256);
    
    ternary_matmul_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        ternary_weight.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
'''

cpp_source = '''
torch::Tensor ternary_matmul_cuda(torch::Tensor input, torch::Tensor ternary_weight);
'''

# Load extension
ternary_ops = load_inline(
    name='ternary_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['ternary_matmul_cuda'],
    with_cuda=True,
    extra_cuda_cflags=['-O3']
)

class OptimizedTernaryLinear(nn.Module):
    """Linear layer with optimized CUDA kernel."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights as int8 for efficiency
        self.ternary_weight = nn.Parameter(
            torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
        )
        
    def forward(self, x):
        return ternary_ops.ternary_matmul_cuda(x, self.ternary_weight)
```

### Specialized ARM/NEON Optimizations

```python
class ARMOptimizedTernary(nn.Module):
    """Ternary operations optimized for ARM processors."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Pack ternary weights into efficient format
        # 4 ternary values can fit in 1 byte (2 bits each)
        self.packed_weights = self.pack_ternary_weights(
            torch.randint(-1, 2, (out_features, in_features))
        )
        
        self.in_features = in_features
        self.out_features = out_features
    
    @staticmethod
    def pack_ternary_weights(weights):
        """Pack ternary weights for efficient storage."""
        # Convert {-1, 0, 1} to {0, 1, 2}
        packed = weights + 1
        
        # Pack 4 values per byte
        # This is a simplified version
        return packed.byte()
    
    def forward(self, x):
        """Forward with ARM-optimized operations."""
        # Use quantized operations available on ARM
        # This is a placeholder - actual implementation would use
        # ARM NEON intrinsics or quantized operations
        
        # Unpack weights
        weights = self.unpack_weights(self.packed_weights)
        
        # Matrix multiplication with zero-skipping
        return self.optimized_matmul(x, weights)
    
    def optimized_matmul(self, x, weights):
        """Optimized matrix multiplication."""
        # Skip zero weights
        mask = (weights != 0)
        sparse_weights = weights * mask
        
        return F.linear(x, sparse_weights)
```

## Advanced Pruning and Compression

### Structured Pruning for Ternary Models

```python
class TernaryPruner:
    """Structured pruning for ternary models."""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        
    def compute_importance(self, layer):
        """Compute importance scores for channels/filters."""
        if isinstance(layer, (nn.Conv2d, TernaryConv2d)):
            # Compute L1 norm of each filter
            weight = layer.weight.data
            importance = weight.abs().sum(dim=(1, 2, 3))
        elif isinstance(layer, (nn.Linear, TernaryConv2d)):
            # Compute L1 norm of each output neuron
            weight = layer.weight.data
            importance = weight.abs().sum(dim=1)
        else:
            return None
        
        return importance
    
    def prune_layer(self, layer, num_channels_to_prune):
        """Prune least important channels."""
        importance = self.compute_importance(layer)
        
        if importance is None:
            return
        
        # Find channels to prune
        _, indices = importance.sort()
        prune_indices = indices[:num_channels_to_prune]
        
        # Zero out pruned channels
        if isinstance(layer, nn.Conv2d):
            layer.weight.data[prune_indices] = 0
        elif isinstance(layer, nn.Linear):
            layer.weight.data[prune_indices] = 0
    
    def prune_model(self):
        """Prune entire model."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, TernaryConv2d)):
                # Calculate number of channels to prune
                if isinstance(module, nn.Conv2d):
                    num_channels = module.out_channels
                else:
                    num_channels = module.out_features
                
                num_to_prune = int(num_channels * self.pruning_ratio)
                
                if num_to_prune > 0:
                    self.prune_layer(module, num_to_prune)
                    print(f'Pruned {num_to_prune}/{num_channels} channels in {name}')
    
    def iterative_pruning(self, train_loader, val_loader, 
                         num_iterations=5, finetune_epochs=10):
        """Iterative pruning with fine-tuning."""
        for iteration in range(num_iterations):
            print(f'\nPruning iteration {iteration + 1}/{num_iterations}')
            
            # Prune
            self.prune_model()
            
            # Fine-tune
            print('Fine-tuning...')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            
            for epoch in range(finetune_epochs):
                train_loss = self.train_epoch(train_loader, optimizer)
                val_acc = self.validate(val_loader)
                print(f'Epoch {epoch}: Loss={train_loss:.4f}, Acc={val_acc:.2f}%')
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100.0 * correct / total

# Usage
pruner = TernaryPruner(model, pruning_ratio=0.3)
pruner.iterative_pruning(train_loader, val_loader, num_iterations=5)
```

### Neural Network Compression Pipeline

```python
class CompressionPipeline:
    """Complete compression pipeline for ternary models."""
    
    def __init__(self, model):
        self.model = model
        self.compression_stats = {}
    
    def measure_model_size(self):
        """Measure model size in MB."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def apply_quantization(self):
        """Apply ternary quantization."""
        print('Step 1: Quantization')
        original_size = self.measure_model_size()
        
        # Convert to ternary
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.quantize_layer(module)
        
        quantized_size = self.measure_model_size()
        
        self.compression_stats['quantization'] = {
            'original_size': original_size,
            'compressed_size': quantized_size,
            'compression_ratio': original_size / quantized_size
        }
        
        print(f'Size: {original_size:.2f} MB -> {quantized_size:.2f} MB')
        print(f'Compression: {original_size/quantized_size:.2f}x')
    
    def apply_pruning(self, pruning_ratio=0.5):
        """Apply structured pruning."""
        print(f'\nStep 2: Pruning ({pruning_ratio*100}%)')
        
        pruner = TernaryPruner(self.model, pruning_ratio)
        pruner.prune_model()
        
        # Calculate sparsity
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params
        self.compression_stats['pruning'] = {
            'pruning_ratio': pruning_ratio,
            'actual_sparsity': sparsity
        }
        
        print(f'Sparsity: {sparsity*100:.2f}%')
    
    def apply_knowledge_distillation(self, teacher, train_loader, epochs=10):
        """Apply knowledge distillation."""
        print(f'\nStep 3: Knowledge Distillation')
        
        distiller = KnowledgeDistillation(teacher, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()
                losses = distiller.train_step(data, target, optimizer)
                total_loss += losses['total_loss']
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}: Loss={avg_loss:.4f}')
    
    def export_compressed_model(self, output_path):
        """Export compressed model."""
        print(f'\nStep 4: Export')
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'compression_stats': self.compression_stats
        }, output_path)
        
        final_size = self.measure_model_size()
        print(f'Final model size: {final_size:.2f} MB')
        print(f'Saved to: {output_path}')
    
    def print_summary(self):
        """Print compression summary."""
        print('\n' + '=' * 60)
        print('COMPRESSION SUMMARY')
        print('=' * 60)
        
        for step, stats in self.compression_stats.items():
            print(f'\n{step.upper()}:')
            for key, value in stats.items():
                print(f'  {key}: {value}')
        
        print('=' * 60)

# Complete pipeline
def compress_model(model, teacher, train_loader):
    """Run complete compression pipeline."""
    pipeline = CompressionPipeline(model)
    
    # Step 1: Quantization
    pipeline.apply_quantization()
    
    # Step 2: Pruning
    pipeline.apply_pruning(pruning_ratio=0.5)
    
    # Step 3: Knowledge distillation
    pipeline.apply_knowledge_distillation(teacher, train_loader, epochs=10)
    
    # Step 4: Export
    pipeline.export_compressed_model('compressed_model.pth')
    
    # Summary
    pipeline.print_summary()
    
    return pipeline.model

# Usage
compressed_model = compress_model(model, teacher_model, train_loader)
```

## State-of-the-Art Ternary Architecture

```python
class SOTATernaryNet(nn.Module):
    """
    State-of-the-art ternary network combining all techniques:
    - Mixed precision
    - Residual connections
    - Attention mechanisms
    - Optimal quantization
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Stem: FP32 for input processing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Mixed precision stages
        self.stage1 = self._make_stage(64, 128, num_blocks=3, ternary=True)
        self.stage2 = self._make_stage(128, 256, num_blocks=4, ternary=True)
        self.stage3 = self._make_stage(256, 512, num_blocks=6, ternary=True)
        
        # Attention
        self.attention = ChannelAttention(512)
        
        # Head: FP32 for final classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, ternary=True):
        """Create a stage with residual blocks."""
        blocks = []
        
        # First block with stride 2
        blocks.append(
            TernaryResidualBlock(in_channels, out_channels, stride=2, ternary=ternary)
        )
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            blocks.append(
                TernaryResidualBlock(out_channels, out_channels, stride=1, ternary=ternary)
            )
        
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.attention(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class TernaryResidualBlock(nn.Module):
    """Residual block with ternary convolutions."""
    
    def __init__(self, in_channels, out_channels, stride=1, ternary=True):
        super().__init__()
        
        if ternary:
            self.conv1 = TernaryConv2d(in_channels, out_channels, 3, 
                                      stride=stride, padding=1)
            self.conv2 = TernaryConv2d(out_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                                  stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Usage
model = SOTATernaryNet(num_classes=10)
print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
```

## Best Practices Summary

1. **Mixed Precision**: Use FP32 for critical layers, ternary for compression
2. **Knowledge Distillation**: Always train student from strong teacher
3. **NAS**: Use architecture search for optimal layer configurations
4. **Hardware Optimization**: Profile and optimize for target device
5. **Compression Pipeline**: Combine quantization, pruning, and distillation
6. **Residual Connections**: Essential for deep ternary networks
7. **Attention**: Improves representational capacity

## Exercises

1. **Custom Precision**: Design a 4-level precision system (FP32, INT8, Ternary, Binary)
2. **Distillation Experiment**: Compare different temperature values
3. **NAS Implementation**: Implement and run evolutionary NAS
4. **CUDA Kernel**: Write optimized CUDA kernel for ternary operations
5. **Complete Pipeline**: Build end-to-end system combining all techniques

## Next Steps

- Explore research papers on ternary networks
- Experiment with custom hardware accelerators
- Contribute optimizations to the Triton project
- Apply to real-world applications

## Additional Resources

- [Binary and Ternary Networks Survey](https://arxiv.org/abs/2003.03488)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- [Efficient Deep Learning](https://efficientdlbook.com/)

## Conclusion

You've now mastered advanced techniques for ternary neural networks. These methods enable you to build highly efficient models that approach full-precision accuracy while maintaining the benefits of extreme compression and fast inference. Combine these techniques creatively to push the boundaries of efficient deep learning!
