"""Ternary Neural Network Model Implementations."""

import torch
import torch.nn as nn
from typing import Optional, List
from .ternary_tensor import TernaryLinear, TernaryConv2d


# ============================================================================
# ResNet-18 Ternary Implementation
# ============================================================================

class TernaryBasicBlock(nn.Module):
    """Basic ResNet block with ternary convolutions."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = TernaryConv2d(in_planes, planes, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = TernaryConv2d(planes, planes, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                TernaryConv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class TernaryResNet18(nn.Module):
    """ResNet-18 with ternary weights."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = TernaryConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = TernaryLinear(512, num_classes)
        
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(TernaryBasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * TernaryBasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ============================================================================
# MobileNetV2 Ternary Implementation
# ============================================================================

class TernaryInvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block with ternary weights."""
    
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # Pointwise
            layers.extend([
                TernaryConv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            # Depthwise
            TernaryConv2d(hidden_dim, hidden_dim, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise-linear
            TernaryConv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TernaryMobileNetV2(nn.Module):
    """MobileNetV2 with ternary weights."""
    
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        
        # Building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult)
        
        self.features = nn.Sequential(
            TernaryConv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )
        
        # Building inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        features_list = [self.features]
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features_list.append(
                    TernaryInvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # Building last several layers
        features_list.append(
            nn.Sequential(
                TernaryConv2d(input_channel, self.last_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True),
            )
        )
        
        self.features = nn.Sequential(*features_list)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            TernaryLinear(self.last_channel, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# BERT-tiny Ternary Implementation
# ============================================================================

class TernaryBertAttention(nn.Module):
    """Simplified BERT attention with ternary weights."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = TernaryLinear(hidden_size, hidden_size)
        self.key = TernaryLinear(hidden_size, hidden_size)
        self.value = TernaryLinear(hidden_size, hidden_size)
        self.output = TernaryLinear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.output(context)
        return output


class TernaryBertLayer(nn.Module):
    """BERT layer with ternary weights."""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = TernaryBertAttention(hidden_size, num_heads)
        self.intermediate = TernaryLinear(hidden_size, intermediate_size)
        self.output = TernaryLinear(intermediate_size, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        attn_output = self.attention(x)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.output(torch.relu(self.intermediate(x)))
        x = self.layernorm2(x + ff_output)
        
        return x


class TernaryBertTiny(nn.Module):
    """BERT-tiny model with ternary weights (2 layers, 128 hidden, 2 heads)."""
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 128, 
                 num_layers: int = 2, num_heads: int = 2, 
                 intermediate_size: int = 512, max_seq_length: int = 512,
                 num_classes: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TernaryBertLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = TernaryLinear(hidden_size, num_classes)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = self.layernorm(token_embeds + position_embeds)
        
        # Transformer layers
        hidden = embeddings
        for layer in self.layers:
            hidden = layer(hidden)
        
        # Classification (use [CLS] token - first token)
        cls_output = hidden[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


# ============================================================================
# Simple Models for MNIST and CIFAR-10
# ============================================================================

class TernaryMNISTNet(nn.Module):
    """Simple CNN for MNIST with ternary weights."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = TernaryConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = TernaryLinear(64 * 7 * 7, 128)
        self.fc2 = TernaryLinear(128, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TernaryCIFAR10Net(nn.Module):
    """Simple CNN for CIFAR-10 with ternary weights."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = TernaryConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = TernaryConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = TernaryConv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = TernaryLinear(256 * 4 * 4, 256)
        self.fc2 = TernaryLinear(256, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
