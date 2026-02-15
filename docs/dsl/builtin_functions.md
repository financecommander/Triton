# Triton DSL Built-in Functions Reference

This comprehensive reference documents all built-in functions available in Triton DSL, organized by category with complete signatures, descriptions, and examples.

## Table of Contents

- [Tensor Creation](#tensor-creation)
- [Tensor Manipulation](#tensor-manipulation)
- [Mathematical Functions](#mathematical-functions)
- [Linear Algebra](#linear-algebra)
- [Neural Network Operations](#neural-network-operations)
- [Quantization Operations](#quantization-operations)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Utility Functions](#utility-functions)

## Tensor Creation

### zeros

Create a tensor filled with zeros.

**Signature**:
```triton
fn zeros<T: Numeric>(shape: [int, ...], dtype: Type = float32) -> Tensor<T, shape>
```

**Parameters**:
- `shape`: Shape of the output tensor
- `dtype`: Data type of tensor elements (default: float32)

**Returns**: Tensor filled with zeros

**Examples**:
```triton
let a = zeros([3, 4])                    # [3, 4] tensor of float32 zeros
let b = zeros([10, 10], dtype=int32)     # [10, 10] tensor of int32 zeros
let c = zeros([2, 3, 4], dtype=float16)  # [2, 3, 4] tensor of float16 zeros
```

---

### ones

Create a tensor filled with ones.

**Signature**:
```triton
fn ones<T: Numeric>(shape: [int, ...], dtype: Type = float32) -> Tensor<T, shape>
```

**Parameters**:
- `shape`: Shape of the output tensor
- `dtype`: Data type of tensor elements (default: float32)

**Returns**: Tensor filled with ones

**Examples**:
```triton
let a = ones([5, 5])                  # [5, 5] tensor of float32 ones
let b = ones([3, 3], dtype=float64)   # [3, 3] tensor of float64 ones
```

---

### full

Create a tensor filled with a specific value.

**Signature**:
```triton
fn full<T: Numeric>(shape: [int, ...], value: T, dtype: Type = float32) -> Tensor<T, shape>
```

**Parameters**:
- `shape`: Shape of the output tensor
- `value`: Fill value
- `dtype`: Data type (default: float32)

**Returns**: Tensor filled with the specified value

**Examples**:
```triton
let a = full([3, 3], 7)              # [3, 3] tensor filled with 7
let b = full([2, 4], 3.14)           # [2, 4] tensor filled with 3.14
```

---

### eye

Create an identity matrix.

**Signature**:
```triton
fn eye<T: Numeric>(n: int32, m: int32 = n, dtype: Type = float32) -> Tensor<T, [n, m]>
```

**Parameters**:
- `n`: Number of rows
- `m`: Number of columns (default: n)
- `dtype`: Data type (default: float32)

**Returns**: Identity matrix with ones on the diagonal

**Examples**:
```triton
let a = eye(3)                    # [3, 3] identity matrix
let b = eye(4, 5)                 # [4, 5] rectangular identity
let c = eye(3, dtype=float64)     # [3, 3] float64 identity
```

---

### arange

Create a tensor with evenly spaced values.

**Signature**:
```triton
fn arange<T: Numeric>(start: T, end: T, step: T = 1, dtype: Type = float32) -> Tensor<T, [?]>
```

**Parameters**:
- `start`: Start value (inclusive)
- `end`: End value (exclusive)
- `step`: Step size (default: 1)
- `dtype`: Data type (default: float32)

**Returns**: 1D tensor with evenly spaced values

**Examples**:
```triton
let a = arange(0, 10)             # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
let b = arange(0, 1, 0.1)         # [0.0, 0.1, 0.2, ..., 0.9]
let c = arange(10, 0, -1)         # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

---

### linspace

Create a tensor with linearly spaced values.

**Signature**:
```triton
fn linspace<T: Float>(start: T, end: T, num: int32, dtype: Type = float32) -> Tensor<T, [num]>
```

**Parameters**:
- `start`: Start value
- `end`: End value
- `num`: Number of samples
- `dtype`: Data type (default: float32)

**Returns**: 1D tensor with linearly spaced values

**Examples**:
```triton
let a = linspace(0, 1, 11)        # [0.0, 0.1, 0.2, ..., 1.0]
let b = linspace(-1, 1, 5)        # [-1.0, -0.5, 0.0, 0.5, 1.0]
```

---

### randn

Create a tensor with random values from standard normal distribution.

**Signature**:
```triton
fn randn<T: Float>(shape: [int, ...], dtype: Type = float32, seed: int32? = None) -> Tensor<T, shape>
```

**Parameters**:
- `shape`: Shape of output tensor
- `dtype`: Data type (default: float32)
- `seed`: Random seed (optional)

**Returns**: Tensor with random normal values (mean=0, std=1)

**Examples**:
```triton
let a = randn([3, 3])                  # Random normal values
let b = randn([5, 5], seed=42)         # Reproducible random values
```

---

### rand

Create a tensor with random values from uniform distribution [0, 1).

**Signature**:
```triton
fn rand<T: Float>(shape: [int, ...], dtype: Type = float32, seed: int32? = None) -> Tensor<T, shape>
```

**Parameters**:
- `shape`: Shape of output tensor
- `dtype`: Data type (default: float32)
- `seed`: Random seed (optional)

**Returns**: Tensor with random uniform values

**Examples**:
```triton
let a = rand([4, 4])                   # Random uniform [0, 1)
let b = rand([2, 3], seed=123)         # Reproducible random values
```

## Tensor Manipulation

### reshape

Reshape a tensor to a new shape.

**Signature**:
```triton
fn reshape<T, S1, S2>(x: Tensor<T, S1>, shape: [int, ...]) -> Tensor<T, S2>
```

**Parameters**:
- `x`: Input tensor
- `shape`: New shape (use -1 for automatic dimension)

**Returns**: Reshaped tensor (view if possible)

**Examples**:
```triton
let x = arange(0, 12).reshape([3, 4])     # [12] -> [3, 4]
let y = x.reshape([2, 6])                 # [3, 4] -> [2, 6]
let z = x.reshape([-1])                   # Flatten to 1D
let w = x.reshape([2, -1])                # [3, 4] -> [2, 6] (auto-calc)
```

---

### transpose

Transpose dimensions of a tensor.

**Signature**:
```triton
fn transpose<T, S>(x: Tensor<T, S>, dim0: int32? = -2, dim1: int32? = -1) -> Tensor<T, S'>
```

**Parameters**:
- `x`: Input tensor
- `dim0`: First dimension to swap (default: -2)
- `dim1`: Second dimension to swap (default: -1)

**Returns**: Transposed tensor

**Examples**:
```triton
let a = tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
let b = a.transpose()                    # [3, 2]
let c = a.transpose(0, 1)                # [3, 2] (explicit dims)

let d = randn([2, 3, 4])
let e = d.transpose(0, 2)                # [4, 3, 2]
```

---

### permute

Permute dimensions of a tensor.

**Signature**:
```triton
fn permute<T, S>(x: Tensor<T, S>, dims: [int, ...]) -> Tensor<T, S'>
```

**Parameters**:
- `x`: Input tensor
- `dims`: Permutation of dimensions

**Returns**: Tensor with permuted dimensions

**Examples**:
```triton
let x = randn([2, 3, 4, 5])
let y = x.permute([0, 2, 1, 3])         # [2, 4, 3, 5]
let z = x.permute([3, 2, 1, 0])         # [5, 4, 3, 2]

# Convert NCHW to NHWC
let nchw = randn([1, 3, 224, 224])
let nhwc = nchw.permute([0, 2, 3, 1])   # [1, 224, 224, 3]
```

---

### squeeze

Remove dimensions of size 1.

**Signature**:
```triton
fn squeeze<T, S>(x: Tensor<T, S>, dim: int32? = None) -> Tensor<T, S'>
```

**Parameters**:
- `x`: Input tensor
- `dim`: Dimension to squeeze (optional, squeezes all if None)

**Returns**: Tensor with singleton dimensions removed

**Examples**:
```triton
let x = randn([1, 3, 1, 5])
let y = x.squeeze()                     # [3, 5]
let z = x.squeeze(0)                    # [3, 1, 5]
let w = x.squeeze(2)                    # [1, 3, 5]
```

---

### unsqueeze

Add a dimension of size 1.

**Signature**:
```triton
fn unsqueeze<T, S>(x: Tensor<T, S>, dim: int32) -> Tensor<T, S'>
```

**Parameters**:
- `x`: Input tensor
- `dim`: Position to insert new dimension

**Returns**: Tensor with new dimension added

**Examples**:
```triton
let x = randn([3, 4])
let y = x.unsqueeze(0)                  # [1, 3, 4]
let z = x.unsqueeze(1)                  # [3, 1, 4]
let w = x.unsqueeze(-1)                 # [3, 4, 1]
```

---

### concat

Concatenate tensors along a dimension.

**Signature**:
```triton
fn concat<T, S>(tensors: [Tensor<T, S>, ...], dim: int32 = 0) -> Tensor<T, S'>
```

**Parameters**:
- `tensors`: List of tensors to concatenate
- `dim`: Dimension along which to concatenate

**Returns**: Concatenated tensor

**Examples**:
```triton
let a = randn([2, 3])
let b = randn([2, 3])
let c = concat([a, b], dim=0)           # [4, 3]
let d = concat([a, b], dim=1)           # [2, 6]

let e = randn([1, 3, 4])
let f = randn([1, 3, 4])
let g = concat([e, f], dim=0)           # [2, 3, 4]
```

---

### stack

Stack tensors along a new dimension.

**Signature**:
```triton
fn stack<T, S>(tensors: [Tensor<T, S>, ...], dim: int32 = 0) -> Tensor<T, S'>
```

**Parameters**:
- `tensors`: List of tensors to stack (must have same shape)
- `dim`: Dimension along which to stack

**Returns**: Stacked tensor

**Examples**:
```triton
let a = randn([3, 4])
let b = randn([3, 4])
let c = stack([a, b], dim=0)            # [2, 3, 4]
let d = stack([a, b], dim=1)            # [3, 2, 4]
```

---

### split

Split a tensor into chunks.

**Signature**:
```triton
fn split<T, S>(x: Tensor<T, S>, sizes: [int, ...] | int32, dim: int32 = 0) 
    -> [Tensor<T, S'>, ...]
```

**Parameters**:
- `x`: Input tensor
- `sizes`: Chunk sizes or number of equal chunks
- `dim`: Dimension along which to split

**Returns**: List of tensor chunks

**Examples**:
```triton
let x = randn([10, 3])
let chunks = split(x, 2, dim=0)         # Two [5, 3] tensors
let parts = split(x, [3, 3, 4], dim=0)  # [3,3], [3,3], [4,3] tensors
```

## Mathematical Functions

### Trigonometric Functions

#### sin

Compute sine element-wise.

**Signature**:
```triton
fn sin<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = linspace(0, 3.14159, 10)
let y = sin(x)                          # Sine of each element
```

#### cos

Compute cosine element-wise.

**Signature**:
```triton
fn cos<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = linspace(0, 3.14159, 10)
let y = cos(x)                          # Cosine of each element
```

#### tan

Compute tangent element-wise.

**Signature**:
```triton
fn tan<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = linspace(-1.5, 1.5, 10)
let y = tan(x)                          # Tangent of each element
```

---

### Exponential and Logarithmic

#### exp

Compute exponential element-wise.

**Signature**:
```triton
fn exp<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = linspace(0, 2, 5)
let y = exp(x)                          # [1, e^0.5, e^1, e^1.5, e^2]
```

#### log

Compute natural logarithm element-wise.

**Signature**:
```triton
fn log<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1, 2, 4, 8])
let y = log(x)                          # [0, 0.693, 1.386, 2.079]
```

#### log10

Compute base-10 logarithm element-wise.

**Signature**:
```triton
fn log10<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1, 10, 100, 1000])
let y = log10(x)                        # [0, 1, 2, 3]
```

---

### Power and Root Functions

#### pow

Compute power element-wise.

**Signature**:
```triton
fn pow<T: Numeric, S>(x: Tensor<T, S>, exponent: T | Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1, 2, 3, 4])
let y = pow(x, 2)                       # [1, 4, 9, 16]
let z = pow(x, x)                       # [1, 4, 27, 256]
```

#### sqrt

Compute square root element-wise.

**Signature**:
```triton
fn sqrt<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1, 4, 9, 16])
let y = sqrt(x)                         # [1, 2, 3, 4]
```

---

### Rounding Functions

#### abs

Compute absolute value element-wise.

**Signature**:
```triton
fn abs<T: Numeric, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([-1, -2, 3, 4])
let y = abs(x)                          # [1, 2, 3, 4]
```

#### ceil

Round up to nearest integer element-wise.

**Signature**:
```triton
fn ceil<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1.2, 2.5, 3.7])
let y = ceil(x)                         # [2, 3, 4]
```

#### floor

Round down to nearest integer element-wise.

**Signature**:
```triton
fn floor<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1.2, 2.5, 3.7])
let y = floor(x)                        # [1, 2, 3]
```

#### round

Round to nearest integer element-wise.

**Signature**:
```triton
fn round<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Examples**:
```triton
let x = tensor([1.2, 2.5, 3.7])
let y = round(x)                        # [1, 2, 4]
```

---

### Statistical Functions

#### sum

Sum tensor elements.

**Signature**:
```triton
fn sum<T: Numeric, S>(x: Tensor<T, S>, dim: int32? = None, keepdim: bool = false) 
    -> Tensor<T, S'> | T
```

**Parameters**:
- `x`: Input tensor
- `dim`: Dimension to reduce (None for all)
- `keepdim`: Keep reduced dimension as size 1

**Examples**:
```triton
let x = tensor([[1, 2, 3], [4, 5, 6]])
let total = sum(x)                      # 21
let col_sum = sum(x, dim=0)             # [5, 7, 9]
let row_sum = sum(x, dim=1)             # [6, 15]
```

#### mean

Compute mean of tensor elements.

**Signature**:
```triton
fn mean<T: Float, S>(x: Tensor<T, S>, dim: int32? = None, keepdim: bool = false) 
    -> Tensor<T, S'> | T
```

**Examples**:
```triton
let x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
let avg = mean(x)                       # 3.5
let col_mean = mean(x, dim=0)           # [2.5, 3.5, 4.5]
```

#### std

Compute standard deviation.

**Signature**:
```triton
fn std<T: Float, S>(x: Tensor<T, S>, dim: int32? = None, keepdim: bool = false) 
    -> Tensor<T, S'> | T
```

**Examples**:
```triton
let x = randn([100, 10])
let total_std = std(x)                  # Scalar
let feature_std = std(x, dim=0)         # [10]
```

#### max

Find maximum values.

**Signature**:
```triton
fn max<T: Comparable, S>(x: Tensor<T, S>, dim: int32? = None, keepdim: bool = false) 
    -> Tensor<T, S'> | T
```

**Examples**:
```triton
let x = tensor([[1, 5, 3], [7, 2, 8]])
let maximum = max(x)                    # 8
let col_max = max(x, dim=0)             # [7, 5, 8]
```

#### min

Find minimum values.

**Signature**:
```triton
fn min<T: Comparable, S>(x: Tensor<T, S>, dim: int32? = None, keepdim: bool = false) 
    -> Tensor<T, S'> | T
```

**Examples**:
```triton
let x = tensor([[1, 5, 3], [7, 2, 8]])
let minimum = min(x)                    # 1
let col_min = min(x, dim=0)             # [1, 2, 3]
```

## Linear Algebra

### matmul

Matrix multiplication.

**Signature**:
```triton
fn matmul<T: Numeric, M, K, N>(
    a: Tensor<T, [M, K]>, 
    b: Tensor<T, [K, N]>
) -> Tensor<T, [M, N]>
```

**Parameters**:
- `a`: First matrix [M, K]
- `b`: Second matrix [K, N]

**Returns**: Product matrix [M, N]

**Examples**:
```triton
let a = randn([128, 256])
let b = randn([256, 512])
let c = matmul(a, b)                    # [128, 512]

# Operator form
let d = a @ b                           # Same as matmul
```

---

### dot

Dot product of vectors.

**Signature**:
```triton
fn dot<T: Numeric, N>(a: Tensor<T, [N]>, b: Tensor<T, [N]>) -> T
```

**Examples**:
```triton
let a = tensor([1, 2, 3])
let b = tensor([4, 5, 6])
let result = dot(a, b)                  # 1*4 + 2*5 + 3*6 = 32
```

---

### outer

Outer product of vectors.

**Signature**:
```triton
fn outer<T: Numeric, M, N>(a: Tensor<T, [M]>, b: Tensor<T, [N]>) -> Tensor<T, [M, N]>
```

**Examples**:
```triton
let a = tensor([1, 2, 3])
let b = tensor([4, 5])
let result = outer(a, b)                # [[4, 5], [8, 10], [12, 15]]
```

## Neural Network Operations

### conv2d

2D convolution operation.

**Signature**:
```triton
fn conv2d<T: Numeric>(
    input: Tensor<T, [batch, in_channels, height, width]>,
    weight: Tensor<T, [out_channels, in_channels, kernel_h, kernel_w]>,
    bias: Tensor<T, [out_channels]>? = None,
    stride: int32 | [int32, int32] = 1,
    padding: int32 | [int32, int32] = 0,
    dilation: int32 | [int32, int32] = 1,
    groups: int32 = 1
) -> Tensor<T, [batch, out_channels, out_h, out_w]>
```

**Parameters**:
- `input`: Input tensor in NCHW format
- `weight`: Convolution kernel weights
- `bias`: Optional bias term
- `stride`: Stride of convolution
- `padding`: Zero padding added to input
- `dilation`: Spacing between kernel elements
- `groups`: Number of blocked connections

**Examples**:
```triton
let input = randn([1, 3, 224, 224])
let kernel = randn([64, 3, 7, 7])
let output = conv2d(input, kernel, stride=2, padding=3)  # [1, 64, 112, 112]

# With bias
let bias = zeros([64])
let output2 = conv2d(input, kernel, bias=bias, stride=2, padding=3)
```

---

### maxpool2d

2D max pooling operation.

**Signature**:
```triton
fn maxpool2d<T: Comparable>(
    input: Tensor<T, [batch, channels, height, width]>,
    kernel_size: int32 | [int32, int32],
    stride: int32 | [int32, int32]? = None,
    padding: int32 | [int32, int32] = 0
) -> Tensor<T, [batch, channels, out_h, out_w]>
```

**Parameters**:
- `input`: Input tensor in NCHW format
- `kernel_size`: Size of pooling window
- `stride`: Stride of pooling (default: kernel_size)
- `padding`: Zero padding

**Examples**:
```triton
let input = randn([1, 64, 112, 112])
let pooled = maxpool2d(input, kernel_size=2, stride=2)  # [1, 64, 56, 56]
let pooled2 = maxpool2d(input, kernel_size=3, stride=2, padding=1)
```

---

### avgpool2d

2D average pooling operation.

**Signature**:
```triton
fn avgpool2d<T: Numeric>(
    input: Tensor<T, [batch, channels, height, width]>,
    kernel_size: int32 | [int32, int32],
    stride: int32 | [int32, int32]? = None,
    padding: int32 | [int32, int32] = 0
) -> Tensor<T, [batch, channels, out_h, out_w]>
```

**Examples**:
```triton
let input = randn([1, 64, 112, 112])
let pooled = avgpool2d(input, kernel_size=2, stride=2)  # [1, 64, 56, 56]
```

---

### batch_norm

Batch normalization.

**Signature**:
```triton
fn batch_norm<T: Float>(
    input: Tensor<T, [batch, channels, ...]>,
    gamma: Tensor<T, [channels]>,
    beta: Tensor<T, [channels]>,
    running_mean: Tensor<T, [channels]>? = None,
    running_var: Tensor<T, [channels]>? = None,
    training: bool = true,
    momentum: T = 0.1,
    eps: T = 1e-5
) -> Tensor<T, [batch, channels, ...]>
```

**Parameters**:
- `input`: Input tensor
- `gamma`: Scale parameter
- `beta`: Shift parameter
- `running_mean`: Running mean for inference
- `running_var`: Running variance for inference
- `training`: Training mode flag
- `momentum`: Momentum for running statistics
- `eps`: Small constant for numerical stability

**Examples**:
```triton
let x = randn([32, 64, 28, 28])
let gamma = ones([64])
let beta = zeros([64])
let normalized = batch_norm(x, gamma, beta)
```

---

### layer_norm

Layer normalization.

**Signature**:
```triton
fn layer_norm<T: Float, S>(
    input: Tensor<T, S>,
    normalized_shape: [int, ...],
    gamma: Tensor<T, normalized_shape>? = None,
    beta: Tensor<T, normalized_shape>? = None,
    eps: T = 1e-5
) -> Tensor<T, S>
```

**Examples**:
```triton
let x = randn([32, 128])
let normalized = layer_norm(x, [128])

# With learnable parameters
let gamma = ones([128])
let beta = zeros([128])
let normalized2 = layer_norm(x, [128], gamma, beta)
```

---

### dropout

Dropout regularization.

**Signature**:
```triton
fn dropout<T: Float, S>(
    input: Tensor<T, S>,
    p: T = 0.5,
    training: bool = true
) -> Tensor<T, S>
```

**Parameters**:
- `input`: Input tensor
- `p`: Dropout probability
- `training`: Apply dropout if true

**Examples**:
```triton
let x = randn([32, 128])
let dropped = dropout(x, p=0.5, training=true)
let inference = dropout(x, p=0.5, training=false)  # No dropout
```

## Quantization Operations

### ternary_quantize

Quantize tensor to ternary values {-1, 0, 1}.

**Signature**:
```triton
fn ternary_quantize<S>(
    x: Tensor<float32, S>,
    threshold: float32 = 0.5,
    method: str = "threshold"
) -> TernaryTensor<trit, S>
```

**Parameters**:
- `x`: Input tensor
- `threshold`: Quantization threshold
- `method`: Quantization method ("threshold", "stochastic", "learned")

**Returns**: Ternary tensor with values in {-1, 0, 1}

**Examples**:
```triton
let x = randn([128, 256])
let quantized = ternary_quantize(x)
let custom = ternary_quantize(x, threshold=0.7)
let stochastic = ternary_quantize(x, method="stochastic")
```

---

### ternary_dequantize

Convert ternary tensor to floating-point.

**Signature**:
```triton
fn ternary_dequantize<S>(
    x: TernaryTensor<trit, S>,
    scale: float32 = 1.0
) -> Tensor<float32, S>
```

**Parameters**:
- `x`: Ternary tensor
- `scale`: Scaling factor

**Examples**:
```triton
let ternary = ternary_quantize(randn([10, 10]))
let float_tensor = ternary_dequantize(ternary)
let scaled = ternary_dequantize(ternary, scale=2.0)
```

---

### ternary_weights

Initialize ternary weight tensor.

**Signature**:
```triton
fn ternary_weights(
    shape: [int, ...],
    sparsity: float32 = 0.0,
    distribution: str = "uniform"
) -> TernaryTensor<trit, shape>
```

**Parameters**:
- `shape`: Weight tensor shape
- `sparsity`: Fraction of zeros (0.0 to 1.0)
- `distribution`: Initialization distribution

**Examples**:
```triton
let w1 = ternary_weights([784, 256])
let w2 = ternary_weights([256, 128], sparsity=0.3)
let w3 = ternary_weights([128, 10], distribution="balanced")
```

---

### ste_quantize

Straight-through estimator quantization (gradient-preserving).

**Signature**:
```triton
fn ste_quantize<S>(x: Tensor<float32, S>) -> Tensor<float32, S>
```

**Description**: Quantizes in forward pass, passes gradients unchanged in backward pass.

**Examples**:
```triton
fn qat_layer(x: Tensor<float32, [?, ?]>, weights: Tensor<float32, [?, ?]>) 
    -> Tensor<float32, [?, ?]> {
    let quantized_weights = ste_quantize(ternary_quantize(weights))
    return x @ quantized_weights
}
```

## Activation Functions

### relu

Rectified Linear Unit activation.

**Signature**:
```triton
fn relu<T: Numeric, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Formula**: `max(0, x)`

**Examples**:
```triton
let x = tensor([-2, -1, 0, 1, 2])
let activated = relu(x)                 # [0, 0, 0, 1, 2]
```

---

### sigmoid

Sigmoid activation function.

**Signature**:
```triton
fn sigmoid<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Formula**: `1 / (1 + exp(-x))`

**Examples**:
```triton
let x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
let activated = sigmoid(x)              # [0.119, 0.269, 0.5, 0.731, 0.881]
```

---

### tanh

Hyperbolic tangent activation.

**Signature**:
```triton
fn tanh<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Formula**: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**Examples**:
```triton
let x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
let activated = tanh(x)                 # [-0.964, -0.762, 0, 0.762, 0.964]
```

---

### softmax

Softmax activation function.

**Signature**:
```triton
fn softmax<T: Float, S>(x: Tensor<T, S>, dim: int32 = -1) -> Tensor<T, S>
```

**Formula**: `exp(x_i) / sum(exp(x_j))`

**Examples**:
```triton
let logits = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
let probs = softmax(logits, dim=1)     # Probabilities sum to 1 along dim 1
```

---

### gelu

Gaussian Error Linear Unit activation.

**Signature**:
```triton
fn gelu<T: Float, S>(x: Tensor<T, S>) -> Tensor<T, S>
```

**Formula**: `x * Φ(x)` where Φ is the cumulative distribution function

**Examples**:
```triton
let x = randn([32, 128])
let activated = gelu(x)
```

## Loss Functions

### cross_entropy

Cross-entropy loss for classification.

**Signature**:
```triton
fn cross_entropy<T: Float>(
    logits: Tensor<T, [batch, num_classes]>,
    targets: Tensor<int32, [batch]>,
    reduction: str = "mean"
) -> Tensor<T, []> | Tensor<T, [batch]>
```

**Parameters**:
- `logits`: Model predictions (raw scores)
- `targets`: Ground truth class indices
- `reduction`: "mean", "sum", or "none"

**Examples**:
```triton
let logits = randn([32, 10])
let targets = tensor([0, 1, 2, ..., 9])
let loss = cross_entropy(logits, targets)
```

---

### mse_loss

Mean squared error loss.

**Signature**:
```triton
fn mse_loss<T: Float, S>(
    predictions: Tensor<T, S>,
    targets: Tensor<T, S>,
    reduction: str = "mean"
) -> Tensor<T, []> | Tensor<T, S>
```

**Examples**:
```triton
let pred = randn([32, 10])
let target = randn([32, 10])
let loss = mse_loss(pred, target)
```

## Utility Functions

### print

Print tensor or value to console.

**Signature**:
```triton
fn print<T>(value: T, label: str? = None) -> void
```

**Examples**:
```triton
let x = randn([3, 3])
print(x)
print(x, label="Tensor X")
```

---

### assert

Assert a condition is true.

**Signature**:
```triton
fn assert(condition: bool, message: str? = None) -> void
```

**Examples**:
```triton
assert x.shape[0] == 32, "Batch size must be 32"
assert all(x >= -1) && all(x <= 1), "Values out of range"
```

---

### clip

Clamp values to a range.

**Signature**:
```triton
fn clip<T: Comparable, S>(x: Tensor<T, S>, min_val: T, max_val: T) -> Tensor<T, S>
```

**Examples**:
```triton
let x = randn([10, 10])
let clipped = clip(x, -1.0, 1.0)       # Values in [-1, 1]
```

This comprehensive reference covers all major built-in functions in Triton DSL. Refer to specific function documentation for additional parameters and advanced usage patterns.
