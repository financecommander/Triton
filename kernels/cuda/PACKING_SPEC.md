# Ternary 2-bit Packing Specification

## Overview
This document describes the 2-bit packing scheme used to compress ternary values {-1, 0, 1} into compact byte storage.

## Encoding

Each trit (ternary digit) is encoded using 2 bits:

| Value | Encoding | Binary | Hex |
|-------|----------|--------|-----|
| -1    | 00       | 0b00   | 0x0 |
| 0     | 01       | 0b01   | 0x1 |
| 1     | 10       | 0b10   | 0x2 |
| (unused) | 11    | 0b11   | 0x3 |

## Packing Layout

Four trits are packed into a single byte (8 bits):

```
Byte: [t3_bit1 t3_bit0 | t2_bit1 t2_bit0 | t1_bit1 t1_bit0 | t0_bit1 t0_bit0]
Bits:  7       6       | 5       4       | 3       2       | 1       0
```

### Example 1: Pack [1, 0, -1, 1]
```
t0 = 1  → 10 (0x2)
t1 = 0  → 01 (0x1)
t2 = -1 → 00 (0x0)
t3 = 1  → 10 (0x2)

Packed byte = 10 00 01 10 (binary) = 0x86 (hex) = 134 (decimal)
              ↑  ↑  ↑  ↑
              t3 t2 t1 t0
```

### Example 2: Pack [-1, -1, -1, -1]
```
t0 = -1 → 00 (0x0)
t1 = -1 → 00 (0x0)
t2 = -1 → 00 (0x0)
t3 = -1 → 00 (0x0)

Packed byte = 00 00 00 00 (binary) = 0x00 (hex) = 0 (decimal)
```

### Example 3: Pack [0, 0, 0, 0]
```
t0 = 0 → 01 (0x1)
t1 = 0 → 01 (0x1)
t2 = 0 → 01 (0x1)
t3 = 0 → 01 (0x1)

Packed byte = 01 01 01 01 (binary) = 0x55 (hex) = 85 (decimal)
```

## Unpacking

To extract trit at index i (0-3):

```c
int shift = i * 2;              // Calculate bit position
uint8_t bits = (packed >> shift) & 0x03;  // Extract 2 bits

// Decode
if (bits == 0x00) return -1;
if (bits == 0x01) return 0;
if (bits == 0x02) return 1;
```

### Example: Unpack 0x86
```
packed = 0x86 = 10 00 01 10 (binary)

Index 0: (0x86 >> 0) & 0x03 = 10 → 1
Index 1: (0x86 >> 2) & 0x03 = 01 → 0
Index 2: (0x86 >> 4) & 0x03 = 00 → -1
Index 3: (0x86 >> 6) & 0x03 = 10 → 1

Result: [1, 0, -1, 1] ✓
```

## Memory Layout for Matrices

### Matrix A (M × K)
- Row-major storage
- Each row padded to multiple of 4 elements
- Total packed size: `M × ⌈K/4⌉` bytes

Example: 3×5 matrix
```
Original (15 elements):
[a00 a01 a02 a03 a04]
[a10 a11 a12 a13 a14]
[a20 a21 a22 a23 a24]

Padded (3 rows × 8 elements):
[a00 a01 a02 a03 | a04   0   0   0]
[a10 a11 a12 a13 | a14   0   0   0]
[a20 a21 a22 a23 | a24   0   0   0]

Packed (3 rows × 2 bytes):
[byte0 byte1] ← row 0 (8 elements → 2 bytes)
[byte0 byte1] ← row 1
[byte0 byte1] ← row 2

Total: 6 bytes (vs 15 bytes unpacked)
```

## Compression Ratio

| Original Type | Size/Element | Packed Size | Compression |
|---------------|--------------|-------------|-------------|
| float32       | 4 bytes      | 0.25 bytes  | 16x         |
| int8          | 1 byte       | 0.25 bytes  | 4x          |

## Implementation

### CUDA Device Functions

```c
// Pack 4 trits into 1 byte
__device__ uint8_t pack_4trits(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    auto encode = [](int8_t t) -> uint8_t {
        if (t == -1) return 0x00;
        if (t == 0) return 0x01;
        if (t == 1) return 0x02;
        return 0x01;  // Default to 0
    };
    
    return (encode(t0) << 0) |
           (encode(t1) << 2) |
           (encode(t2) << 4) |
           (encode(t3) << 6);
}

// Extract single trit from packed byte
__device__ int8_t extract_trit(uint8_t packed, int index) {
    int shift = index * 2;
    uint8_t bits = (packed >> shift) & 0x03;
    
    if (bits == 0x00) return -1;
    if (bits == 0x01) return 0;
    if (bits == 0x02) return 1;
    return 0;  // Invalid
}
```

## Advantages

1. **Memory Efficiency**: 4x compression over int8 storage
2. **Bandwidth**: Reduced memory traffic improves performance
3. **Cache**: More data fits in GPU cache
4. **Simple Encoding**: Fast pack/unpack with bit operations
5. **Lossless**: Exact representation of ternary values

## Disadvantages

1. **Alignment**: Requires padding to multiple of 4
2. **Random Access**: Must unpack to access individual elements
3. **Overhead**: Pack/unpack adds computation (but memory savings dominate)

## Validation

Verify pack/unpack round-trip:
```python
original = [-1, 0, 1, -1, 0, 1, 1, -1]
packed = pack_ternary(original)
unpacked = unpack_ternary(packed, len(original))
assert all(original == unpacked)
```
