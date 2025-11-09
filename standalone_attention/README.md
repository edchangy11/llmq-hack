# Standalone Attention Implementations

This folder contains isolated implementations of two attention mechanisms from the main codebase:

1. **cuDNN Flash Attention** (`cudnn_att.cpp`) - Uses cuDNN Frontend API
2. **Custom CUDA Kernel** (`attention_cuda.cu`) - Memory-efficient attention implementation

## What's Included

### Core Files
- `cudnn_att.cpp` - cuDNN frontend implementation with graph caching
- `attention_cuda.cu` - Custom CUDA kernel with cooperative groups
- `attention.h` - Header with function declarations
- `test_main.cpp` - Test program comparing both implementations

### Build System
- `CMakeLists.txt` - Complete build configuration with cuDNN frontend fetch

## Key Differences Between Implementations

| Feature | cuDNN Implementation | Custom CUDA Kernel |
|---------|---------------------|---------------------|
| **Precision** | BF16 I/O, FP32 compute | FP32 or BF16 |
| **Memory Layout** | `(B,T,H,HS)` interleaved QKV | `(B,T,H,HS)` interleaved QKV |
| **Optimization** | Hardware-optimized Flash Attention | Memory-efficient with split-K |
| **Graph Compilation** | Yes (cached per shape) | No |
| **First Run** | Slow (~100ms, builds graph) | Fast |
| **Subsequent Runs** | Fast (uses cached graph) | Fast |
| **Head Dimensions** | Any supported by cuDNN | Template: 64, 128 only |

## How to Compile

### Prerequisites
- CUDA Toolkit (11.8+)
- cuDNN (8.8+)
- CMake 3.25+
- C++20 compiler

### Environment Setup
```bash
# Set cuDNN paths if not in default locations
export CUDNN_LIBRARY_PATH=/path/to/cudnn/lib64
export CUDNN_INCLUDE_PATH=/path/to/cudnn/include
```

### Build Steps
```bash
cd standalone_attention

# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile library
make attention_lib

# Compile test program (optional)
make test_attention

# Run test
./test_attention
```

### Alternative: Direct Compilation
```bash
# Compile library directly
nvcc -shared -o libattention.so \
  cudnn_att.cpp attention_cuda.cu \
  -lcudnn -lcudnn_frontend \
  -I. -std=c++20 --expt-relaxed-constexpr

# Compile test
nvcc -o test test_main.cpp -L. -lattention -lcudnn -lcudnn_frontend
```

## Usage Example

### Basic Usage
```cpp
#include "attention.h"

// Create cuDNN handle
cudnnHandle_t handle = create_cudnn_handle();

// Allocate workspace
size_t ws_size = cudnn_get_workspace_size(B, T, Hq, Hkv, HS, handle);
std::byte* workspace;
cudaMalloc(&workspace, ws_size);

// Run cuDNN attention
attention_forward_cudnn(out, stats, qkv, workspace, handle, 
                       B, T, Hq, Hkv, HS, stream);

// Or run custom CUDA kernel
attention_forward_cuda(out, stats, qkv, B, T, Hq, Hkv, HS, stream);
```

### Understanding the Memory Layout

Both implementations expect QKV in interleaved format:
```
Input shape: (B, T, Hq + 2*Hkv, HS)
Memory layout: [Q_heads][K_heads][V_heads] per timestep

For B=2, T=512, Hq=32, Hkv=8, HS=128:
- Total input size: 2 * 512 * (32 + 2*8) * 128 = 6,291,456 elements
- Q starts at offset: 0
- K starts at offset: 32 * 128 = 4,096 per timestep  
- V starts at offset: (32 + 8) * 128 = 5,120 per timestep
```

## What You Can Experiment With

### Performance Comparison
- **First vs cached calls** - cuDNN graph compilation overhead
- **Precision effects** - BF16 vs FP32 accuracy/speed tradeoffs  
- **Shape dependencies** - how different B/T/H values affect performance

### Implementation Details
- **Graph caching** - observe the cache behavior in `lookup_cache_or_build_graph_fwd()`
- **Memory access patterns** - stride configurations in tensor attributes
- **Attention scaling** - `1.0 / sqrtf(HS)` applied consistently

### Modifications
- **Add new head dimensions** - extend template specializations in custom kernel
- **Change precision** - modify data types in cuDNN graph setup
- **Add new layouts** - experiment with different stride patterns

## Expected Output

Running the test should show:
```
=== Standalone Attention Test ===
Configuration:
  Batch: 2, Seq: 128, Hq: 8, Hkv: 2, HeadDim: 64

=== Testing Custom CUDA Kernel (FP32) ===
✓ Custom CUDA kernel completed in 0.245 ms

=== Testing cuDNN Kernel (BF16) ===
Workspace size: 1 MB
✓ cuDNN kernel (first call) completed in 15.332 ms  # Graph build overhead
✓ cuDNN kernel (cached) completed in 0.156 ms      # Fast execution

=== Comparing Results ===
RMSE between cuDNN(BF16) and CUDA(FP32): 0.003
✓ Results match within expected precision!
```

The large difference in first vs cached cuDNN calls demonstrates the graph compilation overhead, which is amortized over many training steps.

## Integration with PyTorch

To compare with PyTorch's `scaled_dot_product_attention`:

1. Use same random seed and input data
2. Convert between PyTorch tensors and raw memory layouts
3. Account for stride differences: PyTorch uses `(B, H, T, HS)`, this code uses `(B, T, H, HS)`
4. Both should produce similar results within numerical precision

This standalone version gives you a clean environment to understand the attention implementations without the complexity of the full training framework.