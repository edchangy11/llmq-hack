#ifndef STANDALONE_ATTENTION_H
#define STANDALONE_ATTENTION_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>

// Forward declarations
typedef struct cudnnContext* cudnnHandle_t;

// cuDNN attention functions
void attention_forward_cudnn(nv_bfloat16* out,  // output: (B, T, Hq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + 2*Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_backward_cudnn(nv_bfloat16* dqkvr,                                       // output
                              const float* stats,
                              const nv_bfloat16* dout, const nv_bfloat16* qkvr, const nv_bfloat16* o, // inputs
                              std::byte* workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle);
cudnnHandle_t create_cudnn_handle();

// Custom CUDA attention functions
extern "C" {
    cudaError_t attention_forward_cuda(float* out, float* stats,
                                      const float* qkv, int B, int T, int Hq, int Hkv, int Hs,
                                      cudaStream_t stream);

    cudaError_t attention_forward_cuda_bf16(nv_bfloat16* out, float* stats,
                                           const nv_bfloat16* qkv, int B, int T, int Hq, int Hkv, int Hs,
                                           cudaStream_t stream);
}

#endif // STANDALONE_ATTENTION_H