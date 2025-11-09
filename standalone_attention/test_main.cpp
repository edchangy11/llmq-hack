#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "attention.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

void fill_random_bf16(nv_bfloat16* data, size_t count, float mean = 0.0f, float std = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dis(mean, std);
    
    for (size_t i = 0; i < count; i++) {
        data[i] = __float2bfloat16(dis(gen));
    }
}

void fill_random_f32(float* data, size_t count, float mean = 0.0f, float std = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dis(mean, std);
    
    for (size_t i = 0; i < count; i++) {
        data[i] = dis(gen);
    }
}

float compute_rmse(const float* a, const float* b, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

float compute_rmse_bf16_f32(const nv_bfloat16* a, const float* b, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)__bfloat162float(a[i]) - (double)b[i];
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

int main() {
    std::cout << "=== Standalone Attention Test ===" << std::endl;
    
    // Test parameters
    int B = 2;    // batch size
    int T = 128;  // sequence length (smaller for testing)
    int Hq = 8;   // query heads
    int Hkv = 2;  // key/value heads (GQA)
    int HS = 64;  // head dimension
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch: " << B << ", Seq: " << T << ", Hq: " << Hq << ", Hkv: " << Hkv << ", HeadDim: " << HS << std::endl;
    
    // Calculate sizes
    size_t qkv_size = B * T * (Hq + 2 * Hkv) * HS;
    size_t out_size = B * T * Hq * HS;
    size_t stats_size = B * Hq * T;
    
    std::cout << "Memory requirements:" << std::endl;
    std::cout << "  QKV: " << qkv_size * sizeof(nv_bfloat16) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Output: " << out_size * sizeof(nv_bfloat16) / 1024 / 1024 << " MB" << std::endl;
    
    // =========================
    // Test 1: Custom CUDA kernel (FP32)
    // =========================
    std::cout << "\n=== Testing Custom CUDA Kernel (FP32) ===" << std::endl;
    
    std::vector<float> h_qkv_f32(qkv_size);
    std::vector<float> h_out_cuda_f32(out_size);
    std::vector<float> h_stats_cuda(stats_size);
    
    fill_random_f32(h_qkv_f32.data(), qkv_size, 0.0f, 0.1f);
    
    float *d_qkv_f32, *d_out_cuda_f32, *d_stats_cuda;
    CUDA_CHECK(cudaMalloc(&d_qkv_f32, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_cuda_f32, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stats_cuda, stats_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_qkv_f32, h_qkv_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Time the custom kernel
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = attention_forward_cuda(d_out_cuda_f32, d_stats_cuda, d_qkv_f32, B, T, Hq, Hkv, HS, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (err == cudaSuccess) {
        std::cout << "✓ Custom CUDA kernel completed in " << duration.count() / 1000.0f << " ms" << std::endl;
        
        CUDA_CHECK(cudaMemcpy(h_out_cuda_f32.data(), d_out_cuda_f32, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_stats_cuda.data(), d_stats_cuda, stats_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << "Sample outputs (first 5):" << std::endl;
        for (int i = 0; i < std::min(5, (int)out_size); i++) {
            std::cout << "  out[" << i << "] = " << h_out_cuda_f32[i] << std::endl;
        }
    } else {
        std::cout << "✗ Custom CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // =========================
    // Test 2: cuDNN kernel (BF16)
    // =========================
    std::cout << "\n=== Testing cuDNN Kernel (BF16) ===" << std::endl;
    
    std::vector<nv_bfloat16> h_qkv_bf16(qkv_size);
    std::vector<nv_bfloat16> h_out_cudnn(out_size);
    std::vector<float> h_stats_cudnn(stats_size);
    
    // Convert FP32 input to BF16 for cuDNN
    for (size_t i = 0; i < qkv_size; i++) {
        h_qkv_bf16[i] = __float2bfloat16(h_qkv_f32[i]);
    }
    
    nv_bfloat16 *d_qkv_bf16, *d_out_cudnn;
    float *d_stats_cudnn;
    std::byte *d_workspace;
    
    CUDA_CHECK(cudaMalloc(&d_qkv_bf16, qkv_size * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out_cudnn, out_size * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_stats_cudnn, stats_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_qkv_bf16, h_qkv_bf16.data(), qkv_size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    
    // Create cuDNN handle and get workspace
    cudnnHandle_t cudnn_handle = create_cudnn_handle();
    size_t workspace_size = cudnn_get_workspace_size(B, T, Hq, Hkv, HS, cudnn_handle);
    std::cout << "Workspace size: " << workspace_size / 1024 / 1024 << " MB" << std::endl;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    // Time cuDNN (first call - includes graph build)
    start = std::chrono::high_resolution_clock::now();
    
    attention_forward_cudnn(d_out_cudnn, d_stats_cudnn, d_qkv_bf16, d_workspace, cudnn_handle, 
                           B, T, Hq, Hkv, HS, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✓ cuDNN kernel (first call) completed in " << duration.count() / 1000.0f << " ms" << std::endl;
    
    // Time cuDNN (second call - cached graph)
    start = std::chrono::high_resolution_clock::now();
    
    attention_forward_cudnn(d_out_cudnn, d_stats_cudnn, d_qkv_bf16, d_workspace, cudnn_handle, 
                           B, T, Hq, Hkv, HS, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✓ cuDNN kernel (cached) completed in " << duration.count() / 1000.0f << " ms" << std::endl;
    
    CUDA_CHECK(cudaMemcpy(h_out_cudnn.data(), d_out_cudnn, out_size * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stats_cudnn.data(), d_stats_cudnn, stats_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Sample outputs (first 5):" << std::endl;
    for (int i = 0; i < std::min(5, (int)out_size); i++) {
        std::cout << "  out[" << i << "] = " << __bfloat162float(h_out_cudnn[i]) << std::endl;
    }
    
    // =========================
    // Compare results
    // =========================
    std::cout << "\n=== Comparing Results ===" << std::endl;
    
    float rmse = compute_rmse_bf16_f32(h_out_cudnn.data(), h_out_cuda_f32.data(), out_size);
    std::cout << "RMSE between cuDNN(BF16) and CUDA(FP32): " << rmse << std::endl;
    
    if (rmse < 0.01f) {
        std::cout << "✓ Results match within expected precision!" << std::endl;
    } else {
        std::cout << "⚠ Results differ more than expected" << std::endl;
    }
    
    // =========================
    // Cleanup
    // =========================
    CUDA_CHECK(cudaFree(d_qkv_f32));
    CUDA_CHECK(cudaFree(d_out_cuda_f32));
    CUDA_CHECK(cudaFree(d_stats_cuda));
    CUDA_CHECK(cudaFree(d_qkv_bf16));
    CUDA_CHECK(cudaFree(d_out_cudnn));
    CUDA_CHECK(cudaFree(d_stats_cudnn));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    std::cout << "\n=== Test completed successfully! ===" << std::endl;
    return 0;
}