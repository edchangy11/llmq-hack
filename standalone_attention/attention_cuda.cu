// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// This is a relatively simple baseline implementation of memory-efficient attention.
// Its main purpose is to allow running in *32-bit* precision, which is not supported
// by cuDNN.

#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

// Simplified vector operations
template<typename T, int N>
struct GenericVector {
    static constexpr int size = N;
    T data[N];
    
    __device__ T& operator[](int i) { return data[i]; }
    __device__ const T& operator[](int i) const { return data[i]; }
    
    __device__ static GenericVector zeros() {
        GenericVector v;
        for (int i = 0; i < N; i++) v.data[i] = T(0);
        return v;
    }
    
    __device__ static GenericVector load(const T* ptr) {
        GenericVector v;
        for (int i = 0; i < N; i++) v.data[i] = ptr[i];
        return v;
    }
    
    __device__ void store(T* ptr) const {
        for (int i = 0; i < N; i++) ptr[i] = data[i];
    }
};

template<int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_forward_gpu_kernel(
    scalar_t* out, float* stats, float scale,
    const scalar_t* qkv,
    int B, int T, int Hq, int Hkv) {

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto sub_warp = cg::tiled_partition<16>(block);

    extern __shared__ float scratch[];

    int h = blockIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;

    int hkv = h * Hkv / Hq;
    int TH = Hq + 2*Hkv;
    ptrdiff_t batch_offset = b * T * TH * E;
    qkv += batch_offset;
    const scalar_t* query = qkv + t * TH * E + h * E;
    const scalar_t* keys = qkv + (Hq + hkv) * E;
    const scalar_t* values = qkv + (Hq + Hkv + hkv) * E;

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using q_cache_t = GenericVector<float, E / sub_warp.size()>;
    q_cache_t q_cache;

    // combine values
    using v_cache_t = GenericVector<float, E / sub_warp.size()>;
    v_cache_t v_cache = v_cache_t::zeros();

    // determine maximum and online logsumexp
    float maximum = std::numeric_limits<float>::lowest();
    float lse = 0;

    for (int ee = 0; ee < E / (sub_warp.size() * vec_t::size); ++ee) {
        int e = (ee * sub_warp.size() + sub_warp.thread_rank()) * vec_t::size;
        vec_t qv = vec_t::load(query + e);
        for (int j = 0; j < vec_t::size; ++j) {
            q_cache[ee * vec_t::size + j] = (float)qv[j];
        }
    }

    for (int l = sub_warp.meta_group_rank(); l <= t; l += sub_warp.meta_group_size()) {
        ptrdiff_t kv_offset = l * TH * E;
        float qk = 0;
        for (int ee = 0; ee < E / (sub_warp.size() * vec_t::size); ++ee) {
            int e = (ee * sub_warp.size() + sub_warp.thread_rank()) * vec_t::size;
            vec_t kv = vec_t::load(keys + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                qk += q_cache[ee * vec_t::size + j] * (float)kv[j];
            }
        }
        qk = cg::reduce(sub_warp, qk, cg::plus<float>{});
        if (qk > maximum) {
            float rescale = std::exp(scale * (maximum - qk));
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[j] *= rescale;
            }
            lse *= rescale;
            maximum = qk;
        }
        float att = std::exp(scale * (qk - maximum));
        lse += std::exp(scale * (qk - maximum));

        for (int ee = 0; ee < E / (sub_warp.size() * vec_t::size); ++ee) {
            int e = (ee * sub_warp.size() + sub_warp.thread_rank()) * vec_t::size;
            vec_t vv = vec_t::load(values + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                v_cache[ee * vec_t::size + j] += att * (float)vv[j];
            }
        }
    }

    // combine split-k results
    if (sub_warp.thread_rank() == 0) {
        scratch[sub_warp.meta_group_rank()] = maximum;
        scratch[sub_warp.meta_group_rank() + sub_warp.meta_group_size()] = lse;
    }

    __syncthreads();
    float r_max = maximum;
    float l_max = maximum;
    float r_lse = 0;
    if (warp.thread_rank() < sub_warp.meta_group_size()) {
        r_max = scratch[warp.thread_rank()];
        r_lse = scratch[warp.thread_rank() + sub_warp.meta_group_size()];
    }

    maximum = cg::reduce(warp, r_max, cg::greater<float>{});
    r_lse *= std::exp(scale * (r_max - maximum));
    lse = cg::reduce(warp, r_lse, cg::plus<float>{});
    float rescale = std::exp(scale * (l_max - maximum)) / lse;
    for (int j = 0; j < v_cache_t::size; ++j) {
        v_cache[j] *= rescale;
    }
    if(threadIdx.x == 0) {
        stats[b * Hq * T + h * T + t] = scale * maximum + std::log(lse);
    }
    __syncthreads();

    for (int ee = 0; ee < E / (sub_warp.size() * vec_t::size); ++ee) {
        int e = (ee * sub_warp.size() + sub_warp.thread_rank()) * vec_t::size;
        fvec_t store;
        for (int j = 0; j < vec_t::size; ++j) {
            store[j] = v_cache[ee * vec_t::size + j];
        }
        store.store(scratch + e + E * sub_warp.meta_group_rank());
    }

    if (warp.meta_group_rank() != 0) return;
    __syncthreads();
    // write result
    for (int e = vec_t::size * warp.thread_rank(); e < E; e += vec_t::size * warp.size()) {
        fvec_t res = fvec_t::zeros();
        for (int j = 0; j < sub_warp.meta_group_size(); ++j) {
            fvec_t sv = fvec_t::load(scratch + e + E * j);
            for (int jj = 0; jj < vec_t::size; ++jj) {
                res[jj] += sv[jj];
            }
        }
        vec_t cv;
        for (int j = 0; j < vec_t::size; ++j) {
            cv[j] = (scalar_t)res[j];
        }
        cv.store(out + ((b * Hq + h) * T + t) * E + e);
    }
}

template<class floatX>
cudaError_t attention_gpu_forward(floatX* out, float* stats, float scale,
                          const floatX* qkv,
                          int B, int T, int Hq, int Hkv, int Hs, cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;

    if (Hs == 128) {
        attention_forward_gpu_kernel<128><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv, B, T, Hq, Hkv);
    } else if (Hs == 64) {
        attention_forward_gpu_kernel<64><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv,  B, T, Hq, Hkv);
    } else {
        printf("Unsupported head dimension: %d\n", Hs);
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// C-style interface
extern "C" {
    cudaError_t attention_forward_cuda(float* out, float* stats,
                                      const float* qkv, int B, int T, int Hq, int Hkv, int Hs,
                                      cudaStream_t stream) {
        float scale = 1.0f / sqrtf(Hs);
        return attention_gpu_forward(out, stats, scale, qkv, B, T, Hq, Hkv, Hs, stream);
    }

    cudaError_t attention_forward_cuda_bf16(nv_bfloat16* out, float* stats,
                                           const nv_bfloat16* qkv, int B, int T, int Hq, int Hkv, int Hs,
                                           cudaStream_t stream) {
        float scale = 1.0f / sqrtf(Hs);
        return attention_gpu_forward(out, stats, scale, qkv, B, T, Hq, Hkv, Hs, stream);
    }
}