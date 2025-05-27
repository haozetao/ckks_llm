#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda/barrier>
#include <cuda_pipeline_primitives.h>

#include "uint128.cuh"
#include "Context_23.h"

#define NTT_MAX_THREADS 1024
#define NTT_MIN_BLOCKS 1

using namespace std;

// __device__ __forceinline__ void CT_BFU(uint64_tt& X, uint64_tt& Y, uint64_tt& q, uint128_tt& mu)
// {
// //X = X + Y*psi
// //Y = X - Y*psi
// }

// __device__ __forceinline__ void GS_BFU(uint64_tt& X, uint64_tt& Y, uint64_tt& q, uint128_tt& mu)
// {
// //X = X + Y*psi
// //Y = X - Y*psi
// }

// __global__ void barrett_new(uint64_tt a[], const uint64_tt b[])
// {
//     register int i = blockIdx.x * 256 + threadIdx.x;

//     uint128_tt mu(pqt_mu_cons_high[0], pqt_mu_cons_low[0]);
//     uint64_tt q = pqt_cons[0];
//     register uint64_tt ra = a[i];
//     register uint64_tt rb = b[i];

//     uint128_tt rc;

//     mul64(ra, rb, rc);
//     singleBarrett_new(rc, q, mu);

//     a[i]=rc.low;
// }

// template<uint32_tt l, uint32_tt n, uint32_tt iter_count>
// __global__ void CTBasedNTTInnerSingle_batch(uint64_tt a[], uint64_tt psi_powers[], uint32_tt division, int idx_mod)
// {
//     uint32_tt index = blockIdx.y % division;
//     uint64_tt q = pqt_cons[index + idx_mod];
//     uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

//     register int local_tid = threadIdx.x;

//     extern __shared__ uint64_tt shared_array[];

// #pragma unroll
//     for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//     {
//         register int global_tid = local_tid + iteration_num * 1024;
//         shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];//+-
//     }
//     __syncthreads();

//     register int step = (n>>1)/l;
//     register int mask_len = iter_count;
//     for (int length = l; length < n; length *= 2)
//     {

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             register int psi_step = (global_tid >> mask_len);
//             register int target_index = psi_step * step * 2 + (global_tid & ((1<<mask_len) - 1)) ;

//             psi_step = (global_tid + blockIdx.x * (n / l / 2)) >> mask_len;

//             register uint64_tt psi = psi_powers[length + psi_step + index * n];
//             register uint64_tt first_target_value = shared_array[target_index];
//             register uint128_tt temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

//             mul64(temp_storage.low, psi, temp_storage);

// #if singleBarrett_qq
//     // uint64_tt q2 = pq2_cons[index + idx_mod];
//     // singleBarrett_new(temp_storage, q, q2, mu);
//     singleBarrett_new(temp_storage, q, q<<1, mu);
// #else
//     singleBarrett_new(temp_storage, q, mu);
// #endif // singleBarrett_qq

//             register uint64_tt second_target_value = temp_storage.low;

//             register uint64_tt target_result = first_target_value + second_target_value;

//             target_result -= q * (target_result >= q);

//             shared_array[target_index] = target_result;

//             first_target_value += q * (first_target_value < second_target_value);

//             shared_array[target_index + step] = first_target_value - second_target_value;
//         }
//         step >>= 1;
//         mask_len -= 1;
//         __syncthreads();
//     }

// #pragma unroll
//     for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//     {
//         register int global_tid = local_tid + iteration_num * 1024;
//         a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
//     }
// }

// template<uint32_tt l, uint32_tt n, uint32_tt iter_count>
// __global__ void GSBasedINTTInnerSingle_batch(uint64_tt a[], uint64_tt psiinv_powers[], uint32_tt division, int idx_mod)
// {
//     uint32_tt index = blockIdx.y % division;
//     uint64_tt q = pqt_cons[index + idx_mod];
//     uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

//     register int local_tid = threadIdx.x;

//     extern __shared__ uint64_tt shared_array[];

//     register uint64_tt q2 = (q + 1) >> 1;

// #pragma unroll
//     for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//     {
//         register int global_tid = local_tid + iteration_num * 1024;
//         shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];//+-
//     }
//     __syncthreads();

//     register int step = 1;
//     register int mask_len = 0;
//     for (int length = (n / 2); length >= l; length /= 2)
//     {

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             register int psi_step = (global_tid >> mask_len);
//             register int target_index = psi_step * step * 2 + (global_tid & ((1<<mask_len) - 1));
//             psi_step = (global_tid + blockIdx.x * (n / l / 2)) >> mask_len;

//             register uint64_tt psiinv = psiinv_powers[length + psi_step + index * n];

//             register uint64_tt first_target_value = shared_array[target_index];
//             register uint64_tt second_target_value = shared_array[target_index + step];

//             register uint64_tt target_result = first_target_value + second_target_value;

//             target_result -= q * (target_result >= q);

//             shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

//             first_target_value += q * (first_target_value < second_target_value);

//             register uint128_tt temp_storage = first_target_value - second_target_value;

//             mul64(temp_storage.low, psiinv, temp_storage);


// #if singleBarrett_qq
//     // uint64_tt qq = pq2_cons[index + idx_mod];
//     // singleBarrett_new(temp_storage, q, qq, mu);
//     singleBarrett_new(temp_storage, q, q<<1, mu);
// #else
//     singleBarrett_new(temp_storage, q, mu);
// #endif // singleBarrett_qq

//             register uint64_tt temp_storage_low = temp_storage.low;

//             shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
//         }
//         step *= 2;
//         mask_len += 1;
//         __syncthreads();
//     }

// #pragma unroll
//     for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//     {
//         register int global_tid = local_tid + iteration_num * 1024;
//         a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
//     }
// }

// template<uint32_tt l, uint32_tt n, uint32_tt mask_len>
// __global__ void CTBasedNTTInner_batch(uint64_tt a[], uint64_tt psi_powers[], uint32_tt division, int idx_mod)
// {
//     uint32_tt index = blockIdx.y % division;
//     uint64_tt q = pqt_cons[index + idx_mod];
//     uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);
//     int length = l;
//     register int global_tid = blockIdx.x * 1024 + threadIdx.x;
//     register int step = (n >> 1) / l;
//     register int psi_step = (global_tid >> mask_len);
//     register int target_index = psi_step * step * 2 + (global_tid & ((1<<mask_len) - 1)) + blockIdx.y * n;//+-

//     register uint64_tt psi = psi_powers[length + psi_step + index * n];

//     register uint64_tt first_target_value = a[target_index];

//     register uint128_tt temp_storage = a[target_index + step];

//     mul64(temp_storage.low, psi, temp_storage);

// #if singleBarrett_qq
//     // uint64_tt q2 = pq2_cons[index + idx_mod];
//     // singleBarrett_new(temp_storage, q, q2, mu);
//     singleBarrett_new(temp_storage, q, q<<1, mu);
// #else
//     singleBarrett_new(temp_storage, q, mu);
// #endif // singleBarrett_qq

//     register uint64_tt second_target_value = temp_storage.low;

//     register uint64_tt target_result = first_target_value + second_target_value;

//     target_result -= q * (target_result >= q);

//     a[target_index] = target_result;

//     first_target_value += q * (first_target_value < second_target_value);

//     a[target_index + step] = first_target_value - second_target_value; 
// }

// template<uint32_tt l, uint32_tt n, uint32_tt mask_len>
// __global__ void GSBasedINTTInner_batch(uint64_tt a[], uint64_tt psiinv_powers[], uint32_tt division, int idx_mod)
// {
//     uint32_tt index = blockIdx.y % division;
//     uint64_tt q = pqt_cons[index + idx_mod];
//     uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

//     int length = l;

//     register int global_tid = blockIdx.x * 1024 + threadIdx.x;

//     register int step = (n >> 1) / l;
//     register int psi_step = (global_tid >> mask_len);
//     register int target_index = psi_step * step * 2 + (global_tid & ((1<<mask_len) - 1)) + blockIdx.y * n;//+-

//     register uint64_tt psiinv = psiinv_powers[length + psi_step + index * n];

//     register uint64_tt first_target_value = a[target_index];
//     register uint64_tt second_target_value = a[target_index + step];

//     register uint64_tt target_result = first_target_value + second_target_value;

//     target_result -= q * (target_result >= q);

//     register uint64_tt q2 = (q + 1) >> 1;

//     target_result = (target_result >> 1) + q2 * (target_result & 1);

//     a[target_index] = target_result;

//     first_target_value += q * (first_target_value < second_target_value);

//     register uint128_tt temp_storage = first_target_value - second_target_value;

//     mul64(temp_storage.low, psiinv, temp_storage);

// #if singleBarrett_qq
//     // uint64_tt qq = pq2_cons[index + idx_mod];
//     // singleBarrett_new(temp_storage, q, qq, mu);
//     singleBarrett_new(temp_storage, q, q<<1, mu);
// #else
//     singleBarrett_new(temp_storage, q, mu);
// #endif // singleBarrett_qq

//     register uint64_tt temp_storage_low = temp_storage.low;

//     temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

//     a[target_index + step] = temp_storage_low;
// }
/////////////////////////////////////////new_batch_NTT///////////////////////////////////////////////////////

/** Computer one butterfly in forward NTT
     * x[0] = x[0] + pow * x[1] % mod
     * x[1] = x[0] - pow * x[1] % mod
     * [0, 4*mod)
     */
__device__ __inline__ void CTBased_butterfly(uint64_tt &a, uint64_tt &b,
                                          const uint64_tt &w, const uint64_tt &w_shoup,
                                          uint64_tt mod, uint64_tt mod2)
{
    const uint64_tt hi = __umul64hi(b, w_shoup);
    const uint64_tt tw_b = b * w - hi * mod;
    const uint64_tt s = a - mod2;
    a = s + (s >> 63) * mod2;
    b = a + mod2 - tw_b;
    a += tw_b;
}

/** Computer one butterfly in inverse NTT
     * x[0] = (x[0] + pow * x[1]) / 2 % mod
     * x[1] = (x[0] - pow * x[1]) / 2 % mod
     * [0, 2*mod)
     */
__device__ __inline__ void GSBased_butterfly(uint64_tt &a, uint64_tt &b,
                                          const uint64_tt &w, const uint64_tt &w_shoup, uint64_tt mod, uint64_tt mod2)
{
    const uint64_tt t = a + mod2 - b;
    const uint64_tt s = a + b - mod2;
    const uint64_tt hi = __umul64hi(t, w_shoup);
    a = s + (s >> 63) * mod2;
	b = t * w - hi * mod;
}

__device__ __forceinline__ void ntt8(uint64_tt* a,
                                          const uint64_tt* w,
                                          const uint64_tt* w_shoup,
                                          uint64_tt w_idx,
                                          uint64_tt mod,
                                          uint64_tt mod2)
{
    // stage 1
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        CTBased_butterfly(a[j], a[j + 4], w[w_idx], w_shoup[w_idx], mod, mod2);
    }        
    // stage 2
#pragma unroll
    for(int j = 0; j < 2; j++)
    {
        CTBased_butterfly(a[j * 4], a[j * 4 + 2], w[2 * w_idx + j], w_shoup[2 * w_idx + j], mod, mod2);
        CTBased_butterfly(a[j * 4 + 1], a[j * 4 + 3], w[2 * w_idx + j], w_shoup[2 * w_idx + j], mod, mod2);
    } 
    // stage 3
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        CTBased_butterfly(a[j * 2], a[j * 2 + 1], w[4 * w_idx + j], w_shoup[4 * w_idx + j], mod, mod2);
    }
}

__device__ __forceinline__ void ntt4(uint64_tt* a,
                                          const uint64_tt* w,
                                          const uint64_tt* w_shoup,
                                          uint64_tt w_idx,
                                          uint64_tt mod,
                                          uint64_tt mod2)
{
    // stage 1
#pragma unroll
    for(int j = 0; j < 2;j++)
    {
        CTBased_butterfly(a[j], a[j + 2], w[w_idx], w_shoup[w_idx], mod, mod2);
    }
    // stage 2
#pragma unroll
    for(int j = 0; j < 2;j++)
    {
        CTBased_butterfly(a[j * 2], a[j * 2 + 1], w[2 * w_idx + j], w_shoup[2 * w_idx + j], mod, mod2);
    }
}

__device__ __forceinline__ void intt8(uint64_tt* a,
                                        const uint64_tt* winv,
                                        const uint64_tt* winv_shoup,
                                        uint64_tt w_idx,
                                        uint64_tt mod,
                                        uint64_tt mod2)
{
    // stage 1
#pragma unroll
    for(int j = 0; j < 4;j++)
    {
        GSBased_butterfly(a[j * 2], a[j * 2 + 1], winv[4 * w_idx + j], winv_shoup[4 * w_idx + j], mod, mod2);
    }
    // stage 2
#pragma unroll
    for(int j = 0; j < 2;j++)
    {
        GSBased_butterfly(a[j * 4], a[j * 4 + 2], winv[2 * w_idx + j], winv_shoup[2 * w_idx + j], mod, mod2);
        GSBased_butterfly(a[j * 4 + 1], a[j * 4 + 3], winv[2 * w_idx + j], winv_shoup[2 * w_idx + j], mod, mod2);
    } 
    // stage 3
#pragma unroll
    for(int j = 0; j < 4;j++)
    {
        GSBased_butterfly(a[j], a[j + 4], winv[w_idx], winv_shoup[w_idx], mod, mod2);
    }  
}

__device__ __forceinline__ void intt4(uint64_tt* a,
                                        const uint64_tt* winv,
                                        const uint64_tt* winv_shoup,
                                        uint64_tt w_idx,
                                        uint64_tt mod,
                                        uint64_tt mod2)
{
    // stage 1
#pragma unroll
    for(int j = 0; j < 2;j++)
    {
        GSBased_butterfly(a[j * 4], a[j * 4 + 2], winv[2 * w_idx + j], winv_shoup[2 * w_idx + j], mod, mod2);
    }
    // stage 2
#pragma unroll
    for(int j = 0; j < 2;j++)
    {
        GSBased_butterfly(a[j * 2], a[j * 2 + 4], winv[w_idx], winv_shoup[w_idx], mod, mod2);
    }
}

// #include <cooperative_groups.h>
// #include <cuda/barrier>
//ntt_kernel
__global__ static void NTT8pointPerThread_kernel1(uint64_tt *inout,
                           const uint64_tt *twiddles,
                           const uint64_tt *twiddles_shoup,
                           int poly_num,
                           int start_poly_idx,
                           int mod_num,
                           int start_mod_idx,
                           int n,
                           int n1,
                           int pad,
                           int poly_mod_len) {
    extern __shared__ uint64_tt buffer[];
    // uint64_tt* psi_smem = buffer + (n1 + pad + 1) * pad;
    // uint64_tt* psi_shoup_smem = psi_smem + 8*8;
    // pad address
    int pad_tid = threadIdx.x % pad;
    int pad_idx = threadIdx.x / pad;

    int group = n1 / 8;
    // size of a block
    uint64_tt samples[8];
    // uint64_tt psi_samples[8];
    // uint64_tt psi_shoup_samples[8];
    
    int t = n / 2;
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x ;
         tid < (n / 8 * mod_num); tid += blockDim.x * gridDim.x) {
        // modulus idx
        int twr_idx = tid / (n / 8) + start_mod_idx;
        // index in n/8 range (in each tower)m
        int n_idx = tid % (n / 8);
        int poly_idx = tid / (n / 8);
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx)* n;
        const uint64_tt *psi = twiddles + twr_idx * n;
        const uint64_tt *psi_shoup = twiddles_shoup + twr_idx * n;
        uint64_tt modulus = pqt_cons[twr_idx];
        uint64_tt modulus2 = modulus << 1;
        int n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));


        // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        // auto block = cooperative_groups::this_thread_block();      
        // if (block.thread_rank() == 0) {
        //     init(&barrier, block.size()); // Friend function initializes barrier
        // }
        // block.sync();
        int tw_idx = 1;

        // cuda::memcpy_async(block, psi_smem+1, psi + tw_idx, sizeof(uint64_tt) * 1, barrier);
        // cuda::memcpy_async(block, psi_smem+2, psi + tw_idx*2, sizeof(uint64_tt) * 2, barrier);
        // cuda::memcpy_async(block, psi_smem+4, psi + tw_idx*4, sizeof(uint64_tt) * 4, barrier);

        // cuda::memcpy_async(block, psi_shoup_smem+1, psi_shoup + tw_idx, sizeof(uint64_tt) * 1, barrier);
        // cuda::memcpy_async(block, psi_shoup_smem+2, psi_shoup + tw_idx*2, sizeof(uint64_tt) * 2, barrier);
        // cuda::memcpy_async(block, psi_shoup_smem+4, psi_shoup + tw_idx*4, sizeof(uint64_tt) * 4, barrier);

        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }

        // barrier.arrive_and_wait(); // Waits for all copies to complete
        // ntt8(samples, psi_smem, psi_shoup_smem, 1, modulus, modulus2);
        // block.sync();

        ntt8(samples, psi, psi_shoup, tw_idx, modulus, modulus2);

        for (int j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + pad_idx + group * j] = samples[j];
        }
        int remain_iters = 0;
        __syncthreads();
        for (int j = 8, k = group / 2; j < group + 1; j *= 8, k >>= 3) {
            int m_idx2 = pad_idx / (k / 4);
            int t_idx2 = pad_idx % (k / 4);
            int tw_idx2 = j * tw_idx + m_idx2;

            // cuda::memcpy_async(block, psi_smem+tw_idx2+1, psi + tw_idx2, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_smem+tw_idx2+2, psi + tw_idx2*2, sizeof(uint64_tt) * 2, barrier);
            // cuda::memcpy_async(block, psi_smem+tw_idx2+4, psi + tw_idx2*4, sizeof(uint64_tt) * 4, barrier);

            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2+1, psi_shoup + tw_idx2, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2+2, psi_shoup + tw_idx2*2, sizeof(uint64_tt) * 2, barrier);
            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2*+4, psi_shoup + tw_idx2*4, sizeof(uint64_tt) * 4, barrier);

            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
    
            // barrier.arrive_and_wait(); // Waits for all copies to complete
            // ntt8(samples, psi_smem, psi_shoup_smem, 1, modulus, modulus2);
            // block.sync();
            ntt8(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);

            for (int l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == group / 2)
                remain_iters = 1;
            if (j == group / 4)
                remain_iters = 2;
            __syncthreads();
        }

        if (group < 8)
            remain_iters = (group == 4) ? 2 : 1;
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l];
        }
        if (remain_iters == 1) {
            int tw_idx2 = 4 * group * tw_idx + 4 * pad_idx;
            for( int i = 0 ; i < 4; i++ ){

                // cuda::memcpy_async(block, psi_smem+1, psi + tw_idx2 + i, sizeof(uint64_tt) * 1, barrier);
                // cuda::memcpy_async(block, psi_shoup_smem+1, psi_shoup + tw_idx2 + i, sizeof(uint64_tt) * 1, barrier);
            
                // barrier.arrive_and_wait(); // Waits for all copies to complete
                // CTBased_butterfly(samples[2 * i], samples[2 * i + 1], psi_smem[1], psi_shoup_smem[1], modulus, modulus2);
                // block.sync();

                CTBased_butterfly(samples[2 * i], samples[2 * i + 1], psi[tw_idx2 + i], psi_shoup[tw_idx2 + i], modulus, modulus2);
            }
        }
        else if (remain_iters == 2) {
            int tw_idx2 = 2 * group * tw_idx + 2 * pad_idx;

            // cuda::memcpy_async(block, psi_smem+tw_idx2/8+1, psi + tw_idx2, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_smem+tw_idx2/8+2, psi + tw_idx2*2, sizeof(uint64_tt) * 2, barrier);

            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2/8+1, psi_shoup + tw_idx2, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2/8+2, psi_shoup + tw_idx2*2, sizeof(uint64_tt) * 2, barrier);
        
            // barrier.arrive_and_wait(); // Waits for all copies to complete
            // ntt4(samples, psi_smem, psi_shoup_smem, 1, modulus, modulus2);
            // block.sync();

            
            // cuda::memcpy_async(block, psi_smem+tw_idx2/8+1, psi + tw_idx2+1, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_smem+tw_idx2/8+2, psi + (tw_idx2+1)*2, sizeof(uint64_tt) * 2, barrier);

            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2/8+1, psi_shoup + tw_idx2+1, sizeof(uint64_tt) * 1, barrier);
            // cuda::memcpy_async(block, psi_shoup_smem+tw_idx2/8+2, psi_shoup + (tw_idx2+1)*2, sizeof(uint64_tt) * 2, barrier);
        
            // barrier.arrive_and_wait(); // Waits for all copies to complete
            // ntt4(samples + 4, psi_smem, psi_shoup_smem, 1, modulus, modulus2);
            // block.sync();

            ntt4(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            ntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus, modulus2);
        }
        for (int l = 0; l < 8; l++) {
            buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l] = samples[l];
        }

        __syncthreads();
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = buffer[pad_tid * (n1 + pad) + pad_idx + group * j];
        }
    }
}

__global__ static void
NTT8pointPerThread_kernel2(uint64_tt *inout,
                           const uint64_tt *twiddles,
                           const uint64_tt *twiddles_shoup,
                           int poly_num,
                           int start_poly_idx,
                           int mod_num,
                           int start_mod_idx,
                           int n,
                           int n1,
                           int n2,
                           int poly_mod_len) {
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n / 8 * mod_num); tid += blockDim.x * gridDim.x) {
        int group = n2 / 8;
        int set = threadIdx.x / group;
        // size of a block
        uint64_tt samples[8];
        int t = n2 / 2;
        int twr_idx = mod_num - 1 - (tid / (n / 8)) + start_mod_idx;
        // index in n/2 range
        int n_idx = tid % (n / 8);
        // tid'th block
        int m_idx = n_idx / (t / 4);
        int t_idx = n_idx % (t / 4);

        int poly_idx = mod_num - 1 - (tid / (n / 8));
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx) * n;
        uint64_tt modulus = pqt_cons[twr_idx];
        uint64_tt modulus2 = modulus << 1;
        const uint64_tt *psi = twiddles + n * twr_idx;
        const uint64_tt *psi_shoup = twiddles_shoup + n * twr_idx;
        int n_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }
        int tw_idx = n1 + m_idx;
        ntt8(samples, psi, psi_shoup, tw_idx, modulus, modulus2);
        for (int j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = samples[j];
        }
        int tail = 0;
        __syncthreads();

        for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
            for (int l = 0; l < 8; l++) {
                samples[l] =
                        buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            ntt8(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            for (int l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == t / 8)
                tail = 1;
            if (j == t / 16)
                tail = 2;
            __syncthreads();
        }

        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        if (tail == 1) {
            int tw_idx2 = t * tw_idx + 4 * t_idx;
            for( int i = 0 ; i < 4; i++ ){
                CTBased_butterfly(samples[2 * i], samples[2 * i + 1], psi[tw_idx2 + i], psi_shoup[tw_idx2 + i], modulus, modulus2);
            }
        }
        else if (tail == 2) {
            int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
            ntt4(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            ntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus, modulus2);
        }
        for (int l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        __syncthreads();

        // final reduction
        for (int j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
            csub_q(samples[j], modulus2);
            csub_q(samples[j], modulus);
        }
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}
//intt_kernel
__global__ static void
INTT8pointPerThread_kernel1(uint64_tt *inout,
                           const uint64_tt *itwiddles,
                           const uint64_tt *itwiddles_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int n2,
                           const int poly_mod_len)
{
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n / 8 * mod_num); i += blockDim.x * gridDim.x) {
        int group = n2 / 8;
        int set = threadIdx.x / group;
        // size of a block
        uint64_tt samples[8];
        int t = n / 2 / n1;
        // prime idx
        register int global_tid;
        if(blockIdx.y * gridDim.x == 0) 
            global_tid = i;
        else 
            global_tid = i % (blockIdx.y * gridDim.x * blockDim.x);
        int twr_idx = global_tid / (n / 8) + start_mod_idx;
        // index in N/2 range
        int n_idx = global_tid % (n / 8);

        int poly_idx = (global_tid / (n / 8));
        // i'th block
        int m_idx = n_idx / (t / 4);
        int t_idx = n_idx % (t / 4);
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx)* n;
        const uint64_tt *psi = itwiddles + n * twr_idx;
        const uint64_tt *psi_shoup = itwiddles_shoup + n * twr_idx;
        uint64_tt modulus_value = pqt_cons[twr_idx];
        // uint64_tt modulus_value2 = pqt2_cons[twr_idx];
        uint64_tt modulus_value2 = modulus_value << 1;
        int n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (int j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(data_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        int tw_idx = n1 + m_idx;
        int tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        int tail = 0;
        __syncthreads();

#pragma unroll
        for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
        }
        if (tail == 1) {
            for( int i = 0 ; i < 4; i++ ){
                GSBased_butterfly(samples[i], samples[i + 4], psi[tw_idx], psi_shoup[tw_idx], modulus_value, modulus_value2);
            }
        }
        else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
INTT8pointPerThread_kernel2(uint64_tt *inout,
                           const uint64_tt *itwiddles,
                           const uint64_tt *itwiddles_shoup,
                           const uint64_tt *inv_degree_modulo,
                           const uint64_tt *inv_degree_modulo_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int pad,
                           const int poly_mod_len)
{
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n / 8 * mod_num); i += blockDim.x * gridDim.x) {
        // pad address
        int pad_tid = threadIdx.x % pad;
        int pad_idx = threadIdx.x / pad;

        int group = n1 / 8;
        // size of a block
        uint64_tt samples[8];

        int t = n / 2;
        // prime idx
        register int global_tid;
        if(blockIdx.y * gridDim.x * blockDim.x == 0) 
            global_tid = i ;
        else 
            global_tid = i % (blockIdx.y * gridDim.x * blockDim.x);
            
        int twr_idx = global_tid / (n / 8) + start_mod_idx;
        // index in N/2 range
        int n_idx = global_tid % (n / 8);
        int poly_idx = global_tid / (n / 8);

        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx) * n;
        const uint64_tt *psi = itwiddles + n * twr_idx;
        const uint64_tt *psi_shoup = itwiddles_shoup + n * twr_idx;
        uint64_tt modulus_value = pqt_cons[twr_idx];
        // uint64_tt modulus_value2 = pqt2_cons[twr_idx];
        uint64_tt modulus_value2 = modulus_value << 1;
        uint64_tt inv_degree_mod = inv_degree_modulo[twr_idx];
        uint64_tt inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx];
        int n_init = 2 * t / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 / group * j);
        }
        int tw_idx = 1;
        int tw_idx2 = group * tw_idx + pad_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + 8 * pad_idx + j] = samples[j];
        }
        int tail = 0;
        __syncthreads();

#pragma unroll
        for (int j = group / 8, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = pad_idx / (k / 4);
            int t_idx2 = pad_idx % (k / 4);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                                    (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (group < 8)
            tail = (group == 4) ? 2 : 1;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[pad_tid * (n1 + pad) + pad_idx + group * l];
        }
        if (tail == 1) {
            for( int i = 0 ; i < 4; i++ ){
                GSBased_butterfly(samples[i], samples[i + 4], psi[tw_idx], psi_shoup[tw_idx], modulus_value, modulus_value2);
            }
        }
        else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
        }

        for (int j = 0; j < 4; j++) {
            const uint64_tt hi = __umul64hi(samples[j], inv_degree_mod_shoup);
            samples[j] = samples[j] * inv_degree_mod - hi *  modulus_value;
        }
        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
        for (int j = 0; j < 8; j++) {
            csub_q(samples[j], modulus_value);
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
INTT8pointPerThread_for_ext_kernel1(uint64_tt *inout,
                           const uint64_tt *itwiddles,
                           const uint64_tt *itwiddles_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int n2,
                           const int poly_mod_len,
                           const int cipher_mod_num)
{
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + (blockIdx.z * cipher_mod_num * n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n / 8 * mod_num); i += blockDim.x * gridDim.x) {
        int group = n2 / 8;
        int set = threadIdx.x / group;
        // size of a block
        uint64_tt samples[8];
        int t = n / 2 / n1;
        // prime idx
        register int global_tid;
        if(blockIdx.y * gridDim.x == 0) 
            global_tid = i;
        else 
            global_tid = i % (blockIdx.y * gridDim.x * blockDim.x);
        int twr_idx = global_tid / (n / 8) + start_mod_idx;
        // index in N/2 range
        int n_idx = global_tid % (n / 8);

        int poly_idx = (global_tid / (n / 8)) + start_poly_idx;
        // i'th block
        int m_idx = n_idx / (t / 4);
        int t_idx = n_idx % (t / 4);
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx)* n;
        const uint64_tt *psi = itwiddles + n * twr_idx;
        const uint64_tt *psi_shoup = itwiddles_shoup + n * twr_idx;
        uint64_tt modulus_value = pqt_cons[twr_idx];
        // uint64_tt modulus_value2 = pqt2_cons[twr_idx];
        uint64_tt modulus_value2 = modulus_value << 1;
        int n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (int j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(data_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        int tw_idx = n1 + m_idx;
        int tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        int tail = 0;
        __syncthreads();

#pragma unroll
        for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
        }
        if (tail == 1) {
            for( int i = 0 ; i < 4; i++ ){
                GSBased_butterfly(samples[i], samples[i + 4], psi[tw_idx], psi_shoup[tw_idx], modulus_value, modulus_value2);
            }
        }
        else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
INTT8pointPerThread_for_ext_kernel2(uint64_tt *inout,
                           const uint64_tt *itwiddles,
                           const uint64_tt *itwiddles_shoup,
                           const uint64_tt *inv_degree_modulo,
                           const uint64_tt *inv_degree_modulo_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int pad,
                           const int poly_mod_len,
                           const int cipher_mod_num)
{
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + (blockIdx.z * cipher_mod_num * n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n / 8 * mod_num); i += blockDim.x * gridDim.x) {
        // pad address
        int pad_tid = threadIdx.x % pad;
        int pad_idx = threadIdx.x / pad;

        int group = n1 / 8;
        // size of a block
        uint64_tt samples[8];

        int t = n / 2;
        // prime idx
        register int global_tid;
        if(blockIdx.y * gridDim.x * blockDim.x == 0) 
            global_tid = i ;
        else 
            global_tid = i % (blockIdx.y * gridDim.x * blockDim.x);
            
        int twr_idx = global_tid / (n / 8) + start_mod_idx;
        // index in N/2 range
        int n_idx = global_tid % (n / 8);
        int poly_idx = global_tid / (n / 8) + start_poly_idx;

        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx) * n;
        const uint64_tt *psi = itwiddles + n * twr_idx;
        const uint64_tt *psi_shoup = itwiddles_shoup + n * twr_idx;
        uint64_tt modulus_value = pqt_cons[twr_idx];
        // uint64_tt modulus_value2 = pqt2_cons[twr_idx];
        uint64_tt modulus_value2 = modulus_value << 1;
        uint64_tt inv_degree_mod = inv_degree_modulo[twr_idx];
        uint64_tt inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx];
        int n_init = 2 * t / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 / group * j);
        }
        int tw_idx = 1;
        int tw_idx2 = group * tw_idx + pad_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + 8 * pad_idx + j] = samples[j];
        }
        int tail = 0;
        __syncthreads();

#pragma unroll
        for (int j = group / 8, k = 32; j > 0; j >>= 3, k *= 8) {
            int m_idx2 = pad_idx / (k / 4);
            int t_idx2 = pad_idx % (k / 4);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                                    (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value, modulus_value2);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (group < 8)
            tail = (group == 4) ? 2 : 1;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[pad_tid * (n1 + pad) + pad_idx + group * l];
        }
        if (tail == 1) {
            for( int i = 0 ; i < 4; i++ ){
                GSBased_butterfly(samples[i], samples[i + 4], psi[tw_idx], psi_shoup[tw_idx], modulus_value, modulus_value2);
            }
        }
        else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value, modulus_value2);
        }

        for (int j = 0; j < 4; j++) {
            const uint64_tt hi = __umul64hi(samples[j], inv_degree_mod_shoup);
            samples[j] = samples[j] * inv_degree_mod - hi *  modulus_value;
        }
        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
        for (int j = 0; j < 8; j++) {
            csub_q(samples[j], modulus_value);
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

//ntt_kernel
__global__ static void 
NTT8pointPerThread_for_ext_kernel1(uint64_tt *inout,
                           const uint64_tt *twiddles,
                           const uint64_tt *twiddles_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int pad,
                           const int poly_mod_len,
                           const int cipher_mod_num) {
    extern __shared__ uint64_tt buffer[];
    // pad address
    int pad_tid = threadIdx.x % pad;
    int pad_idx = threadIdx.x / pad;

    int group = n1 / 8;
    // size of a block
    uint64_tt samples[8];
    int t = n / 2;
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n + (blockIdx.z * cipher_mod_num * n);
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x ;
         tid < (n / 8 * mod_num); tid += blockDim.x * gridDim.x) {
        // modulus idx
        int twr_idx = tid / (n / 8) + start_mod_idx;
        // index in n/8 range (in each tower)m
        int n_idx = tid % (n / 8);
        int poly_idx = tid / (n / 8);
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx)* n;
        const uint64_tt *psi = twiddles + twr_idx * n;
        const uint64_tt *psi_shoup = twiddles_shoup + twr_idx * n;
        uint64_tt modulus = pqt_cons[twr_idx];
        uint64_tt modulus2 = modulus << 1;
        int n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }
        int tw_idx = 1;
        ntt8(samples, psi, psi_shoup, tw_idx, modulus, modulus2);
        for (int j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + pad_idx + group * j] = samples[j];
        }
        int remain_iters = 0;
        __syncthreads();
        for (int j = 8, k = group / 2; j < group + 1; j *= 8, k >>= 3) {
            int m_idx2 = pad_idx / (k / 4);
            int t_idx2 = pad_idx % (k / 4);
            for (int l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            ntt8(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            for (int l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == group / 2)
                remain_iters = 1;
            if (j == group / 4)
                remain_iters = 2;
            __syncthreads();
        }

        if (group < 8)
            remain_iters = (group == 4) ? 2 : 1;
        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l];
        }
        if (remain_iters == 1) {
            int tw_idx2 = 4 * group * tw_idx + 4 * pad_idx;
            for( int i = 0 ; i < 4; i++ ){
                CTBased_butterfly(samples[2 * i], samples[2 * i + 1], psi[tw_idx2 + i], psi_shoup[tw_idx2 + i], modulus, modulus2);
            }
        }
        else if (remain_iters == 2) {
            int tw_idx2 = 2 * group * tw_idx + 2 * pad_idx;
            ntt4(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            ntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus, modulus2);
        }
        for (int l = 0; l < 8; l++) {
            buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l] = samples[l];
        }

        __syncthreads();
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = buffer[pad_tid * (n1 + pad) + pad_idx + group * j];
        }
    }
}

__global__ static void
NTT8pointPerThread_for_ext_kernel2(uint64_tt *inout,
                           const uint64_tt *twiddles,
                           const uint64_tt *twiddles_shoup,
                           const int poly_num,
                           const int start_poly_idx,
                           const int mod_num,
                           const int start_mod_idx,
                           const int n,
                           const int n1,
                           const int n2,
                           const int poly_mod_len,
                           const int cipher_mod_num) {
    extern __shared__ uint64_tt buffer[];
    uint64_tt *inout_a = inout + (blockIdx.y * poly_mod_len * n) + start_poly_idx * n + (blockIdx.z * cipher_mod_num * n);
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n / 8 * mod_num); tid += blockDim.x * gridDim.x) {
        int group = n2 / 8;
        int set = threadIdx.x / group;
        // size of a block
        uint64_tt samples[8];
        int t = n2 / 2;
        int twr_idx = mod_num - 1 - (tid / (n / 8)) + start_mod_idx;
        // index in n/2 range
        int n_idx = tid % (n / 8);
        // tid'th block
        int m_idx = n_idx / (t / 4);
        int t_idx = n_idx % (t / 4);

        int poly_idx = mod_num - 1 - (tid / (n / 8));
        // base address
        uint64_tt *data_ptr = inout_a + (poly_idx) * n;
        uint64_tt modulus = pqt_cons[twr_idx];
        uint64_tt modulus2 = modulus << 1;
        const uint64_tt *psi = twiddles + n * twr_idx;
        const uint64_tt *psi_shoup = twiddles_shoup + n * twr_idx;
        int n_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }
        int tw_idx = n1 + m_idx;
        ntt8(samples, psi, psi_shoup, tw_idx, modulus, modulus2);
        for (int j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = samples[j];
        }
        int tail = 0;
        __syncthreads();

        for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
            int m_idx2 = t_idx / (k / 4);
            int t_idx2 = t_idx % (k / 4);
            for (int l = 0; l < 8; l++) {
                samples[l] =
                        buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            int tw_idx2 = j * tw_idx + m_idx2;
            ntt8(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            for (int l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == t / 8)
                tail = 1;
            if (j == t / 16)
                tail = 2;
            __syncthreads();
        }

        for (int l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        if (tail == 1) {
            int tw_idx2 = t * tw_idx + 4 * t_idx;
            for( int i = 0 ; i < 4; i++ ){
                CTBased_butterfly(samples[2 * i], samples[2 * i + 1], psi[tw_idx2 + i], psi_shoup[tw_idx2 + i], modulus, modulus2);
            }
        }
        else if (tail == 2) {
            int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
            ntt4(samples, psi, psi_shoup, tw_idx2, modulus, modulus2);
            ntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus, modulus2);
        }
        for (int l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        __syncthreads();

        // final reduction
        for (int j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
            csub_q(samples[j], modulus2);
            csub_q(samples[j], modulus);
        }
        for (int j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}