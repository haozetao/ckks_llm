#pragma once
#include "../uint128.cuh"
#include "../ntt_60bit.cuh"

#define warp_number 8

__global__ void NTT256_kernel(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size, int N1, uint64_tt* psi_table_device, uint64_tt* psi_shoup_table_device)
{
    register int idx_thread = threadIdx.x;
    register int idx_warp = threadIdx.y;

    register int idx_block = blockIdx.x;
    register int idx_mod = blockIdx.y;
    register int idx_batch = blockIdx.z;
    if(idx_block * warp_number + idx_warp >= poly_num) return;

    uint64_tt* psi = psi_table_device + N1 * (start_mod_idx + idx_mod);
    uint64_tt* psi_shoup = psi_shoup_table_device + N1 * (start_mod_idx + idx_mod);

    uint64_tt mod = ringpack_pq_cons[start_mod_idx + idx_mod];
    uint64_tt mod2 = mod << 1;


    extern __shared__ uint64_tt shared_mem[];
    uint64_tt* shared_mem_this_warp = shared_mem + idx_warp * 256;

    uint64_tt* data_in_this_mod = data + idx_batch * poly_num * mod_num * 256 + idx_mod * poly_num * 256;
    uint64_tt* data_in_this_block = data_in_this_mod + idx_block * warp_number * 256;
    uint64_tt* data_in_this_warp = data_in_this_block + idx_warp * 256;

    register uint64_tt buffer[8];

    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i * 32 + idx_thread] = data_in_this_warp[i * 32 + idx_thread];
    }
    
    // 0-128, 0-64, 0-32
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i * 32 + idx_thread];
    }
    ntt8(buffer, psi, psi_shoup, 1, mod, mod2);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i * 32 + idx_thread] = buffer[i];
    }

    // 0-16, 0-8, 0-4
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i * 4 + (idx_thread%4 + idx_thread/4*32)];
    }
    ntt8(buffer, psi, psi_shoup, 8 + (idx_thread%4 + idx_thread/4*32)/32, mod, mod2);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i * 4 + (idx_thread%4 + idx_thread/4*32)] = buffer[i];
    }

    // 0-2, 0-1
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i + (idx_thread * 8)];
    }

    ntt4(buffer, psi, psi_shoup, 64 + idx_thread*2, mod, mod2);
    ntt4(buffer + 4, psi, psi_shoup, 64 + idx_thread*2+1, mod, mod2);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        csub_q(buffer[i], mod2);
        csub_q(buffer[i], mod);
        data_in_this_warp[i + (idx_thread * 8)] = buffer[i];
    }
}

__global__ void INTT256_kernel(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size, int N1, uint64_tt* psiinv_table_device, uint64_tt* psiinv_shoup_table_device, uint64_tt* N1_inv_device, uint64_tt* N1_inv_shoup_device)
{
    register int idx_thread = threadIdx.x;
    register int idx_warp = threadIdx.y;

    register int idx_block = blockIdx.x;
    register int idx_mod = blockIdx.y;
    register int idx_batch = blockIdx.z;
    if(idx_block * warp_number + idx_warp >= poly_num) return;

    uint64_tt* psiinv = psiinv_table_device + N1 * (start_mod_idx + idx_mod);
    uint64_tt* psiinv_shoup = psiinv_shoup_table_device + N1 * (start_mod_idx + idx_mod);

    uint64_tt mod = ringpack_pq_cons[start_mod_idx + idx_mod];
    uint64_tt mod2 = mod << 1;

    extern __shared__ uint64_tt shared_mem[];
    uint64_tt* shared_mem_this_warp = shared_mem + idx_warp * 256;

    uint64_tt* data_in_this_mod = data + idx_batch * poly_num * mod_num * 256 + idx_mod * poly_num * 256;
    uint64_tt* data_in_this_block = data_in_this_mod + idx_block * warp_number * 256;
    uint64_tt* data_in_this_warp = data_in_this_block + idx_warp * 256;

    register uint64_tt buffer[8];

    uint64_tt inv_degree_mod = N1_inv_device[start_mod_idx + idx_mod];
    uint64_tt inv_degree_mod_shoup = N1_inv_shoup_device[start_mod_idx + idx_mod];
    
    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i * 32 + idx_thread] = data_in_this_warp[i * 32 + idx_thread];
    }

    // 0-1, 0-2, 0-4
    int first_data_idx = idx_thread * 8;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i + first_data_idx];
    }
    intt8(buffer, psiinv, psiinv_shoup, 32 + first_data_idx / 8, mod, mod2);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i + first_data_idx] = buffer[i];
    }

    // 0-8, 0-16, 0-32
    first_data_idx = (idx_thread / 8) * 64 + idx_thread % 8;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i * 8 + first_data_idx];
    }
    intt8(buffer, psiinv, psiinv_shoup, 4 + first_data_idx / 64, mod, mod2);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        shared_mem_this_warp[i * 8 + first_data_idx] = buffer[i];
    }
    
    // 0-64, 0-128
    first_data_idx = idx_thread;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        buffer[i] = shared_mem_this_warp[i * 32 + first_data_idx];
    }
    intt4(buffer, psiinv, psiinv_shoup, 1, mod, mod2);
    intt4(buffer + 1, psiinv, psiinv_shoup, 1, mod, mod2);

    #pragma unroll
    for(int i = 0; i < 4; i++){
        const uint64_tt hi = __umul64hi(buffer[i], inv_degree_mod_shoup);
        buffer[i] = buffer[i] * inv_degree_mod - hi *  mod;
    }

    #pragma unroll
    for(int i = 0; i < 8; i++){
        csub_q(buffer[i], mod);
        data_in_this_warp[i * 32 + first_data_idx] = buffer[i];
    }
}