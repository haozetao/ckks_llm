#pragma once

#include "MLWECiphertext.cuh"
#include "PCMM_Scheme.h"
#include <mma.h>

#define TILE_WIDTH_M 32
#define TILE_WIDTH_N 32
#define TILE_WIDTH_K 32
#define USE_TCU false

__forceinline__ __device__ void ppmm_16x16(int64_tt* mat_A_16x16, uint64_tt* mat_B_16x16, uint128_tt mat_acc_16x16[8], int thread_idx, uint64_tt mod)
{
    // 16x16 matrix multiplication
    // each thread handles 1x16x8
#pragma unroll
    for(int i = 0; i < 8; i++){
        int line_idx = thread_idx / 16 + i * 2;
        int idx_in_line = thread_idx % 16;

    #pragma unroll
        for(int iter = 0; iter < 16; iter++){
            long temp = mat_A_16x16[line_idx*16 + iter];

            uint64_tt a = temp > 0 ? temp : mod + temp;
            uint64_tt b = mat_B_16x16[iter*16 + idx_in_line];

            madc_uint64_uint64_uint128(a, b, mat_acc_16x16[i]);
        }
    }
}

using namespace nvcuda;
__forceinline__ __device__ void ppmm_16x16_tcu(uint8_tt plain_buffer_split[][256], uint8_tt cipher_buffer_split[][256],
                                               int shared_acc_buffer[256], uint128_tt mat_acc_16x16[8], 
                                               int idx_thread, int idx_warp, int idx_block, int idy_block, uint64_tt mod)
{
    #pragma unroll
    for (int diag = 0; diag < 8; diag++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
        wmma::fill_fragment(acc_frag, 0);
        
        #pragma unroll
        for (int k = 0; k <= diag; k++) {
            int l = diag - k;
            wmma::load_matrix_sync(a_frag, plain_buffer_split[k], 16);
            wmma::load_matrix_sync(b_frag, cipher_buffer_split[l], 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        wmma::store_matrix_sync(shared_acc_buffer, acc_frag, 16, wmma::mem_row_major);
        
        #pragma unroll
        for (int elem = 0; elem < 8; elem++) {
            int idx = idx_thread + elem * 32;
            uint128_tt partial = uint128_tt((uint32_t)shared_acc_buffer[idx]) << (8 * diag);
            add_uint128_uint128(mat_acc_16x16[elem], partial);
        }
    }

    #pragma unroll
    for (int diag = 8; diag < 15; diag++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
        wmma::fill_fragment(acc_frag, 0);
        
        #pragma unroll
        for (int k = diag-7; k <= 7; k++) {
            int l = diag - k;
            wmma::load_matrix_sync(a_frag, plain_buffer_split[k], 16);
            wmma::load_matrix_sync(b_frag, cipher_buffer_split[l], 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        wmma::store_matrix_sync(shared_acc_buffer, acc_frag, 16, wmma::mem_row_major);
        
        #pragma unroll
        for (int elem = 0; elem < 8; elem++) {
            int idx = idx_thread + elem * 32;
            uint64_tt partial = uint64_tt(shared_acc_buffer[idx]) << (8 * (diag - 8));
            mat_acc_16x16[elem].high += partial;
        }
    }
}

#include <cooperative_groups.h>
#include <cuda/pipeline>

using namespace cooperative_groups;
template<int mat_N>
__global__ void pcmm_cuda_core_kernel(float* plain_mat, uint64_tt** mlwe_cipher_pointer, uint64_tt* output, int N1, int mlwe_rank, int mod_num, int p_mod_num, long scaler, uint64_tt* qiInv_device, uint64_tt* qiInv_shoup_device)
{
	// shared memory
    #if TILE_WIDTH_K == 32
        __shared__ int64_tt plain_buffer[2][TILE_WIDTH_M*TILE_WIDTH_N/(16*16)][16*16];
        __shared__ uint64_tt cipher_buffer[2][TILE_WIDTH_N*TILE_WIDTH_K/(16*16)][16*16];
    #elif TILE_WIDTH_K == 64
        extern __shared__ char smem[];
        int64_tt* plain_buffer = (int64_tt*)(smem);
        uint64_tt* cipher_buffer = (uint64_tt*)(smem + (TILE_WIDTH_M * TILE_WIDTH_N * sizeof(int64_tt) * 2));
    #endif

	register uint128_tt acc_buffer[8];

    int idx_thread = threadIdx.x;
    int idx_warp = threadIdx.y;

    int idx_tile_M = blockIdx.x;
    int idx_tile_K = blockIdx.y;

    int idx_warp_in_tile = idx_warp % (TILE_WIDTH_K / 16);    
    int idy_warp_in_tile = idx_warp / (TILE_WIDTH_K / 16);
    
            int current_buffer = 0;
            auto block = this_thread_block();
            // 创建一个 pipeline state，用于 double buffering
            // 每个 block 使用一个 pipeline
            __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> shared_state;
            auto pipeline = cuda::make_pipeline(block, &shared_state);
    {
        for(int idx_mod = mod_num - 1; idx_mod >= 0; idx_mod--)
        {
            uint64_tt q = ringpack_pq_cons[p_mod_num + idx_mod];
            uint128_tt mu = {ringpack_pq_mu_high_cons[p_mod_num+idx_mod], ringpack_pq_mu_low_cons[p_mod_num+idx_mod]};

            #pragma unroll
            for(int j = 0; j < 8; j++){
                // register
                acc_buffer[j] = 0;
            }


            for(int idx_mat_N = 0; idx_mat_N < mat_N / TILE_WIDTH_N; idx_mat_N++)
            {
                pipeline.producer_acquire();
                #if TILE_WIDTH_K == 32
                    #pragma unroll
                    for(int j = 0; j < 8; j++){
                        int idx_line = idx_thread % 16;
                        int idy_line = idx_thread / 16 + j*2; 

                        int idx_plain = idx_mat_N * TILE_WIDTH_N + idx_warp_in_tile * 16 + idx_line;
                        int idy_plain = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + idy_line;

                        plain_buffer[current_buffer][idx_warp][j*32 + idx_thread] = plain_mat[(idy_plain*mat_N + idx_plain)] * scaler;
                    }
                    uint64_tt* dst_cipher = &cipher_buffer[current_buffer][idx_warp][idx_thread * 8];
                #elif TILE_WIDTH_K == 64
                    const int ELEMENTS_PER_TILE = 16 * 16; // 256
                    const int NUM_THREADS = 8 * 32; // 8 waprs
                    const int TOTAL_ELEMENTS = TILE_WIDTH_M * TILE_WIDTH_N;

                    int thread_id = idx_warp * 32 + idx_thread; // tid: 0 ~ 255

                    #pragma unroll
                    for (int i = 0; i < (TOTAL_ELEMENTS) / NUM_THREADS; i++) {
                        int flat_idx = thread_id + i * NUM_THREADS;

                        int tile_id = flat_idx / ELEMENTS_PER_TILE;
                        int elem_in_tile = flat_idx % ELEMENTS_PER_TILE;
                        int x = elem_in_tile % 16;
                        int y = elem_in_tile / 16;

                        int tile_n = tile_id % (TILE_WIDTH_N / 16);
                        int tile_m = tile_id / (TILE_WIDTH_N / 16);

                        int idx_plain = idx_mat_N * (TILE_WIDTH_N) + tile_n * 16 + x;  // N 维
                        int idy_plain = idx_tile_M * (TILE_WIDTH_M) + tile_m * 16 + y; // M 维

                        plain_buffer[current_buffer*(TOTAL_ELEMENTS) + tile_id*256 + elem_in_tile] = float(plain_mat[idy_plain * mat_N + idx_plain]) * scaler;
                    }

                    uint64_tt* dst_cipher = &cipher_buffer[current_buffer*(TILE_WIDTH_N*TILE_WIDTH_K) + idx_warp*256 + idx_thread * 8];
                #endif
            
                int idx_line = (idx_thread % 2) * 8;
                int idy_line = idx_thread / 2;

                int idx_cipher = idx_mod * (mlwe_rank + 1) * N1 +
                                idx_tile_K * TILE_WIDTH_K +
                                idx_warp_in_tile * 16 + idx_line;
                int idy_cipher = idx_mat_N * TILE_WIDTH_N +
                                idy_warp_in_tile * 16 + idy_line;

                const uint64_tt* src_cipher = &mlwe_cipher_pointer[idy_cipher][idx_cipher];
                cuda::memcpy_async(dst_cipher, src_cipher, sizeof(uint64_tt) * 8, pipeline);
                pipeline.producer_commit();

                const int idx_m = idx_warp / (TILE_WIDTH_K/16);
                const int idx_k = idx_warp % (TILE_WIDTH_K/16);
                
                pipeline.consumer_wait();
                
                if(idx_warp == idx_m * (TILE_WIDTH_K/16) + idx_k){
                    #pragma unroll
                    for(int idx_n = 0; idx_n < (TILE_WIDTH_N/16); idx_n++){
                        #if TILE_WIDTH_K == 32
                            ppmm_16x16(plain_buffer[current_buffer][(TILE_WIDTH_N/16) * idx_m + idx_n],
                                       cipher_buffer[current_buffer][(TILE_WIDTH_K/16) * idx_n + idx_k], acc_buffer, idx_thread, q);
                        #elif TILE_WIDTH_K == 64
                            ppmm_16x16(&plain_buffer[current_buffer*(TILE_WIDTH_M*TILE_WIDTH_N) + ((TILE_WIDTH_N/16) * idx_m + idx_n)*256], 
                                       &cipher_buffer[current_buffer*(TILE_WIDTH_N*TILE_WIDTH_K) + ((TILE_WIDTH_K/16) * idx_n + idx_k)*256], acc_buffer, idx_thread, q);
                        #endif
                    }
                }
                current_buffer ^= 1;
                pipeline.consumer_release();
            }

            if(idx_mod == mod_num - 1){
                #pragma unroll
                for(int j = 0; j < 8; j++){
                    int idx_line = idx_thread % 16;
                    int idy_line = idx_thread / 16 + j*2;

                    // register
                    uint128_tt acc = acc_buffer[j];

                    singleBarrett_new(acc, q, mu);
                    
                    int idy = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + idy_line;
                    int idx = idx_mod * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_line;
                    output[idy * (mlwe_rank + 1) * N1 * mod_num + idx] = acc.low;
                }
            } else {
                #pragma unroll
                for(int j = 0; j < 8; j++){
                    int idx_line = idx_thread % 16;
                    int idy_line = idx_thread / 16 + j*2;

                    // register
                    uint128_tt acc = acc_buffer[j];

                    singleBarrett_new(acc, q, mu);
                    
                    int idy = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + idy_line;
                    int idx = idx_mod * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_line;
                        
                    // rescale
                    int idx_last = (mod_num - 1) * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_line;
                    uint64_tt temp = acc.low + q - output[idy * (mlwe_rank + 1) * N1 * mod_num + idx_last];
                    uint64_tt qiInv = qiInv_device[idx_mod];
                    uint64_tt qiInv_shoup = qiInv_shoup_device[idx_mod];
                    output[idy * (mlwe_rank + 1) * N1 * mod_num + idx] = mulMod_shoup(temp, qiInv, qiInv_shoup, q);
                }
            }
        }
    }
}

__host__ void PCMM_Scheme::PPMM(float* plain_mat, vector<MLWECiphertext*> mlwe_cipher_decomposed, int mat_M, int mat_N, int mat_K)
{
    if(mat_K != pcmm_context.N1){
        cout << "matrix K should be equal to N1!" << endl;
        return;
    }
    if(mat_N != mlwe_cipher_decomposed.size()){
        cout << "matrix N should be equal to mlwe number!" << endl;
        return;
    }
    int N = scheme.context.N;
    
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;

    int mlwe_num = mlwe_cipher_decomposed.size();
    // if(mlwe_cipher_decomposed.size() != mlwe_rank){
    //     cout << "only support packing k mlwe -> 1 rlwe!" << endl;
    //     return;
    // }
    
    for(int i = 0; i < mlwe_num; i++){
        // repacking_cipher_pointer[mlwe_rank - i - 1] = mlwe_cipher_decomposed[i]->cipher_device;
        repacking_cipher_pointer[i] = mlwe_cipher_decomposed[i]->cipher_device;
    }
    cudaMemcpy(repacking_cipher_pointer_device, repacking_cipher_pointer.data(), sizeof(uint64_tt*) * mlwe_num, cudaMemcpyHostToDevice);

    long scaler = to_double(mlwe_cipher_decomposed[0]->scale);
    cout<<"scaler: "<<scaler<<endl;


    int SMem_size = (TILE_WIDTH_M * TILE_WIDTH_N * (sizeof(int64_tt)) + TILE_WIDTH_N * TILE_WIDTH_K * sizeof(uint64_tt)) * 2;// + TILE_WIDTH_M * TILE_WIDTH_K * sizeof(int);
    cout<<"smem size: " << SMem_size/1024.0<< "KB" << endl;

    // each block handles a 32*32 tile
    dim3 ppmm_block(mat_M / TILE_WIDTH_M, mat_K * (mlwe_rank + 1) / TILE_WIDTH_K);
    // 4 warps, each warp handles 16*16
    // each thread handles 1*16*8
    dim3 ppmm_thread(32, (TILE_WIDTH_M*TILE_WIDTH_K) / (16*16));
    int l = 1;

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    CUDATimer cuTimer;

    cuTimer.start();
    // tile size: 32*32, each warp handles 16*16, need 4 warps
    if(TILE_WIDTH_M == 32){
        if(mat_N == 128){
            // 设置shared_memory 大小
            cudaFuncSetAttribute(&pcmm_cuda_core_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
            pcmm_cuda_core_kernel<128> <<<ppmm_block, ppmm_thread, SMem_size>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
                context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
                cout<<"call 1 mat_N = "<<128<<endl;
        } else if (mat_N == 256){
            // 设置shared_memory 大小
            cudaFuncSetAttribute(&pcmm_cuda_core_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
            pcmm_cuda_core_kernel<256> <<<ppmm_block, ppmm_thread, SMem_size>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
                context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
                cout<<"call 1 mat_N = "<<256<<endl;
        } else if (mat_N == 512){
            // 设置shared_memory 大小
            cudaFuncSetAttribute(&pcmm_cuda_core_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
            pcmm_cuda_core_kernel<512> <<<ppmm_block, ppmm_thread, SMem_size>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
                context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
                cout<<"call 1 mat_N = "<<512<<endl;
        } else if (mat_N == 768){
            // 设置shared_memory 大小
            cudaFuncSetAttribute(&pcmm_cuda_core_kernel<768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
            pcmm_cuda_core_kernel<768> <<<ppmm_block, ppmm_thread, SMem_size>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
                context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
                cout<<"call 1 mat_N = "<<768<<endl;
        }
    } else {
        cout << "tile size should be equal to 32!" << endl;
        return;
    }
    float temp = cuTimer.stop();
    printf("ppmm kernel time: %.3f ms\n", temp);
    printf("pcmm thread(%d, %d), block(%d, %d)\n", ppmm_thread.x, ppmm_thread.y, ppmm_block.x, ppmm_block.y);

    for(int i = 0; i < mlwe_num; i++){
        mlwe_cipher_decomposed[i]->scale = mlwe_cipher_decomposed[i]->scale * scaler / pcmm_context.q_ringpack[1];
        cudaMemcpyAsync(mlwe_cipher_decomposed[i]->cipher_device, ppmm_output + i * N1 * (mlwe_rank+1) * 2, sizeof(uint64_tt) * N1 * (mlwe_rank+1), cudaMemcpyDeviceToDevice);
    }
}