#pragma once

#include "MLWECiphertext.cuh"
#include "PCMM_Scheme.h"


__forceinline__ __device__ void ppmm_16x16_reg(int64_tt* mat_A_16x16, uint64_tt mat_B_16x16[][16*16], uint128_tt mat_acc_2x8[][8], int thread_idx, int p_mod_num)
{
    // 16x16 matrix multiplication
    // each thread handles 1x16x8
#pragma unroll
    for(int i = 0; i < 8; i++){
        int line_idx = thread_idx / 16 + i * 2;
        int idx_in_line = thread_idx % 16;

    #pragma unroll
        for(int idx_mod = 0; idx_mod < 2; idx_mod++)
        {
            uint128_tt acc = mat_acc_2x8[idx_mod][i];

        #pragma unroll
            for(int iter = 0; iter < 16; iter++)
            {
                long temp = mat_A_16x16[line_idx*16 + iter];

                uint64_tt mod = ringpack_pq_cons[p_mod_num + idx_mod];

                uint64_tt a = temp > 0 ? temp : mod + temp;
                uint64_tt b = mat_B_16x16[idx_mod][iter*16 + idx_in_line];

                madc_uint64_uint64_uint128(a, b, acc);
            }
            mat_acc_2x8[idx_mod][i] = acc;
        }
    }
}

template<int TILE_WIDTH_M, int TILE_WIDTH_N, int TILE_WIDTH_K, int mat_N, int ppmm_warp_number>
__global__ void pcmm_cuda_core_reg_kernel(float* plain_mat, uint64_tt** mlwe_cipher_pointer, uint64_tt* output, int N1, int mlwe_rank, int mod_num, int p_mod_num, long scaler, uint64_tt* qiInv_device, uint64_tt* qiInv_shoup_device)
{
	// shared memory
	__shared__ int64_tt plain_buffer[TILE_WIDTH_M*TILE_WIDTH_N/(16*16)][16*16];
	__shared__ uint64_tt cipher_buffer_q[TILE_WIDTH_N*TILE_WIDTH_K/(16*16)][2][16*16];

	register uint128_tt acc_buffer_reg[2][8]={0};

    int idx_thread = threadIdx.x;
    int idx_warp = threadIdx.y;

    int idx_tile_M = blockIdx.x;
    int idx_tile_K = blockIdx.y;


    int idx_warp_in_tile = idx_warp % (TILE_WIDTH_K / 16);  
    int idy_warp_in_tile = idx_warp / (TILE_WIDTH_K / 16);


    uint64_tt q0 = ringpack_pq_cons[p_mod_num];
    uint128_tt mu0 = {ringpack_pq_mu_high_cons[p_mod_num], ringpack_pq_mu_low_cons[p_mod_num]};
    uint64_tt q1 = ringpack_pq_cons[p_mod_num+1];
    uint128_tt mu1 = {ringpack_pq_mu_high_cons[p_mod_num+1], ringpack_pq_mu_low_cons[p_mod_num+1]};

#pragma unroll
    for(int i = 0; i < mat_N / TILE_WIDTH_N; i++)
    {
        // 32 threads read 2 line of the 16*16 matrix
    #pragma unroll
        for(int j = 0; j < 8; j++){
            int line_idx = idx_thread / 16 + j*2;
            int idx_in_line = idx_thread % 16;

            int row_cipher = i * TILE_WIDTH_N + idy_warp_in_tile * 16 + line_idx;
            int col_cipher = idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;
            cipher_buffer_q[idx_warp][0][j*32 + idx_thread] = mlwe_cipher_pointer[row_cipher][col_cipher];
            cipher_buffer_q[idx_warp][1][j*32 + idx_thread] = mlwe_cipher_pointer[row_cipher][col_cipher + (mlwe_rank + 1) * N1];

            if(idx_warp<4){
                int row_plain = idx_tile_M * TILE_WIDTH_M + (idx_warp_in_tile/2) * 16 + line_idx;
                int col_plain = i * TILE_WIDTH_N + (idx_warp_in_tile%2) * 16 + idx_in_line;
                plain_buffer[idx_warp][j*32 + idx_thread]  = double(plain_mat[(row_plain*mat_N + col_plain)]) * scaler;   
            }
        }
        __syncthreads();

        for(int idx_i = 0; idx_i < TILE_WIDTH_M/16; idx_i++){
            for(int idx_j=0; idx_j < TILE_WIDTH_K/16; idx_j++){
                if(idx_warp==idx_i*(TILE_WIDTH_K/16)+idx_j){
                    #pragma unroll
                    for(int idx_k = 0;idx_k<2;idx_k++){
                        ppmm_16x16_reg(plain_buffer[2*idx_i + idx_k], cipher_buffer_q[idx_j + (TILE_WIDTH_K/16) * idx_k], acc_buffer_reg, idx_thread, p_mod_num);
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int j = 0; j < 8; j++){
        int line_idx = idx_thread / 16 + j*2;
        int idx_in_line = idx_thread % 16;

        int row = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + line_idx;
        int col = idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;
        int col_1 = (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;

        uint128_tt acc = acc_buffer_reg[0][j];
        singleBarrett_new(acc, q0, mu0);

        uint128_tt acc_1 = acc_buffer_reg[1][j];
        singleBarrett_new(acc_1, q1, mu1);
        
        uint64_tt temp = acc.low + q0 - acc_1.low;

        uint64_tt qiInv = qiInv_device[0];
        uint64_tt qiInv_shoup = qiInv_shoup_device[0];

        output[row * (mlwe_rank + 1) * N1 * mod_num + col] = mulMod_shoup(temp, qiInv, qiInv_shoup, q0);
    }
}

__forceinline__ __device__ void ppmm_16x16(int64_tt* mat_A_16x16, uint64_tt* mat_B_16x16, uint128_tt mat_acc_16x16[32][8], int thread_idx, uint64_tt mod)
{
    // 16x16 matrix multiplication
    // each thread handles 1x16x8
#pragma unroll
    for(int i = 0; i < 8; i++){
        int line_idx = thread_idx / 16 + i * 2;
        int idx_in_line = thread_idx % 16;

        uint128_tt acc = mat_acc_16x16[thread_idx][i];

    #pragma unroll
        for(int iter = 0; iter < 16; iter++){
            long temp = mat_A_16x16[line_idx*16 + iter];

            uint64_tt a = temp > 0 ? temp : mod + temp;
            uint64_tt b = mat_B_16x16[iter*16 + idx_in_line];

            madc_uint64_uint64_uint128(a, b, acc);
        }
        mat_acc_16x16[thread_idx][i] = acc;
    }
}

__forceinline__ __device__ void ppmm_16x16_reg2(int64_tt* mat_A_16x16, uint64_tt* mat_B_16x16, uint128_tt mat_acc_16x16[8], int thread_idx, uint64_tt mod)
{
    // 16x16 matrix multiplication
    // each thread handles 1x16x8
#pragma unroll
    for(int i = 0; i < 8; i++){
        int line_idx = thread_idx / 16 + i * 2;
        int idx_in_line = thread_idx % 16;

        uint128_tt acc = mat_acc_16x16[i];

    #pragma unroll
        for(int iter = 0; iter < 16; iter++){
            long temp = mat_A_16x16[line_idx*16 + iter];

            uint64_tt a = temp > 0 ? temp : mod + temp;
            uint64_tt b = mat_B_16x16[iter*16 + idx_in_line];

            madc_uint64_uint64_uint128(a, b, acc);
        }
        mat_acc_16x16[i] = acc;
    }
}

template<int TILE_WIDTH_M, int TILE_WIDTH_N, int TILE_WIDTH_K, int mat_N, int ppmm_warp_number>
__global__ void pcmm_cuda_core_kernel(float* plain_mat, uint64_tt** mlwe_cipher_pointer, uint64_tt* output, int N1, int mlwe_rank, int mod_num, int p_mod_num, long scaler, uint64_tt* qiInv_device, uint64_tt* qiInv_shoup_device)
{
	// shared memory
	__shared__ int64_tt plain_buffer[TILE_WIDTH_M*TILE_WIDTH_N/(16*16)][16*16];
	__shared__ uint64_tt cipher_buffer[TILE_WIDTH_N*TILE_WIDTH_K/(16*16)][16*16];
	// __shared__ uint128_tt acc_buffer[TILE_WIDTH_M*TILE_WIDTH_K/(16*16)][32][8];
	uint128_tt acc_buffer[8];


    int idx_thread = threadIdx.x;
    int idx_warp = threadIdx.y;

    int idx_tile_M = blockIdx.x;
    int idx_tile_K = blockIdx.y;

    int idx_warp_in_tile = idx_warp % (TILE_WIDTH_K / 16);    
    int idy_warp_in_tile = idx_warp / (TILE_WIDTH_K / 16);

    for(int idx_mod = mod_num - 1; idx_mod >= 0; idx_mod--)
    {
        uint64_tt q = ringpack_pq_cons[p_mod_num + idx_mod];
        uint128_tt mu = {ringpack_pq_mu_high_cons[p_mod_num+idx_mod], ringpack_pq_mu_low_cons[p_mod_num+idx_mod]};

        #pragma unroll
        for(int j = 0; j < 8; j++){
            int line_idx = idx_thread / 16 + j*2;
            int idx_in_line = idx_thread % 16;
            acc_buffer[j] = 0;
        }

        // #pragma unroll
        for(int i = 0; i < mat_N / TILE_WIDTH_N; i++)
        {
            // 32 threads read 2 line of the 16*16 matrix
        #pragma unroll
            for(int j = 0; j < 8; j++){
                int line_idx = idx_thread / 16 + j*2;
                int idx_in_line = idx_thread % 16;

                int row_plain = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + line_idx;
                int col_plain = i * TILE_WIDTH_N + idx_warp_in_tile * 16 + idx_in_line;

                int row_cipher = i * TILE_WIDTH_N + idy_warp_in_tile * 16 + line_idx;
                int col_cipher = idx_mod * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;

                plain_buffer[idx_warp][j*32 + idx_thread]  = double(plain_mat[(row_plain*mat_N + col_plain)]) * scaler;
                cipher_buffer[idx_warp][j*32 + idx_thread] = mlwe_cipher_pointer[row_cipher][col_cipher];
            }
            __syncthreads();

            for(int idx_i = 0; idx_i < TILE_WIDTH_M/16; idx_i++){
                for(int idx_j = 0; idx_j < TILE_WIDTH_K/16; idx_j++){
                    if(idx_warp == idx_i * (TILE_WIDTH_K/16) + idx_j){
                        #pragma unroll
                        for(int idx_k = 0; idx_k < 2; idx_k++){
                            // ppmm_16x16(plain_buffer[2*idx_i + idx_k], cipher_buffer[idx_j + (TILE_WIDTH_K/16) * idx_k], acc_buffer[idx_warp], idx_thread, q);
                            ppmm_16x16_reg2(plain_buffer[2*idx_i + idx_k], cipher_buffer[idx_j + (TILE_WIDTH_K/16) * idx_k], acc_buffer, idx_thread, q);
                        }
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for(int j = 0; j < 8; j++){
            int line_idx = idx_thread / 16 + j*2;
            int idx_in_line = idx_thread % 16;

            uint128_tt acc = acc_buffer[j];
            singleBarrett_new(acc, q, mu);
            
            int row = idx_tile_M * TILE_WIDTH_M + idy_warp_in_tile * 16 + line_idx;
            int col = idx_mod * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;

            if(idx_mod == mod_num - 1){
                output[row * (mlwe_rank + 1) * N1 * mod_num + col] = acc.low;
            } else { 
                // rescale
                int col_last = (mod_num - 1) * (mlwe_rank + 1) * N1 + idx_tile_K * TILE_WIDTH_K + idx_warp_in_tile * 16 + idx_in_line;
                uint64_tt temp = acc.low + q - output[row * (mlwe_rank + 1) * N1 * mod_num + col_last];
                uint64_tt qiInv = qiInv_device[idx_mod];
                uint64_tt qiInv_shoup = qiInv_shoup_device[idx_mod];
                output[row * (mlwe_rank + 1) * N1 * mod_num + col] = mulMod_shoup(temp, qiInv, qiInv_shoup, q);
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
    int N = scheme.context.N;
    
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;

    if(mlwe_cipher_decomposed.size() != mlwe_rank){
        cout << "only support packing k mlwe -> 1 rlwe!" << endl;
        return;
    }
    
    for(int i = 0; i < mlwe_rank; i++){
        // repacking_cipher_pointer[mlwe_rank - i - 1] = mlwe_cipher_decomposed[i]->cipher_device;
        repacking_cipher_pointer[i] = mlwe_cipher_decomposed[i]->cipher_device;
    }
    cudaMemcpy(repacking_cipher_pointer_device, repacking_cipher_pointer.data(), sizeof(uint64_tt*) * mlwe_rank, cudaMemcpyHostToDevice);

    long scaler = to_double(mlwe_cipher_decomposed[0]->scale);
    cout<<"scaler: "<<scaler<<endl;


    int TILE_WIDTH_M = 32;
    int TILE_WIDTH_N = 32;
    int TILE_WIDTH_K = 32;
    // tile size: 32*32, each warp handles 16*16, need 4 warps
    if(TILE_WIDTH_M == 32 && TILE_WIDTH_K == 32){
        int SMem_size = TILE_WIDTH_M * TILE_WIDTH_N * sizeof(int64_tt) + TILE_WIDTH_N * TILE_WIDTH_K * sizeof(uint64_tt) + TILE_WIDTH_M * TILE_WIDTH_K * sizeof(uint128_tt);
        // 设置shared_memory 大小
        cudaFuncSetAttribute(&pcmm_cuda_core_kernel<32, 32, 32, 256, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
        cout<<"smem size:" << SMem_size<<endl;
        
        // each block handles a 32*32 tile
        dim3 ppmm_block(mat_M / TILE_WIDTH_M, N1 * (mlwe_rank + 1) / TILE_WIDTH_K);
        // 4 warps, each warp handles 16*16
        // each thread handles 1*16*8
        dim3 ppmm_thread(32, (TILE_WIDTH_M*TILE_WIDTH_K) / (16*16));

        int l = 1;
        pcmm_cuda_core_kernel<32, 32, 32, 256, 4> <<<ppmm_block, ppmm_thread>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
            context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);



        // int SMem_size = TILE_WIDTH_M * TILE_WIDTH_N * sizeof(int64_tt) + TILE_WIDTH_N * TILE_WIDTH_K * sizeof(uint64_tt);
        // // 设置shared_memory 大小
        // cudaFuncSetAttribute(&pcmm_cuda_core_reg_kernel<32, 32, 32, 256, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMem_size);
        // cout<<"smem size:" << SMem_size<<endl;
        
        // // each block handles a 32*32 tile
        // dim3 ppmm_block(mat_M / TILE_WIDTH_M, N1 * (mlwe_rank + 1) / TILE_WIDTH_K);
        // // 4 warps, each warp handles 16*16
        // // each thread handles 1*16*8
        // dim3 ppmm_thread(32, (TILE_WIDTH_M*TILE_WIDTH_K) / (16*16));

        // int l = 1;
        // pcmm_cuda_core_reg_kernel<32, 32, 32, 256, 4> <<<ppmm_block, ppmm_thread>>>(plain_mat, repacking_cipher_pointer_device, ppmm_output, N1, mlwe_rank, ringpack_q_count, ringpack_p_count, scaler, 
        //     context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
    } else {
        cout << "tile size should be equal to 32!" << endl;
        return;
    }

    for(int i = 0; i < mlwe_rank; i++){
        mlwe_cipher_decomposed[i]->scale = mlwe_cipher_decomposed[i]->scale * scaler / pcmm_context.q_ringpack[1];
        cudaMemcpyAsync(mlwe_cipher_decomposed[i]->cipher_device, ppmm_output + i * N1 * (mlwe_rank+1) * 2, sizeof(uint64_tt) * N1 * (mlwe_rank+1), cudaMemcpyDeviceToDevice);
    }
}