#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>

#include <device_launch_parameters.h>

using namespace std;
#include <mma.h>

#include "include/RNG.cuh"
#include "include/Utils.cuh"
#include "include/TimeUtils.cuh"
#include "include/uint128.cuh"

#define num_tcu_warp_in_one_block 4
uint32_tt q32 = 0x3ffc0001;
__device__ uint32_tt q32_device = 0x3ffc0001;

uint64_tt q64 = 0x20000018e0001;
__device__ uint64_tt q64_device = 0x20000018e0001;

using namespace nvcuda;
__global__ void wmma_ker(uint8_tt *a, uint8_tt *b, uint8_tt* c, int* out, int N, int matrix_size) {
    // Declare the fragments
    int idx_in_warp = threadIdx.y;

    // printf("%d %d\n", threadIdx.x, threadIdx.y);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256, 16);
    wmma::load_matrix_sync(b_frag, b + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Store the output
    wmma::store_matrix_sync(out + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256, acc_frag, 16, wmma::mem_row_major);
}


using namespace nvcuda;
__global__ void wmma_u32_ker(uint32_tt *a, uint32_tt *b, uint32_tt* c, uint64_tt* out, int N, int matrix_size) {
    // Declare the fragments
    int idx_in_thread = threadIdx.x;
    int idx_in_warp = threadIdx.y;

    __shared__ uint8_tt shared_buffer[((2 * 4 + sizeof(int) + sizeof(uint64_tt)) * 16*16) * num_tcu_warp_in_one_block];
    uint8_tt* shared_in_warp = shared_buffer + ((2 * 4 + sizeof(int) + sizeof(uint64_tt)) * 16*16) * idx_in_warp;
    uint8_tt* shared_split_a_in_warp = shared_in_warp;
    uint8_tt* shared_split_b_in_warp = shared_in_warp + 16*16 * 4;


    // a warp -> 256 data
    uint32_tt* a_in_warp = a + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;
    uint32_tt* b_in_warp = b + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;
    uint64_tt* out_in_warp = out + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;

    int* shared_shift_in_warp = (int*)(shared_in_warp + (2 * 4 * 16*16));
    uint64_tt* shared_acc_in_warp = (uint64_tt*)(shared_in_warp + ((2 * 4 + sizeof(int)) * 16*16));
    
    // 8 data per warp
    #pragma unroll
    for(int i = 0; i < 8; i++){
        uint32_t cc = a_in_warp[idx_in_thread + i * 32];
        #pragma unroll
        for(int num_iter = 0; num_iter < 4; num_iter++){
            shared_split_a_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
        }
    }

    #pragma unroll
    for(int i = 0; i < 8; i++){
        uint32_t cc = b_in_warp[idx_in_thread + i * 32];
        #pragma unroll
        for(int num_iter = 0; num_iter < 4; num_iter++){
            shared_split_b_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
        }
        shared_acc_in_warp[idx_in_thread + i * 32] = 0;
    }
    __syncwarp();

    
    // // 8 data per warp
    // #pragma unroll
    // for(int num_iter = 0; num_iter < 4; num_iter++){
    //     #pragma unroll
    //     for(int i = 0; i < 8; i++){
    //         uint32_t cc = a_in_warp[idx_in_thread + i * 32];
    //         shared_split_a_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
    //     }

    //     #pragma unroll
    //     for(int i = 0; i < 8; i++){
    //         uint32_t cc = b_in_warp[idx_in_thread + i * 32];
    //         shared_split_b_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
    //     }
    // }
    // #pragma unroll
    // for(int i = 0; i < 8; i++){
    //     shared_acc_in_warp[idx_in_thread + i * 32] = 0;
    // }
    // __syncwarp();

    // 16 times u32 shift and add
    // #pragma unroll
    // for(int i = 0; i < 4; i++){
    //     #pragma unroll
    //     for(int j = 0; j < 4; j++){
    //         wmma::fill_fragment(acc_frag, 0);

    //         wmma::load_matrix_sync(a_frag, shared_split_a_in_warp + i*256, 16);
    //         wmma::load_matrix_sync(b_frag, shared_split_b_in_warp + j*256, 16);

    //         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    //         wmma::store_matrix_sync(shared_shift_in_warp, acc_frag, 16, wmma::mem_row_major);

    //         // if(idx_in_thread == 0) printf("%p\n", shared_acc_in_warp);
    //         #pragma unroll
    //         for(int iter = 0; iter < 8; iter++){
    //             shared_acc_in_warp[idx_in_thread + iter * 32] += uint64_tt(shared_shift_in_warp[idx_in_thread + iter * 32]) << (8 * (i + j));
    //         }
    //         __syncwarp();
    //     }
    // }

    // if(idx_in_thread == 1 && idx_in_warp == 1){
    //     for(int i = 0; i < 256; i++){
    //         printf("%x %x %x %x | %x\n", shared_split_a_in_warp[i+256*3], shared_split_a_in_warp[i+256*2], shared_split_a_in_warp[i+256*1], shared_split_a_in_warp[i], a_in_warp[i]);
    //     }
    // }

    // 7 times u32 shift and add
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

        // Initialize the output to zero
        wmma::fill_fragment(acc_frag, 0);

        #pragma unroll
        for(int j = 0; j < i+1; j++)
        {
            int idx1 = i - j;
            int idx2 = j;

            // Load the inputs
            wmma::load_matrix_sync(a_frag, shared_split_a_in_warp + idx1*256, 16);
            wmma::load_matrix_sync(b_frag, shared_split_b_in_warp + idx2*256, 16);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
   
        // Store the output
        wmma::store_matrix_sync(shared_shift_in_warp, acc_frag, 16, wmma::mem_row_major);

        // compute (ra[idx1] * rb[idx2]) << (8 * i)
        #pragma unroll
        for(int iter = 0; iter < 8; iter++){
            shared_acc_in_warp[idx_in_thread + iter * 32] += uint64_tt(shared_shift_in_warp[idx_in_thread + iter * 32]) << (8 * i);
        }
    }

    #pragma unroll
    for(int i = 4; i < 7; i++) 
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

        // Initialize the output to zero
        wmma::fill_fragment(acc_frag, 0);

        #pragma unroll
        for(int j = 4; j < i+1; j++) 
        {
            int idx1 = (7 - j);
            int idx2 = (j - i + 3);

            // Load the inputs
            wmma::load_matrix_sync(a_frag, shared_split_a_in_warp + idx1*256, 16);
            wmma::load_matrix_sync(b_frag, shared_split_b_in_warp + idx2*256, 16);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(shared_shift_in_warp, acc_frag, 16, wmma::mem_row_major);

        // compute (ra[idx1] * rb[idx2]) << (8 * (10 - i))
        #pragma unroll
        for(int iter = 0; iter < 8; iter++){
            shared_acc_in_warp[idx_in_thread + iter * 32] += uint64_tt(shared_shift_in_warp[idx_in_thread + iter * 32]) << (8 * (10 - i));
        }
    }

    #pragma unroll
    for(int iter = 0; iter < 8; iter++){
        out_in_warp[idx_in_thread + iter * 32] = shared_acc_in_warp[idx_in_thread + iter * 32];
    }
}

using namespace nvcuda;
__global__ void wmma_u64_ker(uint64_tt *a, uint64_tt *b, uint64_tt* c, uint128_tt* out, int N, int matrix_size) {
    // Declare the fragments
    int idx_in_thread = threadIdx.x;
    int idx_in_warp = threadIdx.y;

    // __shared__ uint8_tt shared_buffer[((2 * 4 + sizeof(uint32_tt) + sizeof(uint64_tt)) * 16*16) * num_tcu_warp_in_one_block];
    __shared__ uint8_tt shared_buffer[((2 * 8 + sizeof(int) + sizeof(uint128_tt)) * 16*16) * num_tcu_warp_in_one_block];

    uint8_tt* shared_in_warp = shared_buffer + ((2 * 8 + sizeof(int) + sizeof(uint128_tt)) * 16*16) * idx_in_warp;
    uint8_tt* shared_split_a_in_warp = shared_in_warp;
    uint8_tt* shared_split_b_in_warp = shared_in_warp + 16*16 * 8;

    uint64_tt* a_in_warp = a + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;
    uint64_tt* b_in_warp = b + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;
    uint128_tt* out_in_warp = out + (idx_in_warp + blockIdx.x * num_tcu_warp_in_one_block)*256;

    int* shared_shift_in_warp = (int*)(shared_in_warp + (2 * 8 * 16*16));
    uint128_tt* shared_acc_in_warp = (uint128_tt*)(shared_in_warp + ((2 * 8 + sizeof(int)) * 16*16));
    
    // 8 data per warp
    #pragma unroll
    for(int i = 0; i < 8; i++){
        uint64_t cc = a_in_warp[idx_in_thread + i * 32] % q64_device;
        #pragma unroll
        for(int num_iter = 0; num_iter < 8; num_iter++){
            shared_split_a_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
        }
    }

    #pragma unroll
    for(int i = 0; i < 8; i++){
        uint64_t cc = b_in_warp[idx_in_thread + i * 32] % q64_device;
        #pragma unroll
        for(int num_iter = 0; num_iter < 8; num_iter++){
            shared_split_b_in_warp[num_iter*256 + idx_in_thread + i * 32] = (cc >> (num_iter*8)) & 0xff;
        }
        shared_acc_in_warp[idx_in_thread + i * 32] = 0;
    }
    __syncwarp();

    // 15 times u32 shift and add
#pragma unroll
    for(int i = 0; i < 8; i++)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

        // Initialize the output to zero
        wmma::fill_fragment(acc_frag, 0);

    #pragma unroll
        for(int j = 0; j < i+1; j++)
        {
            int idx1 = i - j;
            int idx2 = j;

            // Load the inputs
            wmma::load_matrix_sync(a_frag, shared_split_a_in_warp + idx1*256, 16);
            wmma::load_matrix_sync(b_frag, shared_split_b_in_warp + idx2*256, 16);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
   
        // Store the output
        wmma::store_matrix_sync(shared_shift_in_warp, acc_frag, 16, wmma::mem_row_major);

        // compute (ra[idx1] * rb[idx2]) << (8 * i)
    #pragma unroll
        for(int iter = 0; iter < 8; iter++){
            int res_index = idx_in_thread + iter * 32;
            uint128_tt temp = uint128_tt(shared_shift_in_warp[res_index]) << (8 * i);
            add_uint128_uint128(shared_acc_in_warp[res_index], temp);
        }
    }

// #pragma unroll
    for(int i = 8; i < 15; i++) 
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

        // Initialize the output to zero
        wmma::fill_fragment(acc_frag, 0);

    #pragma unroll
        for(int j = 8; j < i+1; j++) 
        {
            int idx1 = 15 - j;
            int idx2 = j - i + 7;

            // Load the inputs
            wmma::load_matrix_sync(a_frag, shared_split_a_in_warp + idx1*256, 16);
            wmma::load_matrix_sync(b_frag, shared_split_b_in_warp + idx2*256, 16);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(shared_shift_in_warp, acc_frag, 16, wmma::mem_row_major);

        // compute (ra[idx1] * rb[idx2]) << (8 * (22 - i))
    #pragma unroll
        for(int iter = 0; iter < 8; iter++){
            int res_index = idx_in_thread + iter * 32;
            uint128_tt temp = uint128_tt(shared_shift_in_warp[res_index]) << (8 * (22 - i));
            add_uint128_uint128(shared_acc_in_warp[res_index], temp);
        }
    }

#pragma unroll
    for(int iter = 0; iter < 8; iter++){
        out_in_warp[idx_in_thread + iter * 32] = shared_acc_in_warp[idx_in_thread + iter * 32];
    }
}

void mma_host(uint32_tt *a_host, uint32_tt *b_host, uint32_tt* c_host, uint64_tt* out, int N, int matrix_size)
{
    // #pragma omp parallel for(32)
    for(int idx = 0; idx < N; idx+=256)
    {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                for (int k = 0; k < 16; k++) {
                    out[i + j * 16 + idx] += uint64_tt(a_host[i * 16 + k + idx]) * uint64_tt(b_host[k * 16 + j + idx]);
                }
                // out[i + j * 16 + idx*256] %= q32;
            }
        }
    }
}

__host__ void mma_host(uint64_tt *a_host, uint64_tt *b_host, uint64_tt* c_host, __uint128_t* out, int N, int matrix_size)
{
    // #pragma omp parallel for(32)
    for(int idx = 0; idx < N; idx+=256)
    {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                out[i + j * 16 + idx] = 0;
                for (int k = 0; k < 16; k++) {
                    out[i + j * 16 + idx] += static_cast<__uint128_t>(a_host[i * 16 + k + idx] % q64) * (b_host[k * 16 + j + idx] % q64);
                }
                // out[i + j * 16 + idx*256] %= q64;
            }
        }
    }
}

void test_u32()
{
    uint32_tt* in_a, *in_b, *in_c;
    uint32_tt* in_a_host, *in_b_host, *in_c_host;
    uint64_tt* out, *out_host;

    int N = 2*256;
    int matrix_size = 16 * 16;

    cudaMalloc(&in_a, N * matrix_size * sizeof(uint32_tt));
    cudaMalloc(&in_b, N * matrix_size * sizeof(uint32_tt));
    cudaMalloc(&in_c, N * matrix_size * sizeof(uint32_tt));

    in_a_host = new uint32_tt[N * matrix_size];
    in_b_host = new uint32_tt[N * matrix_size]; 
    in_c_host = new uint32_tt[N * matrix_size]; 

    out_host  = new uint64_tt[N * matrix_size];
    cudaMalloc(&out, N * matrix_size * sizeof(uint64_tt));

    RNG::generateRandom_device((uint8_tt*)in_a, sizeof(uint32_tt) * N * matrix_size);
    RNG::generateRandom_device((uint8_tt*)in_b, sizeof(uint32_tt) * N * matrix_size);
    RNG::generateRandom_device((uint8_tt*)in_c, sizeof(uint32_tt) * N * matrix_size);

    cudaMemcpy(in_a_host, in_a, N * matrix_size * sizeof(uint32_tt), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_b_host, in_b, N * matrix_size * sizeof(uint32_tt), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_c_host, in_c, N * matrix_size * sizeof(uint32_tt), cudaMemcpyDeviceToHost);

    mma_host(in_a_host, in_b_host, in_c_host, out_host, N, matrix_size);
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            printf("%llx, ", (out_host)[i + j * 16]);
        }
        cout << endl;
    }

    CUDATimer timer;
    float min_time = 1000;
    dim3 thread_dim(32, num_tcu_warp_in_one_block);
    for(int i = 0; i < 1000; i++){
        timer.start();
            wmma_u32_ker <<< N / num_tcu_warp_in_one_block, thread_dim >>> (in_a, in_b, in_c, out, N, matrix_size);
        min_time = min(min_time, timer.stop());
    }
    cudaDeviceSynchronize();
    print_device_array(out, 256, 1, "out");
    cout<<"time: "<<min_time<<"  us"<<endl;
}
void test_u64()
{
    uint64_tt* in_a, *in_b, *in_c;
    uint64_tt* in_a_host, *in_b_host, *in_c_host;
    uint128_tt* out;
    __uint128_t *out_host;

    int N = 2*256;
    int matrix_size = 16 * 16;

    cudaMalloc(&in_a, N * matrix_size * sizeof(uint64_tt));
    cudaMalloc(&in_b, N * matrix_size * sizeof(uint64_tt));
    cudaMalloc(&in_c, N * matrix_size * sizeof(uint64_tt));

    in_a_host = new uint64_tt[N * matrix_size];
    in_b_host = new uint64_tt[N * matrix_size]; 
    in_c_host = new uint64_tt[N * matrix_size]; 

    out_host  = new __uint128_t[N * matrix_size];
    cudaMalloc(&out, N * matrix_size * sizeof(uint128_tt));

    RNG::generateRandom_device((uint8_tt*)in_a, sizeof(uint64_tt) * N * matrix_size);
    RNG::generateRandom_device((uint8_tt*)in_b, sizeof(uint64_tt) * N * matrix_size);
    RNG::generateRandom_device((uint8_tt*)in_c, sizeof(uint64_tt) * N * matrix_size);

    cudaMemcpy(in_a_host, in_a, N * matrix_size * sizeof(uint64_tt), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_b_host, in_b, N * matrix_size * sizeof(uint64_tt), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_c_host, in_c, N * matrix_size * sizeof(uint64_tt), cudaMemcpyDeviceToHost);

    mma_host(in_a_host, in_b_host, in_c_host, out_host, N, matrix_size);
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            uint64_t data1 = (out_host)[i + j * 16] >> 64;
            uint64_t data2 = static_cast<uint64_t>((out_host)[i + j * 16]);
            printf("(%llx,%llx) ", data1, data2);
        }
        cout << endl;
    }

    CUDATimer timer;
    float min_time = 1000;
    dim3 thread_dim(32, num_tcu_warp_in_one_block);
    for(int i = 0; i < 100; i++){
        timer.start();
            wmma_u64_ker <<< N / num_tcu_warp_in_one_block, thread_dim >>> (in_a, in_b, in_c, out, N, matrix_size);
        min_time = min(min_time, timer.stop());
    }
    cudaDeviceSynchronize();
    print_device_array(out, 256, 1, "out");
    cout<<"time: "<<min_time<<"  us"<<endl;
}

int main(int argc, char* argv[])
{
    test_u32();

    test_u64();

    return 0;
}