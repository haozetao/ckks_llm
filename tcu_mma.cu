#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace std;

#include "include/RNG.cuh"
#include "include/Utils.cuh"
#include "include/TimeUtils.cuh"
#include "include/uint128.cuh"


using namespace nvcuda;
__global__ void wmma_ker(uint8_tt *a, uint8_tt *b, uint8_tt* c, int* out, int N, int matrix_size) {
    // Declare the fragments
    int idx_in_warp = threadIdx.y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + (idx_in_warp + blockIdx.x * 4)*256, 16);
    wmma::load_matrix_sync(b_frag, b + (idx_in_warp + blockIdx.x * 4)*256, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Store the output
    wmma::store_matrix_sync(out + (idx_in_warp + blockIdx.x * 4)*256, acc_frag, 16, wmma::mem_row_major);
}

__global__ void wmma_ptx_ker(uint8_tt *a, uint8_tt *b, uint8_tt* c, int* out, int N, int matrix_size) {
    // Declare the fragments
    int idx_in_warp = threadIdx.y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_tt, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_tt, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + (idx_in_warp + blockIdx.x * 4)*256, 16);
    wmma::load_matrix_sync(b_frag, b + (idx_in_warp + blockIdx.x * 4)*256, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Store the output
    wmma::store_matrix_sync(out + (idx_in_warp + blockIdx.x * 4)*256, acc_frag, 16, wmma::mem_row_major);
}


void mma_host(uint8_tt *a_host, uint8_tt *b_host, uint8_tt* c_host, int* out, int N, int matrix_size)
{
    // #pragma omp parallel for(32)
    for(int idx = 0; idx < N; idx++)
    {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                for (int k = 0; k < 16; k++) {
                    out[i + j * 16 + idx*256] += int(a_host[i * 16 + k + idx*256]) * b_host[k * 16 + j + idx*256];
                }
                out[i + j * 16 + idx*256] += c_host[i + j * 16 + idx*256];
            }
        }
    }
}

int main(int argc, char* argv[])
{

    uint8_tt* in_a, *in_b, *in_c;
    uint8_tt* in_a_host, *in_b_host, *in_c_host;
    int* out, *out_host;

    int N = 1<<16;
    int matrix_size = 16 * 16;

    cudaMalloc(&in_a, N * matrix_size * sizeof(char));
    cudaMalloc(&in_b, N * matrix_size * sizeof(char));
    cudaMalloc(&in_c, N * matrix_size * sizeof(char));

    in_a_host = new uint8_tt[N * matrix_size];
    in_b_host = new uint8_tt[N * matrix_size]; 
    in_c_host = new uint8_tt[N * matrix_size]; 
    out_host  = new int[N * matrix_size];

    cudaMalloc(&out, N * matrix_size * sizeof(int));

    RNG::generateRandom_device(in_a, N * matrix_size);
    RNG::generateRandom_device(in_b, N * matrix_size);
    RNG::generateRandom_device(in_c, N * matrix_size);

    cudaMemcpy(in_a_host, in_a, N * matrix_size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_b_host, in_b, N * matrix_size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(in_c_host, in_c, N * matrix_size * sizeof(char), cudaMemcpyDeviceToHost);

    mma_host(in_a_host, in_b_host, in_c_host, out_host, N, matrix_size);
    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            // printf("%d, ", in_a_host[i + j * 16]);
            printf("%d, ", out_host[i + j * 16]);
        }
        cout << endl;
    }

    CUDATimer timer;

    timer.start();
    for(int i = 0; i < 1000; i++)
        wmma_ker <<< N / 4, dim3(32, 4) >>> (in_a, in_b, in_c, out, N, matrix_size);
    cout << "average tcu cost: " << timer.stop() / 1000 << endl;

    print_device_array(out, 256, 1, "out");


    for(int i = 0; i < 1000; i++)
        wmma_ptx_ker <<< N / 4, 32*4 >>> (in_a, in_b, in_c, out, N, matrix_size);
    cout << "average tcu ptx cost: " << timer.stop() / 1000 << endl;

    print_device_array(out, 256, 1, "out");

    return 0;
}