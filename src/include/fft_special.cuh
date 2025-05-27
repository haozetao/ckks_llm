#pragma once

#include "cuda_runtime.h"
#include "cuComplex.h"
#include "device_launch_parameters.h"
#include "uint128.cuh"
#include "Utils.cuh"

__global__ void bitReverse_kernel(cuDoubleComplex* vals, int logslots, int thread) 
{	
	register int tid = (blockIdx.x * thread + threadIdx.x);
	long x = 0;
    x = bitReverse(tid, logslots);
    if (tid < x)
    {
        register cuDoubleComplex temp = vals[tid];
        vals[tid] = vals[x];
        vals[x] = temp;
    }
}

__host__ __forceinline__ void bitReverse_device(cuDoubleComplex* complexArray, int slots, int block, int thread)
{
	dim3 encode_dim(block);
	int logslots = log2(slots);
	bitReverse_kernel <<<encode_dim, thread, 0, 0 >>> (complexArray, logslots, thread);
}

template<uint32_tt l, uint32_tt n>
__global__ void fft_single(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, uint64_tt* rotGroup, long M)
{
    register int local_tid = threadIdx.x;
    extern __shared__ cuDoubleComplex cu_shared_array[];
#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        cu_shared_array[global_tid] = vals[global_tid + blockIdx.x * (n / l)];
    }
    __syncthreads();

    for (int length = (n / 2); length >= l; length /= 2)
    {
        register int step = (n / length) / 2;
        register int gap = (step << 3);          
    
    #pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num ++)//跨block计算
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;
            register int ksi_tid = (rotGroup[global_tid % step] % gap) * (M) / gap;
            register cuDoubleComplex ksiPow = ksiPows[ksi_tid];
            register cuDoubleComplex u = cu_shared_array[target_index]; 
	        register cuDoubleComplex v = cu_shared_array[target_index + step];
            v = cuCmul(v,ksiPow);
            cu_shared_array[target_index] = cuCadd(u, v);
	        cu_shared_array[target_index + step] =cuCsub(u, v);
        }
        __syncthreads();
    }
#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        vals[global_tid + blockIdx.x * (n / l)] = cu_shared_array[global_tid];
    }
}

template<uint32_tt l, uint32_tt n>
__global__ void fftInv_single(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, long precision, uint64_tt* rotGroup, long M)
{
    register int local_tid = threadIdx.x;
    extern __shared__ cuDoubleComplex cu_shared_array[];
#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        cu_shared_array[global_tid] = vals[global_tid + blockIdx.x * (n / l)];
    }
    __syncthreads();

    for (int length = l; length < n; length *= 2)
    {
        register int step =(n / length) / 2;
        register int gap = (step << 3);
        
    #pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = (local_tid + iteration_num * 1024);
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;
            register int ksi_tid = (gap - (rotGroup[global_tid % step] % gap)) * (M) / gap;
            register cuDoubleComplex ksiPow = ksiPows[ksi_tid];
            register cuDoubleComplex u = cuCadd(cu_shared_array[target_index], cu_shared_array[target_index + step]);
	        register cuDoubleComplex v = cuCsub(cu_shared_array[target_index], cu_shared_array[target_index + step]);
            v = cuCmul(v,ksiPow);
            cu_shared_array[target_index] = u; 
	        cu_shared_array[target_index + step] = v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        cu_shared_array[global_tid].x *= precision/slots;
        cu_shared_array[global_tid].y *= precision/slots;
        vals[global_tid + blockIdx.x * (n / l)] = cu_shared_array[global_tid];
    }
}

__global__ void fft_single_special(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, uint64_tt* rotGroup, long M)
{
    //uint32_tt index = blockIdx.y % slots;//一维即可

    register int local_tid = threadIdx.x;
    extern __shared__ cuDoubleComplex cu_shared_array[];
#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * (slots / 2);
        cu_shared_array[global_tid] = make_cuDoubleComplex(vals[global_tid].x, vals[global_tid].y);
    }
    __syncthreads();

    for (int length = (slots / 2); length >= 1; length /= 2)
    {
        register int step = (slots / length) / 2;
        register int gap = (step << 3);          
        register int global_tid = local_tid;
        register int psi_step = global_tid / step;
        register int target_index = psi_step * step * 2 + global_tid % step;
        register int ksi_tid = (rotGroup[global_tid % step] % gap) * (M) / gap;
        register cuDoubleComplex ksiPow = make_cuDoubleComplex(ksiPows[ksi_tid].x, ksiPows[ksi_tid].y);
        register cuDoubleComplex u = make_cuDoubleComplex(cu_shared_array[target_index].x, cu_shared_array[target_index].y); 
	    register cuDoubleComplex v = make_cuDoubleComplex(cu_shared_array[target_index + step].x, cu_shared_array[target_index + step].y);
        v = cuCmul(v,ksiPow);
        cu_shared_array[target_index] = cuCadd(u, v);
	    cu_shared_array[target_index + step] =cuCsub(u, v);
        
    }
    __syncthreads();
    
#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * (slots / 2);
        vals[global_tid] = make_cuDoubleComplex(cu_shared_array[global_tid].x, cu_shared_array[global_tid].y);
    }
}

__global__ void fftInv_single_special(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, long precision, uint64_tt* rotGroup, long M)
{
    //uint32_tt index = blockIdx.x % slots;
    register int local_tid = threadIdx.x;
    extern __shared__ cuDoubleComplex cu_shared_array[];
#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * (slots / 2);
        cu_shared_array[global_tid] = make_cuDoubleComplex(vals[global_tid].x, vals[global_tid].y);
    }
    __syncthreads();

    for (int length = 1; length < slots; length *= 2)
    {
        register int step =(slots / length) / 2;
        register int gap = (step << 3);
        register int global_tid = (local_tid);
        register int psi_step = global_tid / step;
        register int target_index = psi_step * step * 2 + global_tid % step;
        register int ksi_tid = (gap - (rotGroup[global_tid % step] % gap)) * (M) / gap;
        register cuDoubleComplex ksiPow = make_cuDoubleComplex(ksiPows[ksi_tid].x, ksiPows[ksi_tid].y);
        register cuDoubleComplex u = cuCadd(cu_shared_array[target_index], cu_shared_array[target_index + step]);
	    register cuDoubleComplex v = cuCsub(cu_shared_array[target_index], cu_shared_array[target_index + step]);
        v = cuCmul(v,ksiPow);
        cu_shared_array[target_index] = make_cuDoubleComplex(u.x, u.y); 
	    cu_shared_array[target_index + step] = make_cuDoubleComplex(v.x, v.y);
    }
    __syncthreads();

#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * (slots / 2);
        cu_shared_array[global_tid].x *= precision/slots;
        cu_shared_array[global_tid].y *= precision/slots;
        vals[global_tid] = make_cuDoubleComplex(cu_shared_array[global_tid].x, cu_shared_array[global_tid].y);
    }
}

template<uint32_tt l, uint32_tt n>
__global__ void fft(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, uint64_tt* rotGroup, long M)
{
    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / l) / 2;
    register int gap = (step << 3);
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;
    register int ksi_tid = (rotGroup[global_tid % step] % gap) * (M) / gap;
    register cuDoubleComplex ksiPow = ksiPows[ksi_tid];
    register cuDoubleComplex u = vals[target_index];
	register cuDoubleComplex v = vals[target_index + step];
    v = cuCmul(v,ksiPow);
    vals[target_index] = cuCadd(u, v);
	vals[target_index + step] = cuCsub(u, v);
}

template<uint32_tt l, uint32_tt n>
__global__ void fftInv(cuDoubleComplex* vals, cuDoubleComplex* ksiPows, long slots, uint64_tt* rotGroup, long M)
{
    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / l) / 2;
    register int gap =  (step << 3);
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;
    register int ksi_tid = (gap - (rotGroup[global_tid % step] % gap)) * (M) / gap;
    register cuDoubleComplex ksiPow = ksiPows[ksi_tid];
    register cuDoubleComplex u = cuCadd(vals[target_index], vals[target_index + step]);
	register cuDoubleComplex v = cuCsub(vals[target_index], vals[target_index + step]);
    v = cuCmul(v,ksiPow);
    vals[target_index] = u;
	vals[target_index + step] = v;
}

__host__ void fft_batch(cuDoubleComplex* vals, cuDoubleComplex* kisPows, long slots, uint64_tt* rotGroup, long M)
{
    if (slots == 32768)
    {   
        dim3 single_dim(16);
        dim3 multi_dim(slots / 1024 / 2);
        fft_single<16, 32768> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
        fft<8, 32768> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<4, 32768> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<2, 32768> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<1, 32768> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
    }
    else if (slots == 16384)
    {   
        dim3 single_dim(8);
        dim3 multi_dim(slots / 1024 / 2);
        fft_single<8, 16384> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
        fft<4, 16384> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<2, 16384> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<1, 16384> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
    }
    else if (slots == 8192)
    {   
        dim3 single_dim(4);
        dim3 multi_dim(slots / 1024 / 2);
        fft_single<4, 8192> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
        fft<2, 8192> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
        fft<1, 8192> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
    }
    else if (slots == 4096)
    {
        dim3 single_dim(2);
        dim3 multi_dim(slots / 1024 / 2);
        fft_single<2, 4096> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
        fft<1, 4096> << <multi_dim, 1024, 0, 0 >> > (vals, kisPows, slots, rotGroup, M);
    }
    else if (slots == 2048)
    {
        dim3 single_dim(1);
        dim3 multi_dim(slots / 1024 / 2);

        fft_single<1, 2048> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
    }
    else
    {
        dim3 single_dim(1);
        fft_single_special<< <single_dim, slots / 2, slots * sizeof(cuDoubleComplex), 0 >> > (vals, kisPows, slots, rotGroup, M);//1-1024
    }
}

__host__ void fftInv_batch(cuDoubleComplex* vals, cuDoubleComplex* kisPows, long slots, long precision, uint64_tt* rotGroup, long M)
{
    if (slots == 32768)
    {   
        dim3 single_dim(16);
        dim3 multi_dim(slots / 1024 / 2);
        fftInv<1, 32768> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<2, 32768> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<4, 32768> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<8, 32768> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        
        fftInv_single<16, 32768> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);//1024-1
    }
    else if (slots == 16384)
    {   
        dim3 single_dim(8);
        dim3 multi_dim(slots / 1024 / 2);
        fftInv<1, 16384> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<2, 16384> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<4, 16384> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv_single<8, 16384> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);//1024-1
    }
    else if (slots == 8192)
    {   
        dim3 single_dim(4);
        dim3 multi_dim(slots / 1024 / 2);
        fftInv<1, 8192> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv<2, 8192> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv_single<4, 8192> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);//1024-1
    }
    else if (slots == 4096)
    {   
        dim3 single_dim(2);
        dim3 multi_dim(slots / 1024 / 2);
        fftInv<1, 4096> << <multi_dim, 1024, 0, 0 >> >(vals, kisPows, slots, rotGroup, M);
        fftInv_single<2, 4096> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);//1024-1
    }
    else if (slots == 2048)
    {   
        dim3 single_dim(1);
        dim3 multi_dim(slots / 1024 / 2);
        fftInv_single<1, 2048> << <single_dim, 1024, 2048 * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);//1024-1
    }   
    else
    {
        dim3 single_dim(1);
        fftInv_single_special << <single_dim, slots / 2, slots * sizeof(cuDoubleComplex), 0 >> >(vals, kisPows, slots, precision, rotGroup, M);
    }                            
}