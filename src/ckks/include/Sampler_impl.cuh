#pragma once

#include "uint128.cuh"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define big_block 256
#define dstdev 3.2
#define dmean 0

// in : 64bit thread num = mod_num * n
__global__ void convert_uniform_xq(uint8_tt* in, uint64_tt* out, uint32_tt n, int idx_out, int idx_mod)
{
    int i = blockIdx.x * big_block + threadIdx.x;

    uint64_tt* inl = (uint64_tt*)in;
    register double d = (double)inl[i];

    d /= UINT64_MAX;

    d *= (double)(pqt_cons[i / n + idx_mod] - 1);

    out[i + idx_out * n] = (uint64_tt)d;
}

// in : 32bit thread num = n
__global__ void convert_gaussian_xq(uint8_tt* in, uint64_tt* out, uint32_tt n, int idx_out, int idx_mod, int mod_num)
{
    int i = blockIdx.x * big_block + threadIdx.x;

    float d = ((uint32_tt*)(in))[i % n];

    d /= 4294967295;

    if (d == 0)
        d += 1.192092896e-07F;
    else if (d == 1)
        d -= 1.192092896e-07F;

    d = normcdfinvf(d);

    d = d * (float)dstdev + dmean;

    if (d > 19.2)
    {
        d = 19.2;
    }
    else if (d < -19.2)
    {
        d = -19.2;
    }

    // if(d > 5){
    //     d = 5;
    // }
    // else if(d < -5){
    //     d = -5;
    // }

    int dd = (int)d;

#pragma unroll
    for(int t = 0; t < mod_num; t++)
    {
        if (dd < 0)
            out[i + idx_out * n + t * n] = pqt_cons[t + idx_mod] + dd;
        else
            out[i + idx_out * n + t * n] = dd;
    }
}

// error
// in : 32bit thread num = mod_num * h
__global__ void convert_HWT(uint8_tt* in, uint64_tt* out, int logN, int h, int idx_out, int idx_mod)
{
    register int i = blockIdx.x * h + threadIdx.x;
    register int idx_in_pq = i / h;
    register int idx_in_in = i % h;

    register uint8_tt* _8ptr = (uint8_tt*)(in + h * sizeof(uint32_tt) / sizeof(uint8_tt));
    register uint32_tt* _32ptr = (uint32_tt*)in;
    
    register float d = (float)_8ptr[idx_in_in];

    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];

    d /= (256.0f / 2);
    if(d >= 1)
        out[(_32ptr[idx_in_in]&((1<<logN) - 1)) + ((idx_in_pq + idx_mod) << logN) + (idx_out << logN)] = 1;
    else
        out[(_32ptr[idx_in_in]&((1<<logN) - 1)) + ((idx_in_pq + idx_mod) << logN) + (idx_out << logN)] = q - 1;
}

// in : 8bit thread num = n
__global__ void convert_ZO(uint8_tt* in, uint64_tt* out, uint32_tt n, double probability, int idx_out, int idx_mod, int mod_num)
{
    int i = blockIdx.x * big_block + threadIdx.x;
    
    register float d = (float)in[i];

#pragma unroll
    for(int t = 0; t < mod_num; t++)
    {
        register uint64_tt q = pqt_cons[t + idx_mod];

        if (d < 256.0f * probability)
            out[i + idx_out * n + t * n] = 0;
        else if (d < 256.0f * (1 + probability) / 2)
            out[i + idx_out * n + t * n] = 1;
        else
            out[i + idx_out * n + t * n] = q - 1;
    }
}

// in : 8bit thread num = n
__global__ void convert_binary(uint8_tt* in, uint64_tt* out, int idx_out, int idx_mod)
{
    int i = blockIdx.x * big_block + threadIdx.x;

    register float d = (float)in[i];

    d /= (256.0f / 2);

    if (d >= 1)
        out[i + idx_out] = 0;
    else
        out[i + idx_out] = 1;
}
