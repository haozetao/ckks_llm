#pragma once

#include "Context_23.h"

#define external_product_block 256

__global__
__launch_bounds__(
    external_product_block, 
    POLY_MIN_BLOCKS) 
void external_product_T_kernel(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT,
        int n, int p_num, int q_num, int t_num, int Ri_blockNum, int cipher_blockNum, int dnum)
{
	register int idx_in_poly = blockIdx.z * external_product_block + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * t_num * n + blockIdx.y * n;
	register int idx_in_T = blockIdx.y;
	register int idx_in_block = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

    register uint128_tt acc = 0;
    register uint128_tt temp;

#pragma unroll
    for(int i = 0; i < cipher_blockNum; i++)
    {
        uint64_tt ra = cipher_modUp_QjtoT[(i*t_num + idx_in_T)*n + idx_in_poly];
        uint64_tt rb = swk_modUp_RitoT[(i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];
        // if(idx_in_poly == 0 && idx_in_T == 0 && idx_in_block == 1)printf("[%d]%llu, %llu\n", i, ra, rb);

        madc_uint64_uint64_uint128(ra, rb, acc);
    }
    singleBarrett_new(acc, t, t_mu);
    output[idx] = acc.low;

    acc = 0;

#pragma unroll
    for(int i = 0; i < cipher_blockNum; i++)
    {
        uint64_tt ra = cipher_modUp_QjtoT[(i*t_num + idx_in_T)*n + idx_in_poly];
        uint64_tt rb = swk_modUp_RitoT[Ri_blockNum*t_num*dnum*n + (i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];

        madc_uint64_uint64_uint128(ra, rb, acc);
    }
    singleBarrett_new(acc, t, t_mu);
    output[Ri_blockNum*t_num*n + idx] = acc.low;
}


__global__
__launch_bounds__(
    external_product_block, 
    POLY_MIN_BLOCKS) 
void external_product_T_kernel2(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT,
        int n, int p_num, int q_num, int t_num, int Ri_blockNum, int cipher_blockNum, int dnum)
{
	register int idx_in_poly = blockIdx.z * external_product_block + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * t_num * n + blockIdx.y * n;
	register int idx_in_T = blockIdx.y;
	register int idx_in_block = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

    register uint128_tt acc1 = 0, acc2 = 0;

#pragma unroll
    for(int i = 0; i < cipher_blockNum; i++)
    {
        uint64_tt ra = cipher_modUp_QjtoT[(i*t_num + idx_in_T)*n + idx_in_poly];
        uint64_tt rb1 = swk_modUp_RitoT[(i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];
        uint64_tt rb2 = swk_modUp_RitoT[Ri_blockNum*t_num*dnum*n + (i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];

        madc_uint64_uint64_uint128(ra, rb1, acc1);
        madc_uint64_uint64_uint128(ra, rb2, acc2);
    }
    singleBarrett_new(acc1, t, t_mu);
    singleBarrett_new(acc2, t, t_mu);
    output[idx] = acc1.low;
    output[Ri_blockNum*t_num*n + idx] = acc2.low;
}

void Context_23::external_product_T(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l)
{
    int cipher_blockNum = ceil(double(l+1) / p_num);
    int blockNum = ceil(double(p_num+l+1) / gamma);

    // print_device_array(cipher_modUp_QjtoT, N, t_num*cipher_blockNum, "cipher");
    // print_device_array(swk_modUp_RitoT, N, t_num*Ri_blockNum*dnum, "swk");
        dim3 external_product_dim(blockNum, t_num, N / external_product_block);
        external_product_T_kernel <<< external_product_dim, external_product_block >>>
            (output, cipher_modUp_QjtoT, swk_modUp_RitoT, N, p_num, q_num, t_num, Ri_blockNum, cipher_blockNum, dnum);
    // print_device_array(output, N, t_num*Ri_blockNum, "extPro1");
    // print_device_array(output + t_num*Ri_blockNum*N, N, t_num*Ri_blockNum, "extPro2");
}

__global__
__launch_bounds__(
    external_product_block, 
    POLY_MIN_BLOCKS) 
void external_product_T_kernel_swk_reuse(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT,
        int n, int p_num, int q_num, int t_num, int Ri_blockNum, int Qj_blockNum, int cipher_blockNum, int dnum, int batch_size)
{
	register int idx_in_poly = blockIdx.z * external_product_block + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * t_num * n + blockIdx.y * n;
	register int idx_in_T = blockIdx.y;
	register int idx_in_block = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

    register uint128_tt acc[8] = {0};

#pragma unroll
    for(int iter = 0; iter < batch_size; iter++)
    {
        for(int i = 0; i < cipher_blockNum; i++)
        {
            uint64_tt rb = swk_modUp_RitoT[(i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];
            uint64_tt ra = cipher_modUp_QjtoT[(i*t_num + idx_in_T)*n + idx_in_poly + iter*t_num*Qj_blockNum*n];

            madc_uint64_uint64_uint128(ra, rb, acc[iter]);
        }
    }
#pragma unroll
    for(int iter = 0; iter < batch_size; iter++)
    {
        singleBarrett_new(acc[iter], t, t_mu);
        output[Ri_blockNum*t_num*n*(iter*2) + idx] = acc[iter].low;
        acc[iter].low = 0;
    }

#pragma unroll
    for(int iter = 0; iter < batch_size; iter++)
    {
        for(int i = 0; i < cipher_blockNum; i++)
        {
            uint64_tt ra = cipher_modUp_QjtoT[(i*t_num + idx_in_T)*n + idx_in_poly + iter*t_num*Qj_blockNum*n];
            uint64_tt rb = swk_modUp_RitoT[Ri_blockNum*t_num*dnum*n + (i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + idx_in_poly];

            madc_uint64_uint64_uint128(ra, rb, acc[iter]);
        }
    }
#pragma unroll
    for(int iter = 0; iter < batch_size; iter++)
    {
        singleBarrett_new(acc[iter], t, t_mu);
        output[Ri_blockNum*t_num*n*(iter*2+1) + idx] = acc[iter].low;
    }
}

void Context_23::external_product_T_swk_reuse(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l, int batch_size)
{
    int cipher_blockNum = ceil(double(l+1) / p_num);
    int blockNum = ceil(double(p_num+l+1) / gamma);

    dim3 external_product_dim(blockNum, t_num, N / external_product_block);
    external_product_T_kernel_swk_reuse <<< external_product_dim, external_product_block >>>
        (output, cipher_modUp_QjtoT, swk_modUp_RitoT, N, p_num, q_num, t_num, Ri_blockNum, Qj_blockNum, cipher_blockNum, dnum, batch_size);
}


__global__
__launch_bounds__(
    external_product_block, 
    POLY_MIN_BLOCKS) 
void mult_PlaintextT_kernel(uint64_tt* cipher_modUp_QjtoT, uint64_tt* plain_T,
        int n, int p_num, int q_num, int t_num, int Ri_blockNum, int blockNum)
{
	register int idx_in_poly = blockIdx.y * external_product_block + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * n;
	register int idx_in_T = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

    register uint128_tt acc1 = 0;
    uint64_tt mx = plain_T[idx_in_poly + idx_in_T * n];
    uint64_tt ra;

    for(int i = 0; i < blockNum; i++)
    {
        ra = cipher_modUp_QjtoT[idx + i * t_num * n];

        mul64(ra, mx, acc1);
        singleBarrett_new(acc1, t, t_mu);
        cipher_modUp_QjtoT[idx + i * t_num * n] = acc1.low;
    }

    for(int i = 0; i < blockNum; i++)
    {
        ra = cipher_modUp_QjtoT[idx + i * t_num * n + t_num*Ri_blockNum*n];

        mul64(ra, mx, acc1);
        singleBarrett_new(acc1, t, t_mu);
        cipher_modUp_QjtoT[idx + i * t_num * n + t_num*Ri_blockNum*n] = acc1.low;
    }
}

void Context_23::mult_PlaintextT(uint64_tt* cipher_modUp_QjtoT, PlaintextT& plain_T, int l)
{
    int blockNum = ceil(double(p_num+l+1) / gamma);
    cout<<"mult_PlaintextT blockNum: "<<blockNum<<endl;

    dim3 external_product_dim(t_num, N / external_product_block);
    mult_PlaintextT_kernel <<< external_product_dim, external_product_block >>>
        (cipher_modUp_QjtoT, plain_T.mx_device, N, p_num, q_num, t_num, Ri_blockNum, blockNum);
}