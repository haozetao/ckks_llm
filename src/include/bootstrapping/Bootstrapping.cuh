#pragma once

#include "Bootstrapper.cuh"

#define modUpQ0toQL_block 1024
__global__ void modUpQ0toQL_kernel(uint64_tt* input, int n, int p_num, int q_num, int scale_up)
{
	register int idx_in_poly = blockIdx.x * modUpQ0toQL_block + threadIdx.x;
	register int idx_in_pq = blockIdx.y;
	register int idx_in_cipher = blockIdx.z;
	register uint64_tt qi = pqt_cons[p_num + idx_in_pq+1];
    register uint64_tt q0 = pqt_cons[p_num];
	register uint64_tt ra = input[idx_in_cipher*q_num*n + idx_in_poly];
    register int64_t temp = 0;

    if(ra > q0/2)
        temp = int64_t(ra) - q0 + (qi * scale_up);
    else
        temp = ra;
    temp %= qi;

    input[(idx_in_cipher*q_num + idx_in_pq+1)*n + idx_in_poly] = temp;
}

#define mulNdivslots_block 1024
__global__ void mulInvNdiv2slots_kernel(uint64_tt* input, int n, int p_num, int q_num, uint64_tt* inv_Ndiv2slots_device)
{
	register int idx_in_poly = blockIdx.x * mulNdivslots_block + threadIdx.x;
	register int idx_in_pq = blockIdx.y;
	register int idx_in_cipher = blockIdx.z;
    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
    register uint128_tt q_mu = {pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]};
	register uint64_tt ra = input[(idx_in_cipher*q_num + idx_in_pq)*n + idx_in_poly];
    register uint64_tt rb = inv_Ndiv2slots_device[idx_in_pq];
    register uint128_tt temp;
    mul64(ra, rb, temp);
    singleBarrett_new(temp, q, q_mu);
    input[(idx_in_cipher*q_num + idx_in_pq)*n + idx_in_poly] = temp.low;
}


// #define modUpQ0toQL_block 1024
// __global__ void mulNdivslots_kernel(uint64_tt* input, int n, int p_num, int q_num)
// {
// 	register int idx_in_poly = blockIdx.x * modUpQ0toQL_block + threadIdx.x;
// 	register int idx_in_pq = blockIdx.y;
// 	register int idx_in_cipher = blockIdx.z;
// 	register uint64_tt qi = pqt_cons[p_num + idx_in_pq+1];
// 	register uint64_tt ra = input[idx_in_cipher*q_num*n + idx_in_poly];
// 	csub_q(ra, qi);
// 	input[(idx_in_cipher*q_num + idx_in_pq+1)*n + idx_in_poly] = ra;
// }

// // compute cipher.coeff * slots/N
// __host__ void Bootstrapper::mulNdivslots(uint64_tt* input)
// {
// 	dim3 modUpQ0toQL_dim(N / modUpQ0toQL_block, L+1, 2);
// 	modUpQ0toQL_mulQ0_kernel <<< modUpQ0toQL_dim, modUpQ0toQL_block >>> (input, N, p_num, q_num);
// }
