#pragma once

#include "../uint128.cuh"

#define ringSwitch_block 128
// 1*N -> k*N1
__global__ void ringDown_kernel(uint64_tt* bigRing_polys, uint64_tt* smallRing_polys, int N, int N1, int mlwe_rank, int start_mod_idx)
{
    int global_idx = blockIdx.x * ringSwitch_block + threadIdx.x;
    int idx_mod = blockIdx.y;

    uint64_tt q = pqt_cons[start_mod_idx + idx_mod];

    // each thread moves 1 data from big ring to small ring
    uint64_tt* smallRing_polys_this_mod = smallRing_polys + mlwe_rank*N1 * idx_mod;
    uint64_tt* bigRing_polys_this_mod = bigRing_polys + N * idx_mod;

    smallRing_polys_this_mod[global_idx] = bigRing_polys_this_mod[global_idx / N1 + global_idx % N1 * mlwe_rank];
}

// // 1*N -> k*N1
// __global__ void ringUp_kernel(uint64_tt* smallRing_polys, uint64_tt* bigRing_polys, int N, int N1, int mlwe_rank, int p_num)
// {
//     int global_idx = blockIdx.x * ringSwitch_block + threadIdx.x;
//     int idx_mod = blockIdx.y;

//     uint64_tt q = pqt_cons[p_num + idx_mod];

//     // each thread moves 1 data from big ring to small ring
//     uint64_tt* smallRing_polys_this_mod = smallRing_polys + idx_mod * mlwe_rank*N1;
//     uint64_tt* bigRing_polys_this_mod = bigRing_polys + idx_mod * N;

//     bigRing_polys_this_mod[global_idx] = smallRing_polys_this_mod[global_idx / N1 + global_idx % N1 * mlwe_rank];
// }

