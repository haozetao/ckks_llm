#pragma once
#include <math.h>
#include "uint128.cuh"
#include "RNG.cuh"
#include "Sampler_impl.cuh"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"


#define big_block 256

class Sampler
{
public:

    static void uniformSampler_xq(uint8_tt* in, uint64_tt* out, uint32_tt n, int idx_out, int idx_mod, int mod_num)
    {
        int convert_block_amount = mod_num * n / big_block;

        convert_uniform_xq <<<convert_block_amount, big_block >>> (in, out, n, idx_out, idx_mod);
    }

    static void gaussianSampler_xq(uint8_tt* in, uint64_tt* out, uint32_tt n, int idx_out, int idx_mod, int mod_num)
    {
        int convert_block_amount =  n / big_block;

        convert_gaussian_xq <<<convert_block_amount, big_block >>> (in, out, n, idx_out, idx_mod, mod_num);
    }

    static void HWTSampler(uint8_tt* in, uint64_tt* out, int logN, int h, int idx_out, int idx_mod, int mod_num)
    {
        convert_HWT <<<mod_num, h >>> (in, out, logN, h, idx_out, idx_mod);
    }

    static void ZOSampler(uint8_tt* in, uint64_tt* out, uint32_tt n, double probability, int idx_out, int idx_mod, int mod_num)
    {
        int convert_block_amount = n / big_block;

        convert_ZO <<<convert_block_amount, big_block >>> (in, out, n, probability, idx_out, idx_mod, mod_num);
    }

    static void binarySampler(uint8_tt* in, uint64_tt* out, uint32_tt n, int idx_out, int idx_mod, int mod_num)
    {
        int convert_block_amount = n / big_block;

        convert_binary <<<convert_block_amount, big_block >>> (in, out, idx_out, idx_mod);
    }
};