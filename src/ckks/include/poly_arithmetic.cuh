#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.cuh"
#include "ntt_60bit.cuh"

#define poly_block 128
#define POLY_MAX_THREADS 1024
#define small_block 128
#define POLY_MIN_BLOCKS 1

__global__ 
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void barrett_batch_kernel(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    register uint32_tt index = blockIdx.y;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    a[i + idx_a * n]=rc.low;
}

__host__ void barrett_batch_device(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_batch_kernel <<< barrett_dim, poly_block >>> (a, b, n, idx_a, idx_b, idx_mod, mod_num);
}

__global__ 
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void barrett_2batch_kernel(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int q_num)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_pq * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);
    a[i + idx_a * n]=rc.low;

    ra = a[i + idx_a * n + q_num*n];
    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    a[i + idx_a * n + q_num*n]=rc.low;
}

__host__ void barrett_2batch_device(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int q_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_2batch_kernel <<< barrett_dim, poly_block >>> (a, b, n, idx_a, idx_b, idx_mod, mod_num, q_num);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void barrett_batch_3param_kernel(uint64_tt c[], uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    register uint32_tt index = blockIdx.y;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    c[i + idx_c * n]=rc.low;
}

__host__ void barrett_batch_3param_device(uint64_tt c[], uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_batch_3param_kernel <<< barrett_dim, poly_block >>> (c, a, b, n, idx_c, idx_a, idx_b, idx_mod, mod_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_add_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_poly = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n + idx_in_poly * blockDim.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[i + idx_a * n] + device_b[i + idx_b * n];
    csub_q(ra, q);
    device_a[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + (idx_in_q + idx_in_cipher * q_num) * n;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];
    register uint64_tt ra = device_a[i] + device_b[i];
    csub_q(ra, q);
    device_a[i] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_T_batch_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int t_num)
{
    register uint32_tt idx_in_mod = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + (idx_in_mod + idx_in_cipher * t_num) * n;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_mod];
    register uint64_tt ra = device_a[i] + device_b[i];
    csub_q(ra, q);
    device_a[i] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_axbx_batch_device_kernel(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_q * n;
    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];

    register uint64_tt ra = cipher_device[i] + ax_device[i];
    csub_q(ra, q);
    cipher_device[i] = ra;

    ra = cipher_device[i + q_num*n] + bx_device[i];
    csub_q(ra, q);
    cipher_device[i + q_num*n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[i + idx_a * n] + device_b[i + idx_b * n];
    csub_q(ra, q);
    device_c[i + idx_c * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + (idx_in_q + idx_in_cipher * q_num) * n;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];
    register uint64_tt ra = device_a[i] + device_b[i];
    csub_q(ra, q);
    device_c[i] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_a, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_a) * n + idx_in_poly] + add_const_real_buffer[index];
    csub_q(ra, q);
    device_a[(index + idx_a) * n + idx_in_poly] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void cipher_negate_batch_device_kernel(uint64_tt* device_a, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = q - ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void cipher_negate_3param_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    device_b[(index + idx_in_cipher*q_num) * n + idx_in_poly] = q - ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_complex_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* add_const_buffer, uint32_tt n, uint64_tt* psi_powers, uint64_tt* psi_powers_shoup, int idx_a, int L, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_mod + index], pqt_mu_cons_low[idx_mod + index]};
    register uint64_tt ra = device_a[(index + idx_a) * n + idx_in_poly];
    register uint64_tt Nth_root = psi_powers[index * n + 1];
    register uint64_tt Nth_root_shoup = psi_powers_shoup[index * n + 1];

    register uint64_tt temp = mulMod_shoup(add_const_buffer[index + (L+1)], Nth_root, Nth_root_shoup, q);

    if(idx_in_poly < (n >> 1))
        ra = ra + add_const_buffer[index] + temp;
    else
        ra = ra + add_const_buffer[index] + q - temp;
    csub_q(ra, q);
    device_a[(index + idx_a) * n + idx_in_poly] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* const_real, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    register uint64_tt rb = const_real[index];
    register uint64_tt rb_shoup = const_real[index + q_num*2];

    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = mulMod_shoup(ra, rb, rb_shoup, q);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_const_batch_andAdd_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, uint64_tt target_scale, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt rb = device_b[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    register uint64_tt rc = const_real[index];
    register uint64_tt rc_shoup = const_real[index + q_num*2];

    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = mulMod_shoup(rb, rc, rc_shoup, q);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_sub_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_a[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_sub2_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_b[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_sub_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_c[i + idx_c * n] = ra;
}

// a = a + b
__host__ void poly_add_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// a = a + b
__host__ void cipher_add_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block , mod_num, 2);
    cipher_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_mod, q_num);
}

// a = a + b
__host__ void cipher_add_T_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int mod_num, int t_num, int blockNum)
{
    dim3 add_dim(n / poly_block , mod_num, blockNum);
    cipher_add_T_batch_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_mod, t_num);
}

// a = a + b
__host__ void cipher_add_axbx_batch_device(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block, mod_num);
    cipher_add_axbx_batch_device_kernel<<< add_dim, poly_block >>>(cipher_device, ax_device, bx_device, n, idx_mod, q_num);
}

// c = a + b
__host__ void poly_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_3param_batch_device_kernel<<< add_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c, idx_a, idx_b, idx_mod);
}

// a = a + b
__host__ void cipher_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block , mod_num, 2);
    cipher_add_3param_batch_device_kernel<<< add_dim, poly_block >>>(device_c, device_a, device_b, n, idx_mod, q_num);
}

__host__ void poly_add_real_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_a, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_const_batch_device_kernel<<< add_dim, poly_block >>>(device_a, add_const_real_buffer, n, idx_a, idx_mod);
}

__host__ void cipher_negate_batch_device(uint64_tt* device_a, uint32_tt n, int q_num, int idx_mod, int mod_num)
{
    dim3 negate_dim(n / poly_block , mod_num, 2);
    cipher_negate_batch_device_kernel<<< negate_dim, poly_block >>>(device_a, n, q_num, idx_mod);
}

__host__ void cipher_negate_3param_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int q_num, int idx_mod, int mod_num)
{
    dim3 negate_dim(n / poly_block , mod_num, 2);
    cipher_negate_3param_batch_device_kernel<<< negate_dim, poly_block >>>(device_a, device_b, n, q_num, idx_mod);
}

// a = a - b
__host__ void poly_sub_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub_batch_device_kernel<<< sub_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// b = a - b
__host__ void poly_sub2_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub2_batch_device_kernel<<< sub_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// c = a - b
__host__ void poly_sub_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub_3param_batch_device_kernel<<< sub_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c, idx_a, idx_b, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_add_axbx_double_add_cnst_batch_kernel(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_q * n;
    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];

    register uint64_tt ra = cipher_device[i] + ax_device[i];
    // csub_q(ra, q);
    // ra += ra;
    // csub_q(ra, q);
    ra += ra;
    ra %= q;
    cipher_device[i] = ra;

    ra = cipher_device[i + q_num*n] + bx_device[i];
    // csub_q(ra, q);
    // ra += ra;
    // csub_q(ra, q);
    // ra += add_const_real_buffer[idx_in_q];
    // csub_q(ra, q);
    ra += ra + add_const_real_buffer[idx_in_q];
    ra %= q;
    cipher_device[i + q_num*n] = ra;
}

__host__ void poly_add_axbx_double_add_cnst_batch_device(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 this_block(n / poly_block, mod_num);
    poly_add_axbx_double_add_cnst_batch_kernel <<< this_block, poly_block >>> (cipher_device, ax_device, bx_device, add_const_real_buffer, n, idx_mod, q_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sxsx_mul_P_3param_kernel(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);
    register uint64_tt ra = device_a[i + idx_a * n];

    register uint128_tt rc;

    mul64(ra, Pmodqt_cons[index + idx_mod - K], rc);
    singleBarrett_new(rc, q, mu);

    device_c[i + idx_c * n] = rc.low;
}

__host__ void sxsx_mul_P_3param(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int mod_num)
{
    dim3 mul_dim(n / poly_block , mod_num);
    sxsx_mul_P_3param_kernel <<< mul_dim, poly_block >>> (device_c, device_a, n, idx_c, idx_a, idx_mod, K);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_kernel(uint64_tt* device_a, uint64_tt* device_b, uint64_tt*rotGroup, uint32_tt n, int logn, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
	register long pow = rotGroup[rot_num];
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq + idx_in_cipher * q_num) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq + idx_in_cipher * q_num) * n;
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;

	int	mask = n - 1;
    int indexRaw = global_tid * pow;
    int index = indexRaw & mask;
    int tmp = (indexRaw >> logn) & 1;
    
    ai[index] = bi[global_tid]*(tmp^1) | (pqt_cons[idx_in_pq + p_num] - bi[global_tid])*tmp;
}

__host__ void sk_and_poly_LeftRot_double(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
{
    dim3 leftrot_dim(n / poly_block , mod_num, 2);
    int logn = log2(n);
    sk_and_poly_LeftRot_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, logn, p_num, q_num, rot_num, idx_a, idx_b);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_ntt_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt logNthRoot, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
	register long pow = rotGroup[(n>>1) - rot_num];
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq + idx_in_cipher * q_num) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq + idx_in_cipher * q_num) * n;
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;

	int mask = 2 * n - 1;
    int tmp1 = 2*bitReverse(global_tid, logNthRoot) + 1;
	tmp1 = ((pow * tmp1 & mask) - 1) >> 1;
    tmp1 = bitReverse(tmp1, logNthRoot);

    ai[tmp1] = bi[global_tid];
}

__host__ void sk_and_poly_LeftRot_ntt_double(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
{
    dim3 leftrot_dim(n / poly_block, mod_num, 2);
    int logn = log2(n);
    sk_and_poly_LeftRot_ntt_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, logn, p_num, q_num, rot_num, idx_a, idx_b);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_inv_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt logNthRoot, uint32_tt K, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt index = blockIdx.y;
	register long pow = rotGroup[rot_num];
	register uint64_tt* ai = device_a + idx_a * n + (index * n);
	register uint64_tt* bi = device_b + idx_b * n + (index * n);
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;

    int mask = 2 * n - 1;
    int tmp1 = 2*bitReverse(global_tid, logNthRoot) + 1;
	int	tmp2 = ((pow * tmp1 & mask) - 1) >> 1;
    tmp2 = bitReverse(tmp2, logNthRoot);

    ai[tmp2] = bi[global_tid];
}

__host__ void sk_and_poly_LeftRot_inv(uint64_tt* device_a, uint64_tt* device_b, uint64_tt*rotGroup, uint32_tt n, uint32_tt K, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
{
    dim3 leftrot_dim(n / poly_block , mod_num);
    int logn = log2(n);
    sk_and_poly_LeftRot_inv_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, logn, K, rot_num, idx_a, idx_b);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_conjugate_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, uint32_tt q_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq) * n;

	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	ai[global_tid] = bi[n - 1 - global_tid];
}

__host__ void sk_and_poly_conjugate(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, uint32_tt q_num, int idx_a, int idx_b, int mod_num)
{ 
    dim3 conjugate_dim(n / poly_block , mod_num);
    sk_and_poly_conjugate_kernel <<< conjugate_dim, poly_block >>> (device_a, device_b, n, q_num, idx_a, idx_b);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS,
    POLY_MIN_BLOCKS)
void divByiAndEqual_kernel(uint64_tt* device_a, uint32_tt n, uint32_tt q_num, int idx_mod, uint64_tt psi_powers[])
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
	
    register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
    register uint128_tt q_mu = {pqt_mu_cons_high[idx_mod + idx_in_pq], pqt_mu_cons_low[idx_mod + idx_in_pq]};

    register uint64_tt ra = device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly];
    register uint64_tt rb;
    if(idx_in_poly < (n>>1))
        // 4throot of Zq
        rb = q - psi_powers[(idx_in_pq) * n + 1];
    else
        // -4throot of Zq
        rb = psi_powers[(idx_in_pq) * n + 1];

    register uint128_tt temp;
    mul64(ra, rb, temp);
    singleBarrett_new(temp, q, q_mu);

    device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly] = temp.low;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS,
    POLY_MIN_BLOCKS)
void mulByiAndEqual_kernel(uint64_tt* device_a, uint32_tt n, uint32_tt q_num, int idx_mod, uint64_tt psi_powers[])
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
	
    register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
    register uint128_tt q_mu = {pqt_mu_cons_high[idx_mod + idx_in_pq], pqt_mu_cons_low[idx_mod + idx_in_pq]};

    register uint64_tt ra = device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly];
    register uint64_tt rb;
    if(idx_in_poly >= (n>>1))
        // 4throot of Zq
        rb = q - psi_powers[(idx_in_pq) * n + 1];
    else
        // -4throot of Zq
        rb = psi_powers[(idx_in_pq) * n + 1];

    register uint128_tt temp;
    mul64(ra, rb, temp);
    singleBarrett_new(temp, q, q_mu);

    device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly] = temp.low;
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void compute_c0c1c2_kernel(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a1, uint64_tt* a2, uint64_tt* b1, uint64_tt* b2, int n, int idx_poly, int idx_mod)
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register uint64_tt ra1 = a1[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt ra2 = a2[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb1 = b1[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb2 = b2[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]};

    register uint128_tt t1, t2, t3;
    mul64(ra1, ra2, t1);    // a1a2
    mul64(rb1, rb2, t2);    // b1b2

    // a1+b1
    ra1 = ra1 + rb1;
    // a2+b2
    ra2 = ra2 + rb2;
    mul64(ra1, ra2, t3);
    // t3 = t3 - t1 - t2;      // a1b2 + a2b1
    sub_uint128_uint128(t3, t1);
    sub_uint128_uint128(t3, t2);

    singleBarrett_new(t1, q, mu_q);
    singleBarrett_new(t2, q, mu_q);
    singleBarrett_new(t3, q, mu_q);

    axax_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t1.low;
    bxbx_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t2.low;
    a1b2a2b1_mul[idx_in_poly + (idx_in_pq + idx_poly) * n] = t3.low;
}

__host__ void compute_c0c1c2(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a1, uint64_tt* a2, uint64_tt* b1, uint64_tt* b2, int n, int idx_poly, int idx_mod, int mod_num)
{
    dim3 compute_c0c1c2_dim(n / poly_block, mod_num);
    compute_c0c1c2_kernel <<< compute_c0c1c2_dim, poly_block >>> (a1b2a2b1_mul, axax_mul, bxbx_mul, a1, a2, b1, b2, n, idx_poly, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void compute_c0c1c2_square_kernel(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a, uint64_tt* b, int n, int idx_poly, int idx_mod)
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register uint64_tt ra = a[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb = b[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]};

    register uint128_tt t1, t2, t3;
    mul64(ra, ra, t1);    // aa
    mul64(rb, rb, t2);    // bb
    mul64(ra, 2*rb, t3);  // 2ab

    singleBarrett_new(t1, q, mu_q);
    singleBarrett_new(t2, q, mu_q);
    singleBarrett_new(t3, q, mu_q);

    axax_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t1.low;
    bxbx_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t2.low;
    a1b2a2b1_mul[idx_in_poly + (idx_in_pq + idx_poly) * n] = t3.low;
}

__host__ void compute_c0c1c2_square(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a, uint64_tt* b, int n, int idx_poly, int idx_mod, int mod_num)
{
    dim3 compute_c0c1c2_dim(n / poly_block, mod_num);
    compute_c0c1c2_square_kernel <<< compute_c0c1c2_dim, poly_block >>> (a1b2a2b1_mul, axax_mul, bxbx_mul, a, b, n, idx_poly, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_mul_P_batch_kernel(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int L)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]);
    register uint64_tt ra = device_a[idx_in_poly + (idx_in_pq + idx_a) * n + idx_in_cipher * (L+1) * n];

    register uint128_tt rc;

    mul64(ra, Pmodqt_cons[idx_in_pq], rc);
    singleBarrett_new(rc, q, mu);

    device_c[idx_in_poly + (idx_in_pq + idx_c) * n + blockIdx.z * (K+L+1) * n] = rc.low;
}

__host__ void cipher_mul_P_batch(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int L, int mod_num, int batch_size)
{
    dim3 mul_dim(n / poly_block, mod_num, batch_size);
    cudaMemsetAsync(device_c, 0, sizeof(uint64_tt) * n * K);
    cudaMemsetAsync(device_c + (K+L+1)*n, 0, sizeof(uint64_tt) * n * K);
    cipher_mul_P_batch_kernel <<< mul_dim, poly_block >>> (device_c, device_a, n, idx_c, idx_a, idx_mod, K, L);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void bufferT_mul_P_batch_kernel(uint64_tt* device_a, uint32_tt n, int idx_mod, int K, int L, int t_num, int Qj_blockNum, int batch_size)
{
    register uint32_tt idx_in_T = blockIdx.y;
    register uint32_tt idx_in_block = blockIdx.z;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt t = pqt_cons[idx_in_T + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[idx_in_T + idx_mod], pqt_mu_cons_low[idx_in_T + idx_mod]);

    for(int i = 0; i < batch_size; i++)
    {
        int idx_io = idx_in_poly + idx_in_T * n + idx_in_block * t_num * n + i*t_num*Qj_blockNum*n;
        register uint64_tt ra = device_a[idx_io];

        mulMod_shoup(ra, Pmodqt_cons[L+1 + idx_in_T],Pmodqt_cons[L+1 + idx_in_T], t);
        device_a[idx_io] = ra;
    }
}

__host__ void bufferT_mul_P_batch(uint64_tt* device_a, uint32_tt n, int idx_mod, int K, int L, int t_num, int block_size, int Qj_blockNum, int batch_size)
{
    dim3 mul_dim(n / poly_block, t_num, block_size);
    bufferT_mul_P_batch_kernel <<< mul_dim, poly_block >>> (device_a, n, idx_mod, K, L, t_num, Qj_blockNum, batch_size);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int poly_num)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = device_a[i + idx_a * n + idx_in_cipher * poly_num * n];
    register uint64_tt rb = device_b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    device_c[i + idx_c * n + idx_in_cipher * poly_num * n]=rc.low;
}

__host__ void poly_add_batch_device_many_poly(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num)
{
    dim3 add_dim(n / poly_block , mod_num, poly_num);
    poly_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

__host__ void poly_mul_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num)
{
    dim3 mul_dim(n / poly_block , mod_num, 2);
    poly_mul_3param_batch_device_kernel<<< mul_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c,idx_a, idx_b, idx_mod, poly_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int poly_num, int block_size)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;
    register uint64_tt rb = device_b[i + idx_b * n];

    for(int k = 0; k < block_size; k++){
        register uint64_tt ra = device_a[i + idx_a * n + idx_in_cipher * poly_num * n + k * blockDim.y * n];
        register uint128_tt rc;
        mul64(ra, rb, rc);
        singleBarrett_new(rc, q, mu);
        register uint64_tt rc_add = device_c[i + idx_c * n + idx_in_cipher * poly_num * n + k * blockDim.y * n];
        device_c[i + idx_c * n + idx_in_cipher * poly_num * n + k * blockDim.y * n] = rc.low + rc_add;
    }

}

//c=a*b+c
__host__ void poly_mul_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num, int block_size)
{
    dim3 mul_dim(n / poly_block , mod_num, 2);
    poly_mul_add_3param_batch_device_kernel<<< mul_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c,idx_a, idx_b, idx_mod, poly_num, block_size);
}
