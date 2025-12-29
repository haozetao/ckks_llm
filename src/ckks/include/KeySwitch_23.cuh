#pragma once

#include "Scheme_23.h"
#include "Key_decomp.cuh"

Key_decomp *Scheme_23::addSWKey_23(SecretKey &secretKey, uint64_tt *s2, cudaStream_t stream)
{
    int N = context.N;
    int dnum = context.dnum;
    int gamma = context.gamma;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *swk = new Key_decomp(N, dnum, gamma, Ri_blockNum);

    return swk;
}

/**
 * generates key for multiplication (key is stored in keyMap)
 */
void Scheme_23::addMultKey_23(SecretKey &secretKey, cudaStream_t stream)
{
    if (rlk_23 != nullptr)
        return;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    rlk_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    barrett_batch_3param_device(sxsx, secretKey.sx_device, secretKey.sx_device, N, 0, K, K, K, L + 1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sxsx, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((rlk_23->bx_device) + i * N * t_num * Ri_blockNum, (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, secretKey.sx_device, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (rlk_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((rlk_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((rlk_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);

    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rlk_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (rlk_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rlk_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((rlk_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
}

/**
 * generates key for conjugation (key is stored in keyMap)
 */
void Scheme_23::addConjKey_23(SecretKey &secretKey, cudaStream_t steam)
{
    if (ConjKey_23 != nullptr)
        return;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;
    // ConjKey
    ConjKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);
    // sk_conj
    sk_and_poly_conjugate(sxsx, secretKey.sx_device, N, L+1, 0, K, L+1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sxsx, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        // Sampler::uniformSampler_xq(context.randomArray_conjk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, secretKey.sx_device, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((ConjKey_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((ConjKey_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((ConjKey_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
}

/**
 * generates key for left rotation <Hoisting Rotation> (key is stored in leftRotKeyMap)
 */
void Scheme_23::addLeftRotKey_23(SecretKey &secretkey, long rot_num, cudaStream_t stream)
{
    if (rotKey_vec_23[rot_num] != nullptr){
        return;
    }
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *rotKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    cudaMemcpy(sx_coeff, secretkey.sx_device, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
    sk_and_poly_LeftRot_inv(sxsx, sx_coeff, context.rotGroups_device, N, 0, rot_num, 0, 0, K+L+1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sx_coeff + N * K, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        // Sampler::uniformSampler_xq(context.randomArray_rotk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, sxsx, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((rotKey_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((rotKey_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((rotKey_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
    rotKey_vec_23[rot_num] = rotKey_23;
}

/**
 * generates key for left rotation (key is stored in leftRotKeyMap)
 */
void Scheme_23::addAutoKey_23(SecretKey &secretkey, int d, cudaStream_t stream)
{
    int N = context.N;
    if (d < 0 && d > log2(N))
    {
        throw invalid_argument("autoKey only for i in range(logn, logN)");
    }
    if (autoKey_vec_23[d] != nullptr)
        return;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *autoKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    // xxx add code here
    autoKey_vec_23[d] = autoKey_23;
}

void Scheme_23::mult_23(Ciphertext &cipher_res, Ciphertext &cipher1, Ciphertext &cipher2)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher_res.l = l;
    cipher_res.scale = cipher1.scale * cipher2.scale;

    compute_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_res.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher_res.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher_res.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
    // cudaMemset
}

void Scheme_23::multAndEqual_23(Ciphertext &cipher1, Ciphertext &cipher2)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.scale = cipher1.scale * cipher2.scale;

    compute_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l,2 );
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
}

void Scheme_23::multAndEqual_beforeIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_input, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.scale = cipher1.scale * cipher2.scale;

    compute_c0c1c2(axbx1_mul_batch, axax_mul, bxbx_mul_batch, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(IP_input, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(IP_input, 0, K + L + 1, cipher_blockNum, t_num, t_num);
}

void Scheme_23::multAndEqual_afterIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_output, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }

    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);

    context.FromNTTInplace_for_externalProduct(IP_output, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);
    context.modUpTtoPQl_23(modUp_TtoQj_buffer, IP_output, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul_batch, bxbx_mul_batch, N, K, l+1, L+1);
}

// Homomorphic Squaring
void Scheme_23::square(Ciphertext &cipher1, Ciphertext& cipher2)
{
    if (cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher2.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher2.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.l = cipher2.l;
    cipher1.scale = cipher2.scale * cipher2.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher2.ax_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
}

void Scheme_23::squareAndEqual(Ciphertext &cipher)
{
    if (cipher.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher.scale = cipher.scale * cipher.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher.ax_device, cipher.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
}

void Scheme_23::conjugate_23(Ciphertext& cipher_res, Ciphertext &cipher)
{
    if (ConjKey_23 == nullptr)
    {
        throw invalid_argument("conjKey_23 not exists");
    }

    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher_res.l = cipher.l;
    cipher_res.scale = cipher.scale;

    // conj---(b,a)
    sk_and_poly_conjugate(bxbx_mul, cipher.bx_device, N, L+1, 0, 0, l + 1);
    sk_and_poly_conjugate(axax_mul, cipher.ax_device, N, L+1, 0, 0, l + 1);
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding(no inverse NTT)
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, ConjKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_res.cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher_res.cipher_device, 0, K, 2, l + 1, L + 1);

    poly_add_batch_device(cipher_res.bx_device, bxbx_mul, N, 0, 0, K, l + 1);
}


void Scheme_23::conjugateAndEqual_23(Ciphertext &cipher)
{
    if (ConjKey_23 == nullptr)
    {
        throw invalid_argument("conjKey_23 not exists");
    }

    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // conj---(b,a)
    sk_and_poly_conjugate(bxbx_mul, cipher.bx_device, N, L+1, 0, 0, l + 1);
    sk_and_poly_conjugate(axax_mul, cipher.ax_device, N, L+1, 0, 0, l + 1);
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding(no inverse NTT)
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, ConjKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    poly_add_batch_device(cipher.bx_device, bxbx_mul, N, 0, 0, K, l + 1);
}

// Homomorphic Rotate <Hoisting Rotation>
void Scheme_23::leftRotateAndEqual_23(Ciphertext &cipher, long rotSlots)
{
    if (rotKey_vec_23[rotSlots] == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    Key_decomp *rotKey_23 = rotKey_vec_23[rotSlots];

    context.FromNTTInplace(cipher.ax_device, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding
    context.modUpQjtoT_23(modUp_QjtoT_temp, cipher.ax_device, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_temp_pool->cipher_device, modUp_TtoQj_buffer, l, 2);
    
    // poly_add_batch_device(cipher_temp_pool->bx_device, cipher.bx_device, N, 0, 0, K, l+1);
    // sk_and_poly_LeftRot_double(cipher.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
    
    // context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l+1, L+1);

    context.ToNTTInplace(cipher_temp_pool->cipher_device, 0, K, 2, l+1, L+1);
    poly_add_batch_device(cipher_temp_pool->bx_device, cipher.bx_device, N, 0, 0, K, l+1);

    sk_and_poly_LeftRot_ntt_double(cipher.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
}

void Scheme_23::leftRotate_23(Ciphertext& cipher_res, Ciphertext& cipher, long rotSlots)
{
    if (rotKey_vec_23[rotSlots] == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    Key_decomp *rotKey_23 = rotKey_vec_23[rotSlots];

    context.FromNTTInplace(cipher.ax_device, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding
    context.modUpQjtoT_23(modUp_QjtoT_temp, cipher.ax_device, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);
    context.ToNTTInplace(cipher.ax_device, 0, K, 1, l + 1, L + 1);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_temp_pool->cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher_temp_pool->cipher_device, 0, K, 2, l+1, L+1);
    poly_add_batch_device(cipher_temp_pool->bx_device, cipher.bx_device, N, 0, 0, K, l+1);

    sk_and_poly_LeftRot_ntt_double(cipher_res.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
}

void Scheme_23::leftRotateAddSelf_23(Ciphertext& cipher, long rotSlots)
{
    if (rotKey_vec_23[rotSlots] == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    Key_decomp *rotKey_23 = rotKey_vec_23[rotSlots];

    context.FromNTTInplace(cipher.ax_device, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding
    context.modUpQjtoT_23(modUp_QjtoT_temp, cipher.ax_device, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_temp_pool->cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher_temp_pool->cipher_device, 0, K, 2, l+1, L+1);
    poly_add_batch_device(cipher_temp_pool->bx_device, cipher.bx_device, N, 0, 0, K, l+1);

    context.ToNTTInplace(cipher.ax_device, 0, K, 1, l + 1, L + 1);
    sk_and_poly_LeftRot_Add_ntt_double(cipher.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
}

void Scheme_23::rightRotateAndEqual_23(Ciphertext &cipher, long rotSlots)
{
    long rotslots = context.Nh - (1 << rotSlots); // Convert to left shift
    leftRotateAndEqual_23(cipher, rotslots);
}

// f(X) -> f(X) + f(X^(2^d+1))
void Scheme_23::automorphismAndAdd(Ciphertext &cipher, int d)
{
}


void Scheme_23::square_double_add_const_rescale(Ciphertext& cipher1, Ciphertext &cipher2, double cnst)
{
    if (cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher2.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher2.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.l = cipher2.l;
    cipher1.scale = cipher2.scale * cipher2.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher2.ax_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    NTL::ZZ scaled_real = to_ZZ(round(cipher1.scale * cnst));
    for(int i = 0; i < cipher1.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

    poly_add_axbx_double_add_cnst_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, add_const_buffer, N, K, l+1, L+1);

    rescaleAndEqual(cipher1);
}

void Scheme_23::squareAndEqual_double_add_const_rescale(Ciphertext& cipher, double cnst)
{
    if (cipher.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher.scale = cipher.scale * cipher.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher.ax_device, cipher.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst));
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

    poly_add_axbx_double_add_cnst_batch_device(cipher.cipher_device, axbx1_mul, bxbx_mul, add_const_buffer, N, K, l+1, L+1);

    // context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    // dim3 resc_dim(N / rescale_block, cipher.l, 2);
	// rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, cipher.l, context.qiInvVecModql_device + cipher.l*(cipher.l-1)/2, context.qiInvVecModql_shoup_device + cipher.l*(cipher.l-1)/2);
    // cipher.scale = cipher.scale / context.qVec[cipher.l];
    // cipher.l -= 1;
    // context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    rescaleAndEqual(cipher);
}