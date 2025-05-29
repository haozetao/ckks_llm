#pragma once

#include "Bootstrapping_encoding.h"

#define GS_ROTATE_MULT_MATRIX_BLOCK 512

template<int gs, int bs>
__global__
__launch_bounds__(
    GS_ROTATE_MULT_MATRIX_BLOCK, 
    POLY_MIN_BLOCKS)
void Giant_Rotate_And_Mult_PlaintT_kernel
(uint64_tt* bs_cipher_T, uint64_tt* PQ_to_T_temp, uint64_tt* modUp_QjtoT_temp, uint64_tt** rotKey_pointer_device, int* rotSlots_device,
int n, int p_num, int q_num, int t_num, int Ri_blockNum, int Qj_blockNum, int dnum, int cipher_blockNum,
uint64_tt* rotGroup, int logNthRoot, uint64_tt** plaintextT_pointer_device, int* accNum_vec_device)
{
	register int idx_in_poly = blockIdx.z * GS_ROTATE_MULT_MATRIX_BLOCK + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * t_num * n + blockIdx.y * n;
	register int idx_in_T = blockIdx.y;
	register int idx_in_block = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

    register uint128_tt acc_bs1[bs] = 0, acc_bs2[bs] = 0;

#pragma unroll
    for(int idx_rot = 0; idx_rot < gs; idx_rot++)
    {
        uint64_tt* rot_i_pointer = rotKey_pointer_device[idx_rot];
        register uint128_tt acc1 = 0, acc2 = 0;

        if(rot_i_pointer == nullptr){
            acc1.low = PQ_to_T_temp[idx];
            acc2.low = PQ_to_T_temp[idx + Ri_blockNum*t_num*n];
        }else{
            // compute permuted idx
            register uint64_tt pow = rotGroup[rotSlots_device[idx_rot]];
            register int mask = 2 * n - 1;
            register int permuted_idx = 2*bitReverse(idx_in_poly, logNthRoot) + 1;
            permuted_idx = ((pow * permuted_idx & mask) - 1) >> 1;
            permuted_idx = bitReverse(permuted_idx, logNthRoot);

            register uint64_tt origin_bx = PQ_to_T_temp[permuted_idx + idx_in_block * t_num * n + idx_in_T * n + Ri_blockNum*t_num*n];

            // accumulate <d, rotKey_i>
            // ax
        #pragma unroll
            for(int i = 0; i < cipher_blockNum; i++)
            {
                uint64_tt ra_rot = modUp_QjtoT_temp[(i*t_num + idx_in_T)*n + permuted_idx];

                uint64_tt key_ra = rot_i_pointer[(i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + permuted_idx];
                uint64_tt key_rb = rot_i_pointer[Ri_blockNum*t_num*dnum*n + (i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + permuted_idx];

                madc_uint64_uint64_uint128(ra_rot, key_ra, acc1);
                madc_uint64_uint64_uint128(ra_rot, key_rb, acc2);
            }
            singleBarrett_new(acc1, t, t_mu);

            add_uint128_uint64(acc2, origin_bx, acc2);
            singleBarrett_new(acc2, t, t_mu);
        }

        // bs mult
    #pragma unroll
        for(int idx_in_bs = 0; idx_in_bs < bs; idx_in_bs++)
        {
            if(idx_rot < accNum_vec_device[idx_in_bs])
            {
                uint64_tt* plain = plaintextT_pointer_device[idx_in_bs * gs + idx_rot];

                uint64_tt mx = plain[idx_in_poly + idx_in_T * n];

                madc_uint64_uint64_uint128(acc1.low, mx, acc_bs1[idx_in_bs]);
                madc_uint64_uint64_uint128(acc2.low, mx, acc_bs2[idx_in_bs]);
            }
        }
    }

#pragma unroll
    for(int i = 0; i < bs; i++)
    {
        singleBarrett_new(acc_bs1[i], t, t_mu);
        singleBarrett_new(acc_bs2[i], t, t_mu);
        bs_cipher_T[idx + i*2     * Ri_blockNum*t_num*n] = acc_bs1[i].low;
        bs_cipher_T[idx + (i*2+1) * Ri_blockNum*t_num*n] = acc_bs2[i].low;
    }
}


__host__ void EncodingMatrix::Giant_Rotate_And_Mult_PlaintT
(uint64_tt* bs_cipher_T, uint64_tt* PQ_to_T_temp, int rotNum_here, uint64_tt* modUp_QjtoT_temp, vector<uint64_tt*> rotKey_pointer, vector<int> rotSlots_vec, 
vector<uint64_tt*> plaintextT_pointer, vector<int> accNum_vec, int bs, int gs, int l)
{
    int N = scheme.context.N;
    int p_num = scheme.context.p_num;
    int q_num = scheme.context.q_num;
    int t_num = scheme.context.t_num;
    int Ri_blockNum = scheme.context.Ri_blockNum;
    int Qj_blockNum = scheme.context.Qj_blockNum;
    int gamma = scheme.context.gamma;
    int dnum = scheme.context.dnum;

    int cipher_blockNum = ceil(double(l+1) / p_num);
    int blockNum = ceil(double(p_num+l+1) / gamma);

    // cout<<"cipher.l: "<<l<<endl;
    // cout<<"cipher_blockNum: "<<cipher_blockNum<<endl;
    // cout<<"blockNum: "<<blockNum<<endl;

    cudaMemcpy(rotKey_pointer_device, rotKey_pointer.data(), sizeof(uint64_t*) * rotKey_pointer.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(rotSlots_device, rotSlots_vec.data(), sizeof(int) * rotSlots_vec.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(plaintextT_pointer_device, plaintextT_pointer.data(), sizeof(uint64_tt*) * plaintextT_pointer.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(accNum_vec_device, accNum_vec.data(), sizeof(int) * accNum_vec.size(), cudaMemcpyHostToDevice);

    // cout<<"gs: "<<gs<<endl;
    // cout<<"bs: "<<bs<<endl;

    dim3 external_product_dim(blockNum, t_num, N / GS_ROTATE_MULT_MATRIX_BLOCK);
    if(gs == 16)
    {
        if(bs==4)
        {
            Giant_Rotate_And_Mult_PlaintT_kernel <16, 4> <<< external_product_dim, GS_ROTATE_MULT_MATRIX_BLOCK >>>
                (bs_cipher_T, PQ_to_T_temp, modUp_QjtoT_temp, rotKey_pointer_device, rotSlots_device,
                N, p_num, q_num, t_num, Ri_blockNum, Qj_blockNum, dnum, cipher_blockNum,
                scheme.context.rotGroups_device, log2(N),
                plaintextT_pointer_device, accNum_vec_device);
        }
        else if(bs==2)
        {
            Giant_Rotate_And_Mult_PlaintT_kernel <16, 2> <<< external_product_dim, GS_ROTATE_MULT_MATRIX_BLOCK >>>
                (bs_cipher_T, PQ_to_T_temp, modUp_QjtoT_temp, rotKey_pointer_device, rotSlots_device,
                N, p_num, q_num, t_num, Ri_blockNum, Qj_blockNum, dnum, cipher_blockNum,
                scheme.context.rotGroups_device, log2(N),
                plaintextT_pointer_device, accNum_vec_device);
        }
        else {
            printf("\n\n\nerror singleBarrett_new\n\n\n");
        }
    }
    else if(gs == 8)
    {
        Giant_Rotate_And_Mult_PlaintT_kernel <8, 2> <<< external_product_dim, GS_ROTATE_MULT_MATRIX_BLOCK >>>
            (bs_cipher_T, PQ_to_T_temp, modUp_QjtoT_temp, rotKey_pointer_device, rotSlots_device,
            N, p_num, q_num, t_num, Ri_blockNum, Qj_blockNum, dnum, cipher_blockNum,
            scheme.context.rotGroups_device, log2(N),
            plaintextT_pointer_device, accNum_vec_device);
    }
}

__global__
__launch_bounds__(
    GS_ROTATE_MULT_MATRIX_BLOCK, 
    POLY_MIN_BLOCKS) 
void Baby_Rotate_kernel
(uint64_tt* output, uint64_tt* modUp_QjtoT_temp, uint64_tt* input, int rotNum_here, uint64_tt** rotKey_pointer_device, int* rotSlots_device,
int n, int p_num, int q_num, int t_num, int Ri_blockNum, int Qj_blockNum, int cipher_blockNum, int dnum,
uint64_tt* rotGroup, int logNthRoot)
{
	register int idx_in_poly = blockIdx.z * GS_ROTATE_MULT_MATRIX_BLOCK + threadIdx.x;
    register int idx = idx_in_poly + blockIdx.x * t_num * n + blockIdx.y * n;
	register int idx_in_T = blockIdx.y;
	register int idx_in_block = blockIdx.x;

    register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
    register uint128_tt t_mu = {pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]};

#pragma unroll
    for(int idx_rot = 0; idx_rot < rotNum_here; idx_rot++)
    {
        register uint128_tt acc1 = 0, acc2 = 0;

        uint64_tt* rot_i_pointer = rotKey_pointer_device[idx_rot];
        if(rot_i_pointer == nullptr) continue;

        register long pow = rotGroup[rotSlots_device[idx_rot]];
        // computer permuted idx
        int mask = 2 * n - 1;
        int permuted_idx = 2*bitReverse(idx_in_poly, logNthRoot) + 1;
        permuted_idx = ((pow * permuted_idx & mask) - 1) >> 1;
        permuted_idx = bitReverse(permuted_idx, logNthRoot);
        // permuted_idx = permuted_idx + idx_in_block * t_num * n + idx_in_T * n;


        // accumulate <d, rotKey_i>
        // ax
    #pragma unroll
        for(int i = 0; i < cipher_blockNum; i++)
        {
            uint64_tt ra = modUp_QjtoT_temp[(i*t_num + idx_in_T)*n + permuted_idx];
            uint64_tt rb1 = rot_i_pointer[(i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + permuted_idx];
            uint64_tt rb2 = rot_i_pointer[Ri_blockNum*t_num*dnum*n + (i*t_num*Ri_blockNum + idx_in_block*t_num + idx_in_T)*n + permuted_idx];

            madc_uint64_uint64_uint128(ra, rb1, acc1);
            madc_uint64_uint64_uint128(ra, rb2, acc2);
        }
        add_uint128_uint64(acc1, output[idx], acc1);
        singleBarrett_new(acc1, t, t_mu);
        output[idx] = acc1.low;

        uint64_tt origin_bx = input[permuted_idx + idx_in_block * t_num * n + idx_in_T * n + Ri_blockNum*t_num*n];
        add_uint128_uint64(acc2, origin_bx, acc2);
        add_uint128_uint64(acc2, output[idx + Ri_blockNum*t_num*n], acc2);
        singleBarrett_new(acc2, t, t_mu);
        output[idx + Ri_blockNum*t_num*n] = acc2.low;
    }
}

__host__ void EncodingMatrix::Baby_Rotate(uint64_tt* output, uint64_tt* modUp_QjtoT_temp, uint64_tt* input, int rotNum_here, vector<uint64_tt*> rotKey_pointer, vector<int> rotSlots_vec, int l)
{
    int N = scheme.context.N;
    int p_num = scheme.context.p_num;
    int q_num = scheme.context.q_num;
    int t_num = scheme.context.t_num;
    int Ri_blockNum = scheme.context.Ri_blockNum;
    int Qj_blockNum = scheme.context.Qj_blockNum;
    int gamma = scheme.context.gamma;
    int dnum = scheme.context.dnum;


    int cipher_blockNum = ceil(double(l+1) / p_num);
    int blockNum = ceil(double(p_num+l+1) / gamma);

    cudaMemcpy(rotKey_pointer_device, rotKey_pointer.data(), sizeof(uint64_t*) * rotNum_here, cudaMemcpyHostToDevice);
    cudaMemcpy(rotSlots_device, rotSlots_vec.data(), sizeof(int) * rotNum_here, cudaMemcpyHostToDevice);

    dim3 external_product_dim(blockNum, t_num, N / GS_ROTATE_MULT_MATRIX_BLOCK);
    Baby_Rotate_kernel <<< external_product_dim, GS_ROTATE_MULT_MATRIX_BLOCK >>>
        (output, modUp_QjtoT_temp, input, rotNum_here, rotKey_pointer_device, rotSlots_device,
        N, p_num, q_num, t_num, Ri_blockNum, Qj_blockNum, cipher_blockNum, dnum,
        scheme.context.rotGroups_device, log2(N));
}

void EncodingMatrix::EvalCoeffsToSlots(vector<vector<PlaintextT*>>&A, Ciphertext&cipher)
{
    int stop = -1;
    int flagRem = 0;
    int BASE_NUM_LEVELS_TO_DROP = 1;
    long N = scheme.context.N;
    long M = scheme.context.M;

    int L = scheme.context.L;
    int K = scheme.context.K;
    int p_num = scheme.context.p_num;
    int q_num = scheme.context.q_num;
    int t_num = scheme.context.t_num;
    int Ri_blockNum = scheme.context.Ri_blockNum;
    int gamma = scheme.context.gamma;

    long logN = scheme.context.logN;
    long slots = cipher.slots;

    int levelBudget = m_paramsEnc[0];
    int layersCollapse = m_paramsEnc[1];
    int remCollapse = m_paramsEnc[2];
    int numRotations = m_paramsEnc[3];
    int b = m_paramsEnc[4];
    int g = m_paramsEnc[5];
    int numRotationsRem = m_paramsEnc[6];
    int bRem = m_paramsEnc[7];
    int gRem = m_paramsEnc[8];

    if (remCollapse != 0)
    {
        stop = 0;
        flagRem = 1;
    }

    scheme.context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    
    // input cipher is on Ql
    for(int s = levelBudget - 1; s > stop; s--)
    {
        // cipher now is on NTT
        int cipher_blockNum = ceil(double(cipher.l + 1) / K);

        // decomposed d <- cipher = (cipher.a)
        // (l+1)/K block of T
        scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
        scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

        // (a0, b0) <- P * (cipher.a, cipher.b)
        // on PQl
        cipher_mul_P_batch(cipher_buffer_PQ, cipher.cipher_device, N, K, 0, K, K, L, cipher.l+1, 2);

        // (a0, b0) on Ri_block * T
        scheme.context.modUpPQtoT_23(PQ_to_T_temp, cipher_buffer_PQ, cipher.l, 2);
        // (a0, b0) on T modulars NTT
        scheme.context.ToNTTInplace_for_externalProduct(PQ_to_T_temp, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        int rotNum_here = 0;
        for(int j = 0; j < g; j++)
        {
            if (rot_in_C2S[s][j] != 0)
            {
                rotKey_pointer[j] = scheme.rotKey_vec_23[rot_in_C2S[s][j]]->cipher_device;
                rotSlots[j] = rot_in_C2S[s][j];
                rotNum_here += 1;
            }
            else
            {
                rotKey_pointer[j] = nullptr;
                rotSlots[j] = -1;
            }
        }

        for (int32_t i = 0; i < b; i++)
        {
            int G = g * i;

            accNum_vec[i] = 0;
            for(int j = 0; j < g; j++) {
                if((G + j) != numRotations) {
                    plaintextT_pointer[i * g + j] = A[s][G + j]->mx_device;
                    // plaintextT_pointer[j] = A[s][G + j]->mx_device;
                    accNum_vec[i]++;
                }
            }
        }
        cipher.scale *= A[s][0]->scale;

        // now precompute Giant Hoisting rotate ciphers are on Ri_blockNum * T and NTT
        // for j in range(0, g)
        // accumulate (u0, u1) <- (aj, bj) * A[s][G + j]
        // to bs_cipher_T + i*Ri_blockNum*t_num*N*2
        Giant_Rotate_And_Mult_PlaintT(bs_cipher_T, PQ_to_T_temp, g, scheme.modUp_QjtoT_temp, rotKey_pointer, rotSlots, 
                                    plaintextT_pointer, accNum_vec, b, g, cipher.l);
        // for(int i = 0; i < b; i++)
        // {
        //     // u1 NTT -> Coeff
        //     scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * N*t_num*Ri_blockNum*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        //     // u1 PQl <- u1 T
        //     scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * N*t_num*Ri_blockNum*2, cipher.l, 2);
        //     // u1 <- P^-1 * u1
        //     scheme.context.modDownPQltoQl_23(cipher.cipher_device, T_to_PQ_temp, cipher.l, 2);
        //     scheme.decrypt_display(secretKey, *scheme.cipher_temp_pool);

        //     scheme.context.ToNTTInplace_for_externalProduct(bs_cipher_T + i * N*t_num*Ri_blockNum*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // }


        for(int i = 0; i < b; i++)
        {
            if(i != 0)
            {
                if(rot_out_C2S[s][i] == 0) 
                {
                    cipher_add_T_batch_device(bs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, N, K+L+1, t_num, t_num, Ri_blockNum*2);
                    continue;
                }
                
                // u1 NTT -> Coeff
                cudaMemcpy(gs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, sizeof(uint64_tt) * Ri_blockNum*t_num*2*N, cudaMemcpyDeviceToDevice);
                scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
                // u1 PQl <- u1 T
                scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * Ri_blockNum*t_num*N*2, cipher.l, 1);
                // u1 <- P^-1 * u1
                scheme.context.modDownPQltoQl_23(cipher.ax_device, T_to_PQ_temp, cipher.l, 1);
                
                // EBConv u1 Ql -> T * gamma
                scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
                scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

                rotKey_pointer[0] = scheme.rotKey_vec_23[rot_out_C2S[s][i]]->cipher_device;
                rotSlots[0] = rot_out_C2S[s][i];
                Baby_Rotate(bs_cipher_T, scheme.modUp_QjtoT_temp, gs_cipher_T, 1, rotKey_pointer, rotSlots, cipher.l);
            }
        }
        
        // u1 NTT -> Coeff
        scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // u1 PQl <- u1 T
        scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T, cipher.l, 2);
        // u1 <- P^-1 * u1
        scheme.context.modDownPQltoQl_23(cipher.cipher_device, T_to_PQ_temp, cipher.l, 2);
    
        scheme.rescaleAndEqual_noNTT(cipher);

        // scheme.context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
        // scheme.decrypt_display(secretKey, cipher);
        // scheme.context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    }


    if (flagRem)
    {
        // cipher now is on NTT
        int cipher_blockNum = ceil(double(cipher.l + 1) / K);

        // decomposed d <- cipher = (cipher.a)
        // (l+1)/K block of T
        scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
        scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

        // (a0, b0) <- P * (cipher.a, cipher.b)
        // on PQl
        cipher_mul_P_batch(cipher_buffer_PQ, cipher.cipher_device, N, K, 0, K, K, L, cipher.l+1, 2);

        // (a0, b0) on Ri_block * T
        scheme.context.modUpPQtoT_23(PQ_to_T_temp, cipher_buffer_PQ, cipher.l, 2);
        // (a0, b0) on T modulars NTT
        scheme.context.ToNTTInplace_for_externalProduct(PQ_to_T_temp, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        int rotNum_here = 0;
        for(int j = 0; j < gRem; j++)
        {
            if (rot_in_C2S[stop][j] != 0)
            {
                rotKey_pointer[j] = scheme.rotKey_vec_23[rot_in_C2S[stop][j]]->cipher_device;
                rotSlots[j] = rot_in_C2S[stop][j];
                rotNum_here += 1;
            }
            else
            {
                rotKey_pointer[j] = nullptr;
                rotSlots[j] = -1;
            }
        }
        // cout<<endl;


        for (int32_t i = 0; i < bRem; i++)
        {
            int GRem = gRem * i;

            accNum_vec[i] = 0;
            for(int j = 0; j < gRem; j++) {
                if((GRem + j) != numRotationsRem) {
                    plaintextT_pointer[i * gRem + j] = A[stop][GRem + j]->mx_device;
                    accNum_vec[i]++;
                }
            }
        }
        cipher.scale *= A[stop][0]->scale;

        // now precompute Giant Hoisting rotate ciphers are on Ri_blockNum * T and NTT
        // for j in range(0, g)
        // accumulate (u0, u1) <- (aj, bj) * A[s][G + j]
        // to bs_cipher_T + i*Ri_blockNum*t_num*N*2
        Giant_Rotate_And_Mult_PlaintT(bs_cipher_T, PQ_to_T_temp, gRem, scheme.modUp_QjtoT_temp, rotKey_pointer, rotSlots, 
                                        plaintextT_pointer, accNum_vec, bRem, gRem, cipher.l);
        // for(int i = 0; i < bRem; i++)
        // {
        //     scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        //     // u1 PQl <- u1 T
        //     scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * Ri_blockNum*t_num*N*2, cipher.l, 1);
        //     // u1 <- P^-1 * u1
        //     scheme.context.modDownPQltoQl_23(scheme.cipher_temp_pool->ax_device, T_to_PQ_temp, cipher.l, 1);
        //     scheme.decrypt_display(secretKey, *scheme.cipher_temp_pool);
        //     scheme.context.ToNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // }

        for(int i = 0; i < bRem; i++)
        {
            if(i != 0)
            {
                if(rot_out_C2S[stop][i] == 0) 
                {
                    cipher_add_T_batch_device(bs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, N, K+L+1, t_num, t_num, Ri_blockNum*2);
                    continue;
                }
                // u1 NTT -> Coeff
                cudaMemcpy(gs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, sizeof(uint64_tt) * Ri_blockNum*t_num*2*N, cudaMemcpyDeviceToDevice);
                scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
                // u1 PQl <- u1 T
                scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * Ri_blockNum*t_num*N*2, cipher.l, 1);
                // u1 <- P^-1 * u1
                scheme.context.modDownPQltoQl_23(cipher.ax_device, T_to_PQ_temp, cipher.l, 1);
                
                // EBConv u1 Ql -> T * gamma
                scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
                scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

                rotKey_pointer[0] = scheme.rotKey_vec_23[rot_out_C2S[stop][i]]->cipher_device;
                rotSlots[0] = rot_out_C2S[stop][i];
                Baby_Rotate(bs_cipher_T, scheme.modUp_QjtoT_temp, gs_cipher_T, 1, rotKey_pointer, rotSlots, cipher.l);
            }
        }
        
        // u1 NTT -> Coeff
        scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // u1 PQl <- u1 T
        scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T, cipher.l, 2);
        // u1 <- P^-1 * u1
        scheme.context.modDownPQltoQl_23(cipher.cipher_device, T_to_PQ_temp, cipher.l, 2);

        scheme.rescaleAndEqual_noNTT(cipher);
        
    }
    scheme.context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}


void EncodingMatrix::EvalSlotsToCoeffs(vector<vector<PlaintextT*>>&A, Ciphertext&cipher)
{
    int BASE_NUM_LEVELS_TO_DROP = 1;
    int levelBudget = m_paramsDec[0];
    int layersCollapse = m_paramsDec[1];
    int remCollapse = m_paramsDec[2];
    int numRotations = m_paramsDec[3];
    int b = m_paramsDec[4];
    int g = m_paramsDec[5];
    int numRotationsRem = m_paramsDec[6];
    int bRem = m_paramsDec[7];
    int gRem = m_paramsDec[8];
    int flagRem = 0;

    long N = scheme.context.N;
    long logN = scheme.context.logN;
    long slots = cipher.slots;

    int L = scheme.context.L;
    int K = scheme.context.K;
    int p_num = scheme.context.p_num;
    int q_num = scheme.context.q_num;
    int t_num = scheme.context.t_num;
    int Ri_blockNum = scheme.context.Ri_blockNum;
    int Qj_blockNum = scheme.context.Qj_blockNum;
    int gamma = scheme.context.gamma;


    if (remCollapse != 0)
    {
        flagRem = 1;
    }


    scheme.context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);

    for (int32_t s = 0; s < levelBudget - flagRem; s++)
    {
        int cipher_blockNum = ceil(double(cipher.l + 1) / K);

        // decomposed d <- cipher = (cipher.a)
        // (l+1)/K block of T
        scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
        scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

        // (a0, b0) <- P * (cipher.a, cipher.b)
        // on PQl
        cipher_mul_P_batch(cipher_buffer_PQ, cipher.cipher_device, N, K, 0, K, K, L, cipher.l+1, 2);

        // (a0, b0) on Ri_block * T
        scheme.context.modUpPQtoT_23(PQ_to_T_temp, cipher_buffer_PQ, cipher.l, 2);
        // (a0, b0) on T modulars NTT
        scheme.context.ToNTTInplace_for_externalProduct(PQ_to_T_temp, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        int rotNum_here = 0;
        for(int j = 0; j < g; j++)
        {
            if (rot_in_S2C[s][j] != 0)
            {
                rotKey_pointer[j] = scheme.rotKey_vec_23[rot_in_S2C[s][j]]->cipher_device;
                rotSlots[j] = rot_in_S2C[s][j];
                rotNum_here += 1;
            }
            else
            {
                rotKey_pointer[j] = nullptr;
                rotSlots[j] = -1;
            }
        }

        for (int32_t i = 0; i < b; i++)
        {
            int G = g * i;

            accNum_vec[i] = 0;
            for(int j = 0; j < g; j++) {
                if((G + j) != numRotations) {
                    plaintextT_pointer[i * g + j] = A[s][G + j]->mx_device;
                    // plaintextT_pointer[j] = A[s][G + j]->mx_device;
                    accNum_vec[i]++;
                }
            }
        }
        cipher.scale *= A[s][0]->scale;


        // now precompute Giant Hoisting rotate ciphers are on Ri_blockNum * T and NTT
        // for j in range(0, g)
        // accumulate (u0, u1) <- (aj, bj) * A[s][G + j]
        // to bs_cipher_T + i*Ri_blockNum*t_num*N*2
        Giant_Rotate_And_Mult_PlaintT(bs_cipher_T, PQ_to_T_temp, g, scheme.modUp_QjtoT_temp, rotKey_pointer, rotSlots, 
                                    plaintextT_pointer, accNum_vec, b, g, cipher.l);
        
        for(int i = 0; i < b; i++)
        {
            if(i != 0)
            {
                if(rot_out_S2C[s][i] == 0) 
                {
                    cipher_add_T_batch_device(bs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, N, K+L+1, t_num, t_num, Ri_blockNum*2);
                    continue;
                }

                // u1 NTT -> Coeff
                cudaMemcpy(gs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, sizeof(uint64_tt) * Ri_blockNum*t_num*2*N, cudaMemcpyDeviceToDevice);
                scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
                // u1 PQl <- u1 T
                scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * Ri_blockNum*t_num*N*2, cipher.l, 1);
                // u1 <- P^-1 * u1
                scheme.context.modDownPQltoQl_23(scheme.cipher_temp_pool->ax_device, T_to_PQ_temp, cipher.l, 1);
                
                // EBConv u1 Ql -> T * gamma
                scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, scheme.cipher_temp_pool->ax_device, cipher.l, 1);
                scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

                rotKey_pointer[0] = scheme.rotKey_vec_23[rot_out_S2C[s][i]] -> cipher_device;
                rotSlots[0] = rot_out_S2C[s][i];
                Baby_Rotate(bs_cipher_T, scheme.modUp_QjtoT_temp, gs_cipher_T, 1, rotKey_pointer, rotSlots, cipher.l);
            }
        }
        
        // u1 NTT -> Coeff
        scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // u1 PQl <- u1 T
        scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T, cipher.l, 2);
        // u1 <- P^-1 * u1
        scheme.context.modDownPQltoQl_23(cipher.cipher_device, T_to_PQ_temp, cipher.l, 2);

        scheme.rescaleAndEqual_noNTT(cipher);
    }

    if (flagRem)
    {
        int32_t stop = levelBudget - flagRem;

        int cipher_blockNum = ceil(double(cipher.l + 1) / K);

        // decomposed d <- cipher = (cipher.a)
        // (l+1)/K block of T
        scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, cipher.ax_device, cipher.l, 1);
        scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

        // (a0, b0) <- P * (cipher.a, cipher.b)
        // on PQl
        cipher_mul_P_batch(cipher_buffer_PQ, cipher.cipher_device, N, K, 0, K, K, L, cipher.l+1, 2);

        // (a0, b0) on Ri_block * T
        scheme.context.modUpPQtoT_23(PQ_to_T_temp, cipher_buffer_PQ, cipher.l, 2);
        // (a0, b0) on T modulars NTT
        scheme.context.ToNTTInplace_for_externalProduct(PQ_to_T_temp, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        int rotNum_here = 0;
        for(int j = 0; j < gRem; j++)
        {
            if (rot_in_S2C[stop][j] != 0)
            {
                rotKey_pointer[j] = scheme.rotKey_vec_23[rot_in_S2C[stop][j]]->cipher_device;
                rotSlots[j] = rot_in_S2C[stop][j];
                rotNum_here += 1;
            }
            else
            {
                rotKey_pointer[j] = nullptr;
                rotSlots[j] = -1;
            }
        }


        for (int32_t i = 0; i < bRem; i++)
        {
            int GRem = gRem * i;

            accNum_vec[i] = 0;
            for(int j = 0; j < gRem; j++) {
                if((GRem + j) != numRotationsRem) {
                    plaintextT_pointer[i * gRem + j] = A[stop][GRem + j]->mx_device;
                    accNum_vec[i]++;
                }
            }
        }
        cipher.scale *= A[stop][0]->scale;

        // now precompute Giant Hoisting rotate ciphers are on Ri_blockNum * T and NTT
        // for j in range(0, g)
        // accumulate (u0, u1) <- (aj, bj) * A[s][G + j]
        // to bs_cipher_T + i*Ri_blockNum*t_num*N*2
        Giant_Rotate_And_Mult_PlaintT(bs_cipher_T, PQ_to_T_temp, gRem, scheme.modUp_QjtoT_temp, rotKey_pointer, rotSlots, 
                                        plaintextT_pointer, accNum_vec, bRem, gRem, cipher.l);

        for(int i = 0; i < bRem; i++)
        {
            if(i != 0)
            {
                if(rot_out_S2C[stop][i] == 0) 
                {
                    cipher_add_T_batch_device(bs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, N, K+L+1, t_num, t_num, Ri_blockNum*2);
                    continue;
                }
                // u1 NTT -> Coeff
                cudaMemcpy(gs_cipher_T, bs_cipher_T + i * Ri_blockNum*t_num*N*2, sizeof(uint64_tt) * Ri_blockNum*t_num*2*N, cudaMemcpyDeviceToDevice);
                scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T + i * Ri_blockNum*t_num*N*2, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
                // u1 PQl <- u1 T
                scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T + i * Ri_blockNum*t_num*N*2, cipher.l, 1);
                // u1 <- P^-1 * u1
                scheme.context.modDownPQltoQl_23(scheme.cipher_temp_pool->ax_device, T_to_PQ_temp, cipher.l, 1);
                
                // EBConv u1 Ql -> T * gamma
                scheme.context.modUpQjtoT_23(scheme.modUp_QjtoT_temp, scheme.cipher_temp_pool->ax_device, cipher.l, 1);
                scheme.context.ToNTTInplace(scheme.modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

                rotKey_pointer[0] = scheme.rotKey_vec_23[rot_out_S2C[stop][i]] -> cipher_device;
                rotSlots[0] = rot_out_S2C[stop][i];
                Baby_Rotate(bs_cipher_T, scheme.modUp_QjtoT_temp, gs_cipher_T, 1, rotKey_pointer, rotSlots, cipher.l);
            }
        }
        
        // u1 NTT -> Coeff
        scheme.context.FromNTTInplace_for_externalProduct(bs_cipher_T, 0, K+L+1, ceil(double(K+cipher.l+1) / gamma), t_num, t_num, Ri_blockNum*t_num, 2);
        // u1 PQl <- u1 T
        scheme.context.modUpTtoPQl_23(T_to_PQ_temp, bs_cipher_T, cipher.l, 2);
        // u1 <- P^-1 * u1
        scheme.context.modDownPQltoQl_23(cipher.cipher_device, T_to_PQ_temp, cipher.l, 2);

        scheme.rescaleAndEqual_noNTT(cipher);
        
    }
    scheme.context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}