#pragma once

#include "PCMM_Scheme.h"
#include "PPMM_kernel.cuh"

PCMM_Scheme::PCMM_Scheme(PCMM_Context& pcmm_context, Scheme_23& scheme, Bootstrapper& bootstrapper): pcmm_context(pcmm_context), scheme(scheme), context(scheme.context), bootstrapper(bootstrapper)
{
    int N = context.N;
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    cudaMalloc(&embeded_mlwe_buffer, sizeof(uint64_tt) * (mlwe_rank+1) * N * (pcmm_context.ringpack_pq_count - 1));
    
    repacking_cipher_pointer = vector<uint64_tt*>(mlwe_rank, nullptr);
    cudaMalloc(&repacking_cipher_pointer_device, sizeof(uint64_tt*) * mlwe_rank);

    cudaMalloc(&ppmm_output, sizeof(uint64_tt) * N1 * (mlwe_rank+1) * 256 * pcmm_context.ringpack_q_count);
}

__host__ void PCMM_Scheme::convertMLWESKfromRLWESK(MLWESecretKey& mlwe_sk, SecretKey& rlwe_sk)
{
    // scheme.context.FromNTTInplace(rlwe_sk.sx_device, 0, 0, 1, rlwe_sk.K + rlwe_sk.L + 1);
    int K = scheme.context.K;
    int p_num = scheme.context.p_num;
    int L = scheme.context.L;
    int N = scheme.context.N;
    
    int N1 = mlwe_sk.N1;
    int mlwe_rank = mlwe_sk.k;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;

	scheme.context.FromNTTInplace(rlwe_sk.sx_device, 0, 0, 1, K+L+1, K+L+1);

    dim3 ringDown_dim1(N / ringSwitch_block, ringpack_q_count);
    ringDown_kernel <<< ringDown_dim1, ringSwitch_block >>> (rlwe_sk.sx_device, mlwe_sk.sx_device, N, N1, mlwe_rank, 0);
    dim3 ringDown_dim2(N / ringSwitch_block, ringpack_q_count);
    ringDown_kernel <<< ringDown_dim2, ringSwitch_block >>> (rlwe_sk.sx_device + p_num*N, mlwe_sk.sx_device + ringpack_p_count*N1*mlwe_rank, N, N1, mlwe_rank, p_num);

	scheme.context.ToNTTInplace(rlwe_sk.sx_device, 0, 0, 1, K+L+1, K+L+1);

    pcmm_context.ToNTTInplace(mlwe_sk.sx_device, mlwe_rank, ringpack_pq_count, 0, 0, 1);
}

template<int mlwe_rank>
__global__ void rlweCipherDecompose_kernel(uint64_tt* rlwe_device, uint64_tt** mlwe_pointer_device, int N, int N1, int mlwe_num, int mlwe_L, int L, int p_num)
{
    int global_idx = blockIdx.x * ringSwitch_block + threadIdx.x;
    int idx_mod = blockIdx.y;
    int idx_mlwe = blockIdx.z;

    uint64_tt q = pqt_cons[p_num + idx_mod];

    uint64_tt* rlwe_ax_this_mod = rlwe_device + N * idx_mod;
    uint64_tt* rlwe_bx_this_mod = rlwe_device + N * (L + 1 + idx_mod);

    uint64_tt* mlwe_this_batch = mlwe_pointer_device[idx_mlwe];

    uint64_tt* mlwe_ax_this_mod = mlwe_this_batch + N1 * (mlwe_rank + 1) * idx_mod;
    uint64_tt* mlwe_bx_this_mod = mlwe_ax_this_mod + N1 * mlwe_rank;

    int gap = mlwe_rank / mlwe_num;

    int addr = (idx_mlwe * gap + global_idx * mlwe_rank);
    int over = addr >= N;

    mlwe_ax_this_mod[global_idx] = over ? negate_modq(rlwe_ax_this_mod[addr - N], q) : rlwe_ax_this_mod[addr];
    mlwe_bx_this_mod[global_idx] = over ? negate_modq(rlwe_bx_this_mod[addr - N], q) : rlwe_bx_this_mod[addr];

    #pragma unroll
    for(int i = 1; i < mlwe_rank; i++){
        if(global_idx == 0){
            addr = (N1-1)*mlwe_rank + (mlwe_rank - i) + idx_mlwe * gap;
            over = addr >= N;
            mlwe_ax_this_mod[global_idx + i*N1] = over ? rlwe_ax_this_mod[addr - N] : negate_modq(rlwe_ax_this_mod[addr], q);
        }
        else{
            addr = ((global_idx - 1) * mlwe_rank + (mlwe_rank - i)) + idx_mlwe * gap;
            over = addr >= N;
            mlwe_ax_this_mod[global_idx + i*N1] = over ? negate_modq(rlwe_ax_this_mod[addr - N], q) : rlwe_ax_this_mod[addr];
        }
    }
}

__host__ void PCMM_Scheme::rlweCipherDecompose(Ciphertext& rlwe_cipher, vector<MLWECiphertext*> mlwe_cipher_decomposed)
{
    int N = context.N;
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int p_num = context.p_num;

    int l = rlwe_cipher.l;
    int L = rlwe_cipher.L;
    int mlwe_L = mlwe_cipher_decomposed[0]->L;
    int K = context.K;

    int mlwe_num = mlwe_cipher_decomposed.size();
    // if(mlwe_num != mlwe_rank / 2){
    //     cout << "rlweCipherDecomposeReal need rank / 2 mlwe cipher!" << endl;
    //     return;
    // }
    context.FromNTTInplace(rlwe_cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    for(int i = 0; i < mlwe_num; i++){
        // repacking_cipher_pointer[mlwe_rank - i - 1] = mlwe_cipher_decomposed[i]->cipher_device;
        repacking_cipher_pointer[i] = mlwe_cipher_decomposed[i]->cipher_device;
    }
    cudaMemcpy(repacking_cipher_pointer_device, repacking_cipher_pointer.data(), sizeof(uint64_tt*) * mlwe_num, cudaMemcpyHostToDevice);

    dim3 rlweCipherDecompose_dim(N1 / ringSwitch_block, l + 1, mlwe_num);
    if(mlwe_rank == 256){
        rlweCipherDecompose_kernel <256> <<< rlweCipherDecompose_dim, ringSwitch_block >>> (rlwe_cipher.cipher_device, repacking_cipher_pointer_device, N, N1, mlwe_num, mlwe_L, L, p_num);
    } else {
        cout << "mlwe rank not supported!" << endl;
    }
    context.ToNTTInplace(rlwe_cipher.cipher_device, 0, K, 2, l + 1, L + 1);
}

template<int dnum>
__global__ void EmbedMLWESKtoRLWESK(uint64_tt* rlwekey_cipher_device, uint64_tt* mlwekey_cipher_device, int N, int N1, int mlwe_rank, int p_count, int q_count)
{
    int idx_N1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_mlwe = blockIdx.y;
    int idx_mod = blockIdx.z;

    // [a]%q0, ..., [a]%qi, ..., [a]%q{k-1}   [b]%q0, ..., [b]%qi, ..., [b]%q{k-1}
    uint64_tt* rlwe_key_this = rlwekey_cipher_device + (N * (p_count + q_count) * 2) * idx_mlwe;
    // [b]%qi
    uint64_tt* rlwe_bx_this_mod = rlwe_key_this + N * (p_count + q_count) + N * (p_count + idx_mod);
    
    uint64_tt* mlwesx_this_mod = mlwekey_cipher_device + N1 * mlwe_rank * (p_count + idx_mod);
    uint64_tt* mlwesx_this_mod_this_rank = mlwesx_this_mod + idx_mlwe * N1;

    uint64_tt p0 = ringpack_pq_cons[0];
    uint64_tt qi = ringpack_pq_cons[p_count + idx_mod];
    uint128_tt mu = {ringpack_pq_mu_high_cons[p_count + idx_mod], ringpack_pq_mu_low_cons[p_count + idx_mod]};

    uint128_tt temp;
    mul64(mlwesx_this_mod_this_rank[idx_N1], p0, temp);
    singleBarrett_new(temp, qi, mu);
    rlwe_bx_this_mod[idx_N1 * mlwe_rank] = temp.low;
}

__host__ void PCMM_Scheme::addRepakcingKey(MLWESecretKey& mlwe_sk, SecretKey& rlwe_sk)
{
    int K = scheme.context.K;
    int p_num = scheme.context.p_num;
    int L = scheme.context.L;
    int N = scheme.context.N;
    
    int N1 = mlwe_sk.N1;
    int mlwe_rank = mlwe_sk.k;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;
    if(repackingKeys.size() == mlwe_rank){
        cout << "repacking keys already generated!" << endl;
        return;
    }

    pcmm_context.FromNTTInplace(mlwe_sk.sx_device, mlwe_rank, ringpack_pq_count, 0, 0, 1);

    uint64_tt* ringpackingKeys_cipher;
    cudaMalloc(&ringpackingKeys_cipher, sizeof(uint64_tt) * N * (ringpack_p_count + ringpack_q_count - 1) * 2 * mlwe_rank);
    
    int dnum = (ringpack_q_count - 1) / ringpack_p_count;
    int repacking_key_pnum = ringpack_p_count;
    int repacking_key_qnum = ringpack_q_count - 1;

    printf("before key gen Key.L: %d, K: %d, dnum: %d\n", ringpack_q_count - 1, ringpack_p_count, dnum);
    for(int i = 0; i < mlwe_rank; i++)
    {
        Key* temp_key = new Key(ringpackingKeys_cipher + i*N*(ringpack_p_count+ringpack_q_count-1)*2, N, repacking_key_qnum - 1, repacking_key_pnum, dnum);
        repackingKeys.push_back(temp_key);
    }

    // key[i].bx = \iota_(mlwe_sx[i]) * p0
    if(dnum == 1){
        dim3 ringPacking_dim(N1 / ringSwitch_block, mlwe_rank, ringpack_q_count - 1);
        EmbedMLWESKtoRLWESK <1> <<< ringPacking_dim, ringSwitch_block >>> (ringpackingKeys_cipher, mlwe_sk.sx_device, N, N1, mlwe_rank, ringpack_p_count, ringpack_q_count - 1);
    }else{
        cout << "dnum not supported!" << endl;
    }
    pcmm_context.ToNTTInplace(mlwe_sk.sx_device, mlwe_rank, ringpack_pq_count - 1, 0, 0, 1);

    for(int i = 0; i < mlwe_rank; i++)
    {
        long randomArray_len = sizeof(uint64_tt) * N * (ringpack_p_count + ringpack_q_count - 1) + sizeof(uint32_tt) * N;

        uint8_tt* random_buffer_device = context.randomArray_device;
        uint8_tt* random_ex_device = random_buffer_device;
        uint8_tt* random_ax_device = random_ex_device + sizeof(uint32_tt) * N;
        RNG::generateRandom_device(random_buffer_device, randomArray_len);

        // sample ex and convert to NTT
        uint64_tt* ex_swk = scheme.ex_swk;
        Sampler::gaussianSampler_xq(random_ex_device, ex_swk, N, 0, K - ringpack_p_count, ringpack_pq_count - 1);

        // sample ax (in NTT)
        Sampler::uniformSampler_xq(random_ax_device, repackingKeys[i]->ax_device, N, 0, K - ringpack_p_count, ringpack_pq_count - 1);

        // key[i].bx = ex + \iota_(mlwe_sx[i]) * p0 (in NTT)
        poly_add_batch_device(repackingKeys[i]->bx_device, ex_swk, N, 0, 0, K - ringpack_p_count, ringpack_pq_count - 1);

        // convert bx = ex + \iota_(mlwe_sx[i]) * p0 to NTT
        context.ToNTTInplace(repackingKeys[i]->bx_device, 0, K - ringpack_p_count, 1, ringpack_pq_count - 1, ringpack_pq_count - 1);

        // temp_mul = ax*rlwe_sx (in NTT)
        uint64_tt* temp_mul = scheme.temp_mul;
        barrett_batch_3param_device(temp_mul, rlwe_sk.sx_device, repackingKeys[i]->ax_device, N, 0, K - ringpack_p_count, 0, K - ringpack_p_count, ringpack_pq_count - 1);

        // key[i].bx = -ax*sx + \iota_(mlwe_sx[i]) * p0 + ex (in NTT)
        poly_sub_batch_device(repackingKeys[i]->bx_device, temp_mul, N, 0, 0, K - ringpack_p_count, ringpack_pq_count - 1);
    }
}

// embed mlwe_{N/k}^k -> mlwe_{N}^k and modup q0 -> pq0
__global__ void mlweCipherEmbedding_kernel(uint64_tt* embeded_mlwe_cipher, uint64_tt** mlwe_cipher_pointer_device, int N1, int N, int mlwe_rank, int mlwe_num, int modup_p_count, int mlwe_q_count)
{
    int idx_in_N1 = blockIdx.x * ringSwitch_block + threadIdx.x;
    int idx_mod = blockIdx.y;
    int idx_mlwe = blockIdx.z;
    
    uint64_tt p0 = ringpack_pq_cons[0];
    uint64_tt qi = ringpack_pq_cons[modup_p_count + idx_mod];

    uint64_tt* this_big_mlwe_poly = embeded_mlwe_cipher + idx_mlwe * N * (modup_p_count+mlwe_q_count-1);

    int gap = mlwe_rank / mlwe_num;
#pragma unroll
    for(int i = 0; i < mlwe_rank; i++)
    {
        register uint64_tt small_mlwe_poly_mod_q;
        if(i % gap == 0){
            uint64_tt* this_small_mlwe_sample = mlwe_cipher_pointer_device[i / gap];
            uint64_tt* small_mlwe_this_mod = this_small_mlwe_sample + idx_mod * N1 * (mlwe_rank + 1);
            small_mlwe_poly_mod_q = small_mlwe_this_mod[idx_in_N1 + idx_mlwe * N1];
        } else {
            small_mlwe_poly_mod_q = 0;
        }
        // mod q
        this_big_mlwe_poly[(modup_p_count + idx_mod)*N + idx_in_N1*mlwe_rank + i] = small_mlwe_poly_mod_q;
        // mod p and guarantee pi > qi
        if(small_mlwe_poly_mod_q > (qi / 2)){
            this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + i] = (p0 - (qi - small_mlwe_poly_mod_q)) % p0;
        } else {
            this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + i] = small_mlwe_poly_mod_q;
        }
    }
}

template<int mlwe_rank>
__global__ void mlweCipherMultRepackingKey_kernel(uint64_tt* output, uint64_tt* embeded_mlwe_cipher, uint64_tt* repacking_keys, int N1, int N, int mod_num)
{
    int idx_in_N = blockIdx.x * ringSwitch_block + threadIdx.x;
    int idx_mod = blockIdx.y;

    register uint64_tt mod = ringpack_pq_cons[idx_mod];
    register uint128_tt mu = {ringpack_pq_mu_high_cons[idx_mod], ringpack_pq_mu_low_cons[idx_mod]};

    uint128_tt acc_ax = 0, acc_bx = 0;
#pragma unroll
    for(int i = 0; i < mlwe_rank; i++)
    {
        uint64_tt* this_mlwe_sample = embeded_mlwe_cipher + i * N * mod_num;
        register uint64_tt cipher_ax = this_mlwe_sample[idx_in_N + idx_mod * N];

        uint64_tt* this_ringpacking_key = repacking_keys + i * N * mod_num * 2;
        register uint64_tt key_ax = this_ringpacking_key[idx_in_N + idx_mod * N];
        register uint64_tt key_bx = this_ringpacking_key[idx_in_N + (idx_mod + mod_num) * N];

        madc_uint64_uint64_uint128(cipher_ax, key_ax, acc_ax);
        madc_uint64_uint64_uint128(cipher_ax, key_bx, acc_bx);
    }
    // ax
    singleBarrett_new(acc_ax, mod, mu);
    output[idx_in_N + idx_mod * N] = acc_ax.low;
    // bx
    singleBarrett_new(acc_bx, mod, mu);
    (output + N*mod_num)[idx_in_N + idx_mod * N] = acc_bx.low;
}

__global__ void ringpacking_ModDown_kernel(uint64_tt* output_q, uint64_tt* input_pq, int N, int ring_packing_p_count, int ring_packing_q_count, int cipher_q_num, uint64_tt* p_inv_mod_q, uint64_tt* p_inv_mod_q_shoup)
{
    register int idx_in_poly = blockIdx.x * ringSwitch_block + threadIdx.x;
	register int idx_in_pq = blockIdx.y;
	register int idx_in_cipher = blockIdx.z;

	register uint64_tt qi = ringpack_pq_cons[ring_packing_p_count + idx_in_pq];
	register uint64_tt pi = ringpack_pq_cons[0];
    register uint64_tt ra = input_pq[idx_in_cipher*N*(ring_packing_p_count+ring_packing_q_count-1) + idx_in_pq*N + idx_in_poly];

    ra %= qi;

	register uint64_tt Pinvmodqi = p_inv_mod_q[idx_in_pq];
	register uint64_tt Pinvmodqi_shoup = p_inv_mod_q_shoup[idx_in_pq];
	ra = qi - ra + input_pq[idx_in_cipher*N*(ring_packing_p_count+ring_packing_q_count-1) + (ring_packing_p_count+idx_in_pq) * N + idx_in_poly];

	output_q[idx_in_cipher*N*cipher_q_num + idx_in_pq*N + idx_in_poly] = mulMod_shoup(ra, Pinvmodqi, Pinvmodqi_shoup, qi);
}


// embed mlwe_{N/k}^k -> mlwe_{N}^k and modup q0 -> pq0
__global__ void mlweCipherEmbedding_new_kernel(uint64_tt* embeded_mlwe_cipher, uint64_tt** mlwe_cipher_pointer_device, int N1, int N, int mlwe_rank, int mlwe_num, int modup_p_count, int mlwe_q_count)
{
    int idx_in_N1 = blockIdx.x * ringSwitch_block + threadIdx.x;
    int idx_mod = blockIdx.y;
    int idx_mlwe = blockIdx.z;
    
    uint64_tt p0 = ringpack_pq_cons[0];
    uint64_tt qi = ringpack_pq_cons[modup_p_count + idx_mod];


    int gap = mlwe_rank / mlwe_num;
#pragma unroll
    for(int i = 0; i < mlwe_rank + 1; i++)
    {
        uint64_tt* this_big_mlwe_poly = embeded_mlwe_cipher + i * N * (modup_p_count+mlwe_q_count-1);


        uint64_tt* this_small_mlwe_sample = mlwe_cipher_pointer_device[idx_mlwe];
        uint64_tt* small_mlwe_this_mod = this_small_mlwe_sample + idx_mod * N1 * (mlwe_rank + 1);
        register uint64_tt small_mlwe_poly_mod_q = small_mlwe_this_mod[idx_in_N1 + i * N1];

        this_big_mlwe_poly[(modup_p_count + idx_mod)*N + idx_in_N1*mlwe_rank + idx_mlwe*gap] = small_mlwe_poly_mod_q;
        // mod p and guarantee pi > qi
        if(small_mlwe_poly_mod_q > (qi / 2)){
            this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + idx_mlwe*gap] = (p0 - (qi - small_mlwe_poly_mod_q)) % p0;
        } else {
            this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + idx_mlwe*gap] = small_mlwe_poly_mod_q;
        }


        // register uint64_tt small_mlwe_poly_mod_q;
        // if(i % gap == 0){
        //     uint64_tt* this_small_mlwe_sample = mlwe_cipher_pointer_device[i / gap];
        //     uint64_tt* small_mlwe_this_mod = this_small_mlwe_sample + idx_mod * N1 * (mlwe_rank + 1);
        //     small_mlwe_poly_mod_q = small_mlwe_this_mod[idx_in_N1 + idx_mlwe * N1];
        // } else {
        //     small_mlwe_poly_mod_q = 0;
        // }
        // // mod q
        // this_big_mlwe_poly[(modup_p_count + idx_mod)*N + idx_in_N1*mlwe_rank + i] = small_mlwe_poly_mod_q;
        // // mod p and guarantee pi > qi
        // if(small_mlwe_poly_mod_q > (qi / 2)){
        //     this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + i] = (p0 - (qi - small_mlwe_poly_mod_q)) % p0;
        // } else {
        //     this_big_mlwe_poly[(            idx_mod)*N + idx_in_N1*mlwe_rank + i] = small_mlwe_poly_mod_q;
        // }
    }
}

// repacking k mlwe -> 1 rlwe
__host__ void PCMM_Scheme::mlweCipherPacking(Ciphertext& rlwe_cipher, vector<MLWECiphertext*> mlwe_cipher_decomposed)
{
    int K = scheme.context.K;
    int L = rlwe_cipher.L;
    int N = rlwe_cipher.N;
    
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;

    int mlwe_num = mlwe_cipher_decomposed.size();
    // if(mlwe_num != mlwe_rank){
    //     cout << "mlweCipherPackingAll only support packing k mlwe -> 1 rlwe!" << endl;
    //     // return;
    // }

    if(mlwe_rank == 256){
        // Combine k mlwe_{N/k,k} -> 1 mlwe_{N,k}, then ModUp q0 -> pq0
        // small mlwe (a0,a1,...,ak-1,b) mod q0, ..., mod qi
        // big   mlwe (a0) mod p, q0, ..., qi   (a1), ... (ak-1), b

        for(int i = 0; i < mlwe_num; i++){
            repacking_cipher_pointer[i] = mlwe_cipher_decomposed[i]->cipher_device;
        }
        cudaMemcpy(repacking_cipher_pointer_device, repacking_cipher_pointer.data(), sizeof(uint64_tt*) * mlwe_num, cudaMemcpyHostToDevice);

        // embeded mlwe buffer a_0 % p, q  [a_1], ..., [a_{k-1}], b
        // dim3 mlweEmbed_dim(N1 / ringSwitch_block, ringpack_q_count - 1, mlwe_rank + 1);
        // mlweCipherEmbedding_kernel <<< mlweEmbed_dim, ringSwitch_block >>> (embeded_mlwe_buffer, repacking_cipher_pointer_device, N1, N, mlwe_rank, mlwe_num, ringpack_p_count, ringpack_q_count);

        dim3 mlweEmbed_dim(N1 / ringSwitch_block, ringpack_q_count - 1, mlwe_num);
        cudaMemset(embeded_mlwe_buffer, 0, sizeof(uint64_tt) * N * (mlwe_rank + 1) * (ringpack_pq_count - 1));
        mlweCipherEmbedding_new_kernel <<< mlweEmbed_dim, ringSwitch_block >>> (embeded_mlwe_buffer, repacking_cipher_pointer_device, N1, N, mlwe_rank, mlwe_num, ringpack_p_count, ringpack_q_count);

        context.ToNTTInplace(embeded_mlwe_buffer, 0, K - ringpack_p_count, mlwe_rank, ringpack_pq_count - 1, ringpack_pq_count - 1);
        
        // mult with repacking keys
        // output = \sum{big_mlwe.a[i] * repacking_keys[i]}
        dim3 mlweCipherPacking_dim(N / ringSwitch_block, ringpack_pq_count - 1);
        uint64_tt* mult_key_output = scheme.modUp_TtoQj_buffer;
        mlweCipherMultRepackingKey_kernel <256> <<< mlweCipherPacking_dim, ringSwitch_block >>> (mult_key_output, embeded_mlwe_buffer, repackingKeys[0]->ax_device, N1, N, ringpack_pq_count - 1);

        context.FromNTTInplace(mult_key_output, 0, K - ringpack_p_count, 2, ringpack_pq_count - 1, ringpack_pq_count - 1);

        int l = ringpack_q_count - 1 - 1;
        rlwe_cipher.l = l;
        rlwe_cipher.scale = mlwe_cipher_decomposed[0]->scale;

        // output ModDown pq -> q cipher
        dim3 modDown_dim(N / ringSwitch_block, ringpack_q_count - 1, 2);
        ringpacking_ModDown_kernel <<< modDown_dim, ringSwitch_block >>> (rlwe_cipher.cipher_device, mult_key_output, N, ringpack_p_count, ringpack_q_count, L+1, pcmm_context.p_inv_mod_qi, pcmm_context.p_inv_mod_qi_shoup);       
        // \sum{big_mlwe.a[i] * repacking_keys[i]} + (0, b)
        poly_add_batch_device(rlwe_cipher.bx_device, embeded_mlwe_buffer + (ringpack_pq_count-1)*N*mlwe_rank + ringpack_p_count*N, N, 0, 0, K, ringpack_q_count - 1);

        context.ToNTTInplace(rlwe_cipher.cipher_device, 0, K, 2, ringpack_q_count - 1, L+1);
    } else {
        cout << "mlwe rank not supported!" << endl;
    }
}


__host__ void PCMM_Scheme::PCMM_Boot(float* plain_mat, Ciphertext& rlwe_cipher, vector<MLWECiphertext*>& mlwe_cipher_buffer, int mat_M, int mat_N, int mat_K)
{
    int K = scheme.context.K;
    int L = rlwe_cipher.L;
    int N = rlwe_cipher.N;
    
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_context.ringpack_pq_count;

    int mlwe_num = mlwe_cipher_buffer.size();
    
    if(mat_K != pcmm_context.N1){
        cout << "matrix K should be equal to N1!" << endl;
        return;
    }
    if(mlwe_num < mlwe_rank / 2){
        cout << "mlwe_cipher_buffer size should be 4!" << endl;
        return;
    }
    EncodingMatrix& encodingMatrix = bootstrapper.encodingMatrix;

    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, rlwe_cipher);

    rlweCipherDecompose(rlwe_cipher, mlwe_cipher_buffer);

    PPMM(plain_mat, mlwe_cipher_buffer, mat_M, mat_N, mat_K);

    mlweCipherPacking(rlwe_cipher, mlwe_cipher_buffer);

    bootstrapper.modUpQ0toQL(rlwe_cipher);

    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, rlwe_cipher);

    bootstrapper.newResetScale(rlwe_cipher);

    scheme.conjugate_23(*bootstrapper.ctReal, rlwe_cipher);
    scheme.sub(*bootstrapper.ctImag, rlwe_cipher, *bootstrapper.ctReal);
    
    // Real part * 2
    scheme.addAndEqual(*bootstrapper.ctReal, rlwe_cipher);
    // Imag part
    scheme.divByiAndEqual(*bootstrapper.ctImag);

    bootstrapper.EvalModAndEqual(*bootstrapper.ctReal);
    bootstrapper.EvalModAndEqual(*bootstrapper.ctImag);

    scheme.mulByiAndEqual(*bootstrapper.ctImag);
    scheme.add(rlwe_cipher, *bootstrapper.ctReal, *bootstrapper.ctImag);

    scheme.multConstAndEqual(rlwe_cipher, 256./16*16);
}



#define mlweDecrypt_block 128
template<int mlwe_rank>
__global__ void mlweDecrypt_kernel(uint64_tt* mlwe_cipher_device, uint64_tt* mlwe_sk_device, uint64_tt* mlwe_plain_device, int N1, int p_num)
{
    int idx_poly = blockIdx.x * mlweDecrypt_block + threadIdx.x;
    int idx_mod = blockIdx.y;

    uint64_tt q = pqt_cons[p_num + idx_mod];
    uint128_tt mu = {pqt_mu_cons_high[p_num + idx_mod], pqt_mu_cons_low[p_num + idx_mod]};

    uint64_tt* mlwe_this_mod = mlwe_cipher_device + N1 * (mlwe_rank + 1) * idx_mod;
    uint64_tt* mlwe_this_mod_ax = mlwe_this_mod;
    uint64_tt* mlwe_this_mod_bx = mlwe_this_mod_ax + N1 * mlwe_rank;
    uint64_tt* sk_this_mod = mlwe_sk_device + idx_mod * N1 * mlwe_rank;
    
    uint128_tt acc = mlwe_this_mod_bx[idx_poly];
#pragma unroll
    for(int i = 0; i < mlwe_rank; i++)
    {
        madc_uint64_uint64_uint128(mlwe_this_mod_ax[idx_poly + i * N1], sk_this_mod[idx_poly + i * N1], acc);
    }
    singleBarrett_new(acc, q, mu);

    (mlwe_plain_device + idx_mod * N1)[idx_poly] = acc.low;
}

__host__ void PCMM_Scheme::mlweDecrypt(MLWECiphertext& mlwe_cipher, MLWESecretKey& mlwe_sk, MLWEPlaintext& mlwe_plain)
{
    int N1 = mlwe_cipher.N1;
    int l = mlwe_cipher.l;
    int p_num = context.p_num;
    int ringpack_p_count = pcmm_context.ringpack_p_count;
    int mlwe_rank = mlwe_cipher.k;
    mlwe_plain.l = mlwe_cipher.l;
    
    pcmm_context.ToNTTInplace(mlwe_cipher.cipher_device, mlwe_rank+1, l+1, 0, pcmm_context.ringpack_p_count, 1);
    dim3 mlweDecrypt_dim(N1 / mlweDecrypt_block, l + 1);
    if(mlwe_rank == 256){
        mlweDecrypt_kernel <256> <<< mlweDecrypt_dim, mlweDecrypt_block >>> (mlwe_cipher.cipher_device, mlwe_sk.sx_device + ringpack_p_count*N1*mlwe_rank, mlwe_plain.mx_device, N1, p_num);
    } else {
        cout << "mlwe rank not supported!" << endl;
    }
    pcmm_context.FromNTTInplace(mlwe_cipher.cipher_device, mlwe_rank+1, l+1, 0, pcmm_context.ringpack_p_count, 1);
}