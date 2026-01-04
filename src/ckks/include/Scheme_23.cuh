#pragma once

#include "Scheme_23.h"
#include "KeySwitch_23.cuh"
#include "poly_arithmetic.cuh"

void Scheme_23::mallocMemory()
{
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    // int gamma = context.gamma;
    int Ri_blockNum = context.Ri_blockNum;
    int Qj_blockNum = context.Qj_blockNum;
    // printf("Ri_blockNum: %d Qj_blockNum: %d\n", Ri_blockNum, Qj_blockNum);

    cudaMalloc(&ex_swk, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&sxsx, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&sx_coeff, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&axbx1_mul, sizeof(uint64_tt) * N * (L+1));
    cudaMalloc(&axax_mul, sizeof(uint64_tt) * N * (L+1));
    cudaMalloc(&bxbx_mul, sizeof(uint64_tt) * N * (L+1));

    cudaMalloc(&temp_mul, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&vx_enc, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&ex_enc, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&modUp_RitoT_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * context.dnum);
    cudaMalloc(&modUp_QjtoT_temp, sizeof(uint64_tt) * N * t_num * Qj_blockNum);
    cudaMalloc(&exProduct_T_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);
	cudaMalloc(&modUp_TtoQj_buffer, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);

    cudaMalloc(&rescale_buffer, sizeof(uint64_tt) * N * (L+1) * 2);

    random_len_for_enc = N * sizeof(uint8_tt) + N * sizeof(uint32_tt) + N * sizeof(uint32_tt);
    cudaMalloc(&in_enc, random_len_for_enc);
    
    rotKey_vec_23 = vector<Key_decomp*>(N, nullptr);
    autoKey_vec_23 = vector<Key_decomp*>(context.logN, nullptr);
    cipher_temp_pool = new Ciphertext(N, L, L, context.slots, NTL::RR(context.precision));

    add_const_copy_vec = vector<uint64_tt>((L+1) * 4);

    cudaMalloc(&add_const_buffer, sizeof(uint64_tt) * (L+1) * 4);

    cudaMalloc(&rotKey_pointer_device, sizeof(uint64_tt*));
    cudaMalloc(&rotSlots_device, sizeof(int));
}

/**
 * generates key for public encryption (key is stored in keyMap)
*/
void Scheme_23::addEncKey(SecretKey& secretKey, cudaStream_t stream)
{
    int N = context.N;
    int L = context.L;
    int K = context.K;

    // pk = (a, -as + e)
    publicKey = new Key(N, L, 0, 1);

    Sampler::uniformSampler_xq(context.randomArray_pk_device, publicKey->ax_device, N, 0, K, L+1);

    uint64_tt* ex;
    cudaMalloc(&ex, sizeof(uint64_tt) * N * (L+1));

    // only NTT on QL
    Sampler::gaussianSampler_xq(context.randomArray_e_pk_device, ex, N, 0, K, L+1);
    //context.forwardNTT_batch(ex, 0, K, 1, L+1);
    context.ToNTTInplace(ex, 0, K, 1, L+1,L+1);

    barrett_batch_3param_device(publicKey->bx_device, publicKey->ax_device, secretKey.sx_device, N, 0, 0, K, K, L+1);
    poly_sub2_batch_device(ex, publicKey->bx_device, N, 0, 0, K, L+1);

	cudaFree(ex);
}

void Scheme_23::encryptZero(Ciphertext& cipher, int l, int slots, cudaStream_t stream)
{
    int N = context.N;
    int L = context.L;
    int K = context.K;

    RNG::generateRandom_device(in_enc, random_len_for_enc);
    
    Sampler::ZOSampler(in_enc, vx_enc, N, 0.5, 0, K, l+1);
    context.ToNTTInplace(vx_enc, 0, K, 1, l+1, L+1);

    barrett_batch_3param_device(cipher.ax_device, vx_enc, publicKey->ax_device, N, 0, 0, 0, K, l+1);

    Sampler::gaussianSampler_xq(in_enc + N * sizeof(uint8_tt), ex_enc, N, 0, K, l+1);
    context.ToNTTInplace(ex_enc, 0, K, 1, l+1, L+1);

    poly_add_batch_device(cipher.ax_device, ex_enc, N, 0, 0, K, l+1);
    barrett_batch_3param_device(cipher.bx_device, vx_enc, publicKey->bx_device, N, 0, 0, 0, K, l+1);

    Sampler::gaussianSampler_xq(in_enc + N * sizeof(uint8_tt) + N * sizeof(uint32_tt), ex_enc, N, 0, K, l+1);
    context.ToNTTInplace(ex_enc, 0, K, 1, l+1, L+1);

    poly_add_batch_device(cipher.bx_device, ex_enc, N, 0, 0, K, l+1);
}


void Scheme_23::encryptMsg(Ciphertext& cipher, Plaintext& message, cudaStream_t stream)
{
    cipher.l = message.l;
    cipher.scale = message.scale;
    encryptZero(cipher, message.l, message.slots, stream);
    poly_add_batch_device(cipher.bx_device, message.mx_device, context.N, 0, 0, context.K, message.l+1);
}

void Scheme_23::decryptMsg(Plaintext& plain, SecretKey& secretKey, Ciphertext& cipher, cudaStream_t stream)
{
    int N = context.N;
    int l = cipher.l;
    int K = context.K;
    plain.l = cipher.l;
    plain.scale = cipher.scale;
    plain.slots = cipher.slots;

    cudaMemset(plain.mx_device + (plain.l+1)*N, 0, sizeof(uint64_tt) * N * (context.L-plain.l));
    barrett_batch_3param_device(plain.mx_device, cipher.ax_device, secretKey.sx_device, N, 0, 0, K, K, l+1);
    poly_add_batch_device(plain.mx_device, cipher.bx_device, N, 0, 0, K, l+1);
}

// Homomorphic Addition
void Scheme_23::add(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    int K = context.K;
    int slots = cipher1.slots;
    cipher_res.l = l;
    cipher_res.scale = cipher1.scale;

    cipher_add_3param_batch_device(cipher_res.cipher_device, cipher1.cipher_device, cipher2.cipher_device, N, K, l+1, L+1);
}

void Scheme_23::addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    int K = context.K;
    cipher1.l = l;

    cipher_add_batch_device(cipher1.cipher_device, cipher2.cipher_device, N, K, l+1, L+1);
}

void Scheme_23::addConstAndEqual(Ciphertext& cipher, Plaintext& cnst)
{
    int l = min(cipher.l, cnst.l);
    cipher.l = l;
    poly_add_batch_device(cipher.bx_device, cnst.mx_device, context.N, 0, 0, context.K, cnst.l+1);
}

void Scheme_23::addConstAndEqual(Ciphertext& cipher, double cnst)
{
    NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst));
    if(scaled_real == 0) return;
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    poly_add_real_const_batch_device(cipher.bx_device, add_const_buffer, context.N, 0, context.K, cipher.l+1);
}

void Scheme_23::addConstAndEqual(Ciphertext& cipher, cuDoubleComplex cnst)
{
    int L = context.L;
    NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst.x));
    NTL::ZZ scaled_imag = to_ZZ(round(cipher.scale * cnst.y));
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
        add_const_copy_vec[i + L+1] = scaled_imag % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    context.poly_add_complex_const_batch_device(cipher.bx_device, add_const_buffer, 0, context.K, cipher.l+1);
}

void Scheme_23::multConstAndEqual(Ciphertext& cipher, Plaintext& cnst)
{
    int L = context.L;
    int N = context.N;
    int K = context.K;
    if(cipher.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }

    barrett_2batch_device(cipher.cipher_device, cnst.mx_device, N, 0, 0, K, cipher.l+1, L+1);
    // barrett_batch_device(cipher.ax_device, cnst.mx_device, N, 0, 0, K, cipher.l+1);
    // barrett_batch_device(cipher.bx_device, cnst.mx_device, N, 0, 0, K, cipher.l+1);

    cipher.scale *= cnst.scale;
}


// can't mult negative real number
void Scheme_23::multConstAndEqual(Ciphertext& cipher, double cnst)
{
    int L = context.L;
    if(int(cnst) == 1){
        return;
    }
    if(int(cnst) == -1)
    {
        negateAndEqual(cipher);
        return;
    }
    if(cipher.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }

    NTL::ZZ scaled_cnst;

    if(abs(int(cnst) - cnst) < 1e-6){
        scaled_cnst = NTL::ZZ(cnst);
    } else {
        scaled_cnst = NTL::ZZ(context.qVec[cipher.l] * cnst);
    }
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_cnst % context.qVec[i];
        add_const_copy_vec[i + 2*(L+1)] = x_Shoup(add_const_copy_vec[i], context.qVec[i]);
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    context.poly_mul_const_batch_device(cipher.ax_device, add_const_buffer, context.K, cipher.l+1);

    if(abs(int(cnst) - cnst) >= 1e-6){
        cipher.scale *= context.qVec[cipher.l];
    }
}

// c1 += c2*cnst
// c1.scale = c2.scale * target_scale
void Scheme_23::multConstAndAddCipherEqual(Ciphertext& c1, Ciphertext& c2, double cnst, NTL::RR target_scale)
{
    if(c2.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }
    int L = context.L;
    c1.l = min(c1.l, c2.l);

    NTL::ZZ scaled_cnst = to_ZZ(round(target_scale * cnst));
    // printf("evalmod mult const: ");
    // cout<<"scaled_cnst: "<<target_scale * cnst<<"   "<<scaled_cnst<<endl;
    if(scaled_cnst == 0) {
        cudaMemset(c1.cipher_device, 0, sizeof(uint64_tt) * context.N * (context.L+1) * 2);
        return ;
    }

    for(int i = 0; i < c1.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_cnst % context.qVec[i];
        add_const_copy_vec[i + 2*(L+1)] = x_Shoup(add_const_copy_vec[i], context.qVec[i]);
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    context.poly_mul_const_add_cipher_batch_device(c1.cipher_device, c2.cipher_device, add_const_buffer, to_long(target_scale), context.K, c1.l+1);
}

// Homomorphic Substraction
void Scheme_23::sub(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    cipher_res.l = cipher1.l;
    cipher_res.scale = cipher1.scale;
    int K = context.K;
    int slots = cipher1.slots;

    poly_sub_3param_batch_device(cipher_res.ax_device, cipher1.ax_device, cipher2.ax_device, N, 0, 0, 0, K, l+1);
    poly_sub_3param_batch_device(cipher_res.bx_device, cipher1.bx_device, cipher2.bx_device, N, 0, 0, 0, K, l+1);
}

void Scheme_23::subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int K = context.K;

    poly_sub_batch_device(cipher1.ax_device, cipher2.ax_device, N, 0, 0, K, l+1);
    poly_sub_batch_device(cipher1.bx_device, cipher2.bx_device, N, 0, 0, K, l+1);
}

void Scheme_23::constSub(Ciphertext& cipher_res, Ciphertext& cipher, double cnst)
{
    cipher_res.scale = cipher.scale;
    cipher_res.l = cipher.l;
    NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst));
    if(scaled_real == 0) return;
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    poly_real_const_sub_3param_batch_device(cipher_res.cipher_device, cipher.cipher_device, add_const_buffer, context.N, context.L+1, context.K, cipher.l+1);
}

void Scheme_23::constSubAndEqual(Ciphertext& cipher, double cnst)
{
    NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst));
    if(scaled_real == 0) return;
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = scaled_real % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    poly_real_const_sub_batch_device(cipher.cipher_device, add_const_buffer, context.N, context.L+1, context.K, cipher.l+1);
}

#define rescale_block 256

__global__ void rescaleAndEqual_kernel(uint64_tt* device_a, uint32_tt n, int p_num, int q_num, int l, uint64_tt* qiInvVecModql_device, uint64_tt* qiInvVecModql_shoup_device)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    register int idx = idx_in_poly + (idx_in_cipher * q_num + idx_in_pq) * n;

    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
	register uint128_tt mu_q(pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]);

	register uint64_tt ra = device_a[idx] + 2*q - device_a[(idx_in_cipher*q_num + l) * n + idx_in_poly];

	csub_q(ra, q);
	register uint64_tt ql_inv = qiInvVecModql_device[idx_in_pq];
	register uint64_tt ql_inv_shoup = qiInvVecModql_shoup_device[idx_in_pq];	
	device_a[idx] = mulMod_shoup(ra, ql_inv, ql_inv_shoup, q);
}

__global__ void rescaleAndEqual_new_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int p_num, int q_num, int l, uint64_tt* qiInvVecModql_device, uint64_tt* qiInvVecModql_shoup_device)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    register int idx = idx_in_poly + (idx_in_cipher * q_num + idx_in_pq) * n;

    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
	register uint128_tt mu_q(pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]);

	register uint64_tt ra = device_a[idx] + 2*q - device_b[idx];
	csub_q(ra, q);
	register uint64_tt ql_inv = qiInvVecModql_device[idx_in_pq];
	register uint64_tt ql_inv_shoup = qiInvVecModql_shoup_device[idx_in_pq];	
	device_a[idx] = mulMod_shoup(ra, ql_inv, ql_inv_shoup, q);
}

__global__ void mod_to_qi(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int p_num, int q_num, int l)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    
    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
    register uint64_tt rb = device_b[idx_in_poly + (idx_in_cipher*q_num + l)*n];
    barrett_reduce_uint64_uint64(rb, q, pqt_mu_cons_high[p_num + idx_in_pq]);
    device_a[idx_in_poly + (idx_in_cipher*q_num + idx_in_pq)*n] = rb;
}

void Scheme_23::rescaleAndEqual(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int L = cipher.L;

    context.FromNTTInplace(cipher.cipher_device, cipher.l, K+cipher.l, 2, 1, L+1);
    dim3 resc_dim(N / rescale_block, cipher.l, 2);
    mod_to_qi <<< resc_dim, rescale_block >>> (rescale_buffer, cipher.cipher_device, N, K, L+1, cipher.l);

    context.ToNTTInplace(rescale_buffer, 0, K, 2, cipher.l, L+1);
    rescaleAndEqual_new_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, rescale_buffer, N, K, L+1, cipher.l, context.qiInvVecModql_device + cipher.l*(cipher.l-1)/2, context.qiInvVecModql_shoup_device + cipher.l*(cipher.l-1)/2);
    cipher.scale = cipher.scale / context.qVec[cipher.l];
    cipher.l -= 1;

    // cudaMemset(cipher.ax_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
    // cudaMemset(cipher.bx_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
}

void Scheme_23::rescaleAndEqual(Ciphertext& cipher, NTL::RR target_scale)
{
    int N = context.N;
    int K = context.K;
    int L = cipher.L;

    context.FromNTTInplace(cipher.cipher_device, cipher.l, K+cipher.l, 2, 1, L+1);
    dim3 resc_dim(N / rescale_block, cipher.l, 2);
    mod_to_qi <<< resc_dim, rescale_block >>> (rescale_buffer, cipher.cipher_device, N, K, L+1, cipher.l);

    context.ToNTTInplace(rescale_buffer, 0, K, 2, cipher.l, L+1);
    rescaleAndEqual_new_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, rescale_buffer, N, K, L+1, cipher.l, context.qiInvVecModql_device + cipher.l*(cipher.l-1)/2, context.qiInvVecModql_shoup_device + cipher.l*(cipher.l-1)/2);
    cipher.scale = cipher.scale / context.qVec[cipher.l];
    cipher.l -= 1;
    // cipher.scale = target_scale;

    // cudaMemset(cipher.ax_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
    // cudaMemset(cipher.bx_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
}

void Scheme_23::rescaleToAndEqual(Ciphertext& cipher, int level)
{
    int N = context.N;
    int K = context.K;
    int L = context.L;
    int last_l = cipher.l;

    context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    while(cipher.l > level)
    {
        int l = cipher.l;
	    dim3 resc_dim(N / rescale_block, l, 2);
	    rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, l, context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
        cipher.l -= 1;
        cipher.scale = cipher.scale / context.qVec[l];
        cout<<"cipher.level: "<<cipher.l<<"  cipher.scale: "<<cipher.scale<<endl;
    }
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}

void Scheme_23::rescaleAndEqual_noNTT(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int l = cipher.l;
    int L = cipher.L;

	//context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    dim3 resc_dim(N / rescale_block, l, 2);
	rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, l, context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
    cipher.l -= 1;
    cipher.scale = cipher.scale / context.qVec[l];
    //context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}

// Homomorphic Negation
void Scheme_23::negate(Ciphertext& cipher_res, Ciphertext& cipher)
{
    cipher_negate_3param_batch_device(cipher.cipher_device, cipher_res.cipher_device, context.N, cipher.L+1, context.K, cipher.l+1);
}

void Scheme_23::negateAndEqual(Ciphertext& cipher)
{
    cipher_negate_batch_device(cipher.cipher_device, context.N, cipher.L+1, context.K, cipher.l+1);
}

void Scheme_23::divByiAndEqual(Ciphertext& cipher)
{
    context.divByiAndEqual(cipher.cipher_device, context.K, cipher.l+1);
}

void Scheme_23::mulByiAndEqual(Ciphertext& cipher)
{
    context.mulByiAndEqual(cipher.cipher_device, context.K, cipher.l+1);
}

__global__ void divByPo2_device_kernel(uint64_tt* device_c, uint32_tt n, int q_num)
{
    register uint32_tt index = blockIdx.y;
	register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;
    register uint64_tt ra = device_c[i + idx_in_cipher * q_num * n] >> 1;
    device_c[i + idx_in_cipher * q_num * n] = ra;
}

void Scheme_23::divByPo2AndEqual(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int l = cipher.l;
    int L = context.L;
    dim3 div_dim(N / poly_block , l, 2);
    divByPo2_device_kernel<<< div_dim, poly_block >>>(cipher.cipher_device, N, L+1);
}

void Scheme_23::decrypt_display(SecretKey& sk, Ciphertext& cipher, char* s, int row_num)
{
    int N = context.N;
    int K = context.K;
    int l = cipher.l;
    int L = context.L;
    int slots = context.slots;
    Plaintext plain(N, L, L, context.slots, NTL::RR(context.precision));
    decryptMsg(plain, sk, cipher);

    context.decode(plain, context.encode_buffer);
    cout<<s<<" cipher scale: "<<cipher.scale<<endl;
    // print_device_array(context.encode_buffer, cipher.slots, s);
    cuDoubleComplex* array_PQ = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, context.encode_buffer, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    printf("%s = [", s);
    double max_value = 0;
    for(int i = 0; i < slots; i++)
    {
        if (i < 8)printf("%.8lf + %.8f*i, ", array_PQ[i].x, array_PQ[i].y);
        // if(i % row_num == row_num - 1) printf("\n");
        max_value = max(max_value, array_PQ[i].x);
    }
    cout<<"] max: "<<max_value<<endl;
    delete array_PQ;
}