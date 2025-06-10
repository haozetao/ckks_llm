#pragma once

#include "Bootstrapper.h"
// #include "precompute.h"
#include "Bootstrapping.cuh"
#include "../Utils.cuh"

Bootstrapper::Bootstrapper(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, SecretKey& secretkey, EncodingMatrix& encodingMatrix, int is_STC_first, int logradix)
    :context(context), scheme(scheme), scheme_algo(scheme_algo), logradix(logradix), encodingMatrix(encodingMatrix), is_STC_first(is_STC_first)
{
    N = context.N;
	slots = context.slots;
	logN = context.logN;
	logslots = context.logslots;
    q_num = context.q_num;
    p_num = context.p_num;
    t_num = context.t_num;
    maxLevel = context.L;
    precision = context.precision;
    radix = 1<<logradix;

    // printf("logN: %d logslots: %d\n", logN, logslots);

    for(int i = 0; i < q_num; i++)
    {
        uint64_tt q = context.qVec[i];
        inv_Ndiv2slots.push_back(modinv128(1<<(logN-logslots-1), q));
    }
    cudaMalloc(&inv_Ndiv2slots_device, sizeof(uint64_tt) * q_num);
    cudaMemcpy(inv_Ndiv2slots_device, inv_Ndiv2slots.data(), sizeof(uint64_tt)*inv_Ndiv2slots.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&diag_idx_buffer, sizeof(cuDoubleComplex) * N);

    ctReal = new Ciphertext(N, maxLevel, maxLevel, slots, NTL::RR(precision));
    ctImag = new Ciphertext(N, maxLevel, maxLevel, slots, NTL::RR(precision));

    message_ratio = 4.0;

    c2s_cost_level = ceil(double(logslots) / logradix);
    s2c_cost_level = c2s_cost_level;

    sine_cost_level = log2(context.eval_sine_chebyshev_coeff.size());

    if(context.h == 64){
        double_angle_cost_level = context.double_angle_cost_level;
    } else if(context.h == 192){
        double_angle_cost_level = context.double_angle_cost_level;
    }

    bootstrapping_cost_level = 0;
    bootstrapping_cost_level += c2s_cost_level + sine_cost_level + double_angle_cost_level + s2c_cost_level;

    scheme_algo.malloc_bsgs_buffer(context.eval_sine_chebyshev_coeff.size());
    prepare_sine_chebyshev_poly();
}


// sine <- cosine + double angle
void Bootstrapper::prepare_sine_chebyshev_poly()
{
    if(context.h == 64){
        eval_sine_K = 12;
    } else if(context.h == 192){
        eval_sine_K = 25;
    }

    sine_factor = pow(2, double_angle_cost_level);

    eval_sine_scaling_factor = context.precision;

    sine_A = -eval_sine_K/sine_factor;
    sine_B = -sine_A;

    qDiff = NTL::RR(context.qVec[0]);
    qDiff = qDiff / pow(2, to_double(round(log(qDiff) / log(2))));
    // cout<<"qDiff: "<<qDiff<<endl;
    // sqrt2Pi = NTL::pow(1/20/M_PI*qDiff, NTL::RR(1.0 / sine_factor));
    sqrt2Pi = NTL::pow(1*qDiff/2/M_PI, NTL::RR(1.0 / sine_factor));
    // cout<<"sqrt2Pi: "<<sqrt2Pi<<endl;

/**************************************************chebyshev coeff tree***************************************************/
    eval_sine_chebyshev_coeff = context.eval_sine_chebyshev_coeff;
    int sine_tree_node_num = 1<<int(ceil(log2(eval_sine_chebyshev_coeff.size())));
    for(int i = 0; i < sine_tree_node_num; i++)
    {
        eval_sine_chebyshev_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    eval_sine_chebyshev_poly_pool[1]->coeffs = eval_sine_chebyshev_coeff;
    eval_sine_chebyshev_poly_pool[1]->maxDegree = eval_sine_chebyshev_coeff.size() - 1;

    {
        int degree = eval_sine_chebyshev_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, eval_sine_chebyshev_poly_pool);
    }
}

void Bootstrapper::addBootstrappingKey(SecretKey& secretKey, cudaStream_t stream)
{
    encodingMatrix.addBootstrappingKey();
    scheme.addConjKey_23(secretKey, stream);
    scheme.addMultKey_23(secretKey, stream);
    boot_key_flag = 1;
}
void Bootstrapper::Bootstrapping(Ciphertext& cipher)
{
    if(is_STC_first){
        FirstSTCBootstrapping(cipher);
    } else {
        FirstModUpBootstrapping(cipher);
    }
}
void Bootstrapper::FirstSTCBootstrapping(Ciphertext& cipher)
{
    if(cipher.l != encodingMatrix.is_sqrt_rescale * encodingMatrix.rescale_times){
        cout<<"bootstrapping cipher not on level0!!!"<<endl;
    }
    if(boot_key_flag == 0){
        throw invalid_argument("bootstrapping key not generate OK!!!");
    }

    int L = context.L;

    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, cipher);

    // scheme.decrypt_display(scheme_algo.secretkey, cipher, "ctReal after stc ");
	// Step 1: scale to q0/|m|
    // q0 / message_ratio = q0 / 4.0 == input.scale
	// Step 2 : Extend the basis from q to Q
    modUpQ0toQL(cipher);

    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, cipher);

    // newResetScale(cipher);

    scheme.conjugate_23(*ctReal, cipher);
    
    // Real part * 2
    scheme.addAndEqual(cipher, *ctReal);
    // // c1.l -= 4;
    EvalModAndEqual(cipher);

    scheme.multConstAndEqual(cipher, 256./16*16);
}

void Bootstrapper::newResetScale(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 1./eval_sine_K/2);
    scheme.rescaleAndEqual(cipher);
}

void Bootstrapper::FirstModUpBootstrapping(Ciphertext& cipher)
{
    if(cipher.l != 0){
        cout<<"bootstrapping cipher not on level0!!!"<<endl;
    }
    if(boot_key_flag == 0){
        throw invalid_argument("bootstrapping key not generate OK!!!");
    }

    int L = context.L;
	// Step 1: scale to q0/|m|
    // q0 / message_ratio = q0 / 4.0 == input.scale
	// Step 2 : Extend the basis from q to Q
    modUpQ0toQL(cipher);

    cipher.scale /= 1./ N;

    coeffToSlot(cipher, *ctReal, *ctImag);
    
    resetScale(*ctReal);
    resetScale(*ctImag);

    EvalModAndEqual(*ctReal);
    EvalModAndEqual(*ctImag);

    slotToCoeff(cipher, *ctReal, *ctImag);
}

void Bootstrapper::resetScale(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 1./ eval_sine_K /N);
    cipher.scale /= N;
    scheme.rescaleAndEqual(cipher);
}


// modUp then scaleUp
void Bootstrapper::modUpQ0toQL(Ciphertext& cipher)
{
    if(cipher.l != 0) return;
    int N = context.N;
    int L = context.L;
    int K = context.K;

    context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);

    // print_device_array(cipher.cipher_device, N, 2*q_num, "cx1");
    dim3 modUpQ0toQL_dim(N / modUpQ0toQL_block, q_num-1, 2);
	modUpQ0toQL_kernel <<< modUpQ0toQL_dim, modUpQ0toQL_block >>> (cipher.cipher_device, N, p_num, q_num, context.qVec[0] / context.precision);
    // print_device_array(cipher.cipher_device, N, 2*q_num, "cx2");

    cipher.l = L;
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    // cout<<"cipher.scale: "<<cipher.scale<<endl;
}

void Bootstrapper::subAndSum(Ciphertext& cipher)
{
    int N = context.N;
    int L = context.L;
    int K = context.K;

    int logslots = log2(cipher.slots);
    int logN = log2(cipher.N);

    int gap = 1 << (logN - logslots - 1);
    if(logslots == 0)
    {
        gap <<= 1;
    }

    if(gap > 1)
    {
        // print_device_array(cipher.cipher_device, N, 2*q_num, "cx3");
        dim3 modUpQ0toQL_dim(N / mulNdivslots_block, q_num, 2);
        mulInvNdiv2slots_kernel <<< modUpQ0toQL_dim, mulNdivslots_block >>> (cipher.cipher_device, N, p_num, q_num, inv_Ndiv2slots_device);
        // print_device_array(cipher.cipher_device, N, 2*q_num, "cx4");

        //context.ToNTTInplace(cipher.cipher_device, 0, K, 2, L+1, L+1);

        for(int i = logN; i > logslots + 1; i--)
        {
            printf("X->X^({2^%d}+1)\n", i);
            // If you need to perform this operation on the NTT field, you need to add the ntt transformation on top and the intt transformation to the CTS
            scheme.automorphismAndAdd(cipher, i);
        }
        printf("gap > 1\n");
    }
    else
    {
        //context.ToNTTInplace(cipher.cipher_device, 0, K, 2, L+1, L+1);

        // printf("gap == 1\n");
    }
}

void Bootstrapper::PrefetchRotKeys(vector<int> rotIdx)
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    cout<<"device_id: "<<device_id<<endl;

    cout<<"Prefetch Rot Keys.size(): "<<rotIdx.size()<<endl;
    for(auto i : rotIdx){
        cout<<i<<", ";
        Key_decomp* key = scheme.rotKey_vec_23[i];
        cudaMemPrefetchAsync(scheme.rotKey_vec_23[i]->cipher_device, sizeof(uint64_tt) * N * t_num * key->blockNum * key->dnum * 2, device_id, encodingMatrix.stream_prefetch);
    }
    cout<<endl;
}

void Bootstrapper::coeffToSlot(Ciphertext& cipher, Ciphertext& cipherReal, Ciphertext& cipherImag)
{
    // PrefetchRotKeys(encodingMatrix.rotIdx_C2S);
    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, cipher);
    
    scheme.conjugate_23(cipherReal, cipher);
    scheme.sub(cipherImag, cipher, cipherReal);
    
    // Real part * 2
    scheme.addAndEqual(cipherReal, cipher);
    // Imag part
    scheme.divByiAndEqual(cipherImag);
}

void Bootstrapper::slotToCoeff(Ciphertext& cipher, Ciphertext& cipherReal, Ciphertext& cipherImag)
{
    // PrefetchRotKeys(encodingMatrix.rotIdx_S2C);
    scheme.mulByiAndEqual(cipherImag);
    scheme.add(cipher, cipherReal, cipherImag);

    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, cipher);
}

void Bootstrapper::EvalModAndEqual(Ciphertext& cipher)
{
    NTL::RR target_scale = cipher.scale;
    for(int i = 0; i < double_angle_cost_level; i++)
    {
        target_scale = NTL::sqrt(target_scale * context.qVec[i + q_num - c2s_cost_level - sine_cost_level - double_angle_cost_level]);
        // cout<<"Qi: "<<context.qVec[i + q_num - c2s_cost_level - sine_cost_level - double_angle_cost_level + 1]<<endl;
    }

    scheme.addConstAndEqual(cipher, -0.5/(sine_factor * (sine_B - sine_A)));

    // chebyshev
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, eval_sine_chebyshev_poly_pool);

    double sqrt2Pi_temp = to_double(sqrt2Pi);
	// Double angle
    for(int i = 0; i < double_angle_cost_level; i++)
    {
        sqrt2Pi_temp *= sqrt2Pi_temp;
        scheme.squareAndEqual_double_add_const_rescale(cipher, -sqrt2Pi_temp);
    }
}  
