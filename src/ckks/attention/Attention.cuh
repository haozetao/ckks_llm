#pragma once

#include "Attention.h"
#include "../include/advanced/SchemeAlgo.cuh"

Attention::Attention(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, int token_len, int head_num, int d)
    : context(context), scheme(scheme), scheme_algo(scheme_algo), token_len(token_len), head_num(head_num), d(d)
{
    // exp(5(x-1)) 
    // x \in [-1, 1] -> [-10, 0]
    exp_cheby_coeffs = {0.18354092959380028, 0.32794453388908545, 0.23590404563197528, 0.13922148455874572, 
        0.0688382641622435, 0.029080636255965363, 0.010676991703074352, 0.003456418101541191, 
        0.0009990240054485726, 0.0002603107017095204, 6.204105467877648e-05, 1.391652806979088e-05};
    int exp_tree_node_num = 1<<int(ceil(log2(exp_cheby_coeffs.size())));
    for(int i = 0; i < exp_tree_node_num; i++)
    {
        exp_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    exp_cheby_poly_pool[1]->coeffs = exp_cheby_coeffs;
    exp_cheby_poly_pool[1]->maxDegree = exp_cheby_coeffs.size() - 1;

    {
        int degree = exp_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, exp_cheby_poly_pool);
    }


    sigmoid_cheby_coeffs = {0.5, 0.5876811235265726, -2.007269726835591e-17, -0.12149623872303088, 
        -2.0072697268355913e-17, 0.035317888765259355, -1.0917210757920756e-16, -0.010693712782924763, 
        -1.8652562430325655e-16, 0.003260293428351401, -2.2019441004172328e-16, -0.0009931752451993756, 
        -9.895698703494615e-18, 0.00029550137814745605, -1.761892554953261e-17, -6.453127146099342e-05};
    int sigmoid_tree_node_num = 1<<int(ceil(log2(sigmoid_cheby_coeffs.size())));
    for(int i = 0; i < sigmoid_tree_node_num; i++)
    {
        sigmoid_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    sigmoid_cheby_poly_pool[1]->coeffs = sigmoid_cheby_coeffs;
    sigmoid_cheby_poly_pool[1]->maxDegree = sigmoid_cheby_coeffs.size() - 1;
    {
        int degree = sigmoid_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, sigmoid_cheby_poly_pool);
    }
    
    CDF_cheby_coeffs = {0.5, 0.6234577610429392, 8.07567900971873e-17, -0.17602452170600366, 
        1.1776422425135919e-16, 0.0761747991351066, -1.2929493435476463e-16, -0.03371833834600424, 
        -1.6116422569495642e-16, 0.014111526810286363, -2.0330980266310844e-16, -0.005444024438389357, 
        -1.224401966956653e-16, 0.0018840923558029613, -4.5685761667957374e-17, -0.0004415815053097376};
    int CDF_tree_node_num = 1<<int(ceil(log2(CDF_cheby_coeffs.size())));
    for(int i = 0; i < CDF_tree_node_num; i++)
    {
        CDF_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    CDF_cheby_poly_pool[1]->coeffs = CDF_cheby_coeffs;
    CDF_cheby_poly_pool[1]->maxDegree = CDF_cheby_coeffs.size() - 1;
    {
        int degree = CDF_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, CDF_cheby_poly_pool);
    }
    
    softmax_x_max = 5;


    /**********************************************prepare mask********************************************/
    int N = context.N;
    int L = context.L;
    int slots = context.slots;
    int column_num = slots / token_len;
    cuDoubleComplex* column_mask_buffer_host = new cuDoubleComplex[slots];
    cuDoubleComplex* column_mask_buffer_device;
    cudaMalloc(&column_mask_buffer_device, sizeof(cuDoubleComplex) * slots);

    for(int i = 0; i < 2; i++){
        Plaintext* mask_i = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
        column_mask.push_back(mask_i);

        vector<int> mask_idx;
        for(int t = 0; t < column_num / d; t++){
            mask_idx.push_back(t * d + i);
        }

        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, *mask_i);
    }
    delete column_mask_buffer_host;
    cudaFree(column_mask_buffer_device);
    
    
    /******************************************Inv and Square Inv*********************************************/
    K_inv = {1.708209458521407, 1.3347202619383138, 1.0593437148393925, 1.0017626264306325, 1.0000034823730521};
}

void Attention::prepareMask(cuDoubleComplex* mask_host, vector<int> idx_vector)
{
    int slots = context.slots;
    // memset(mask_host, 0, sizeof(cuDoubleComplex));
    // for(int i = 0; i < slots; i++) mask_host[i].x = mask_host[i].y = 0;

    for(auto idx : idx_vector){
        for(int i = 0; i < token_len; i++){
            int column_num = slots / token_len;
            mask_host[i * column_num + idx].x = 1;
        }
    }
}

void Attention::addKey(SecretKey& sk)
{
    for(int i = 0; i < log(d) + 1; i++){
        scheme.addLeftRotKey_23(sk, 1 << i);
        printf("%d ", 1<<i);
        scheme.addLeftRotKey_23(sk, 32768 - (1 << i));
        printf("%d ", 32768 - (1 << i));
    }
    cout<<endl;
}


void Attention::evalExp(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 0.2);
    scheme.rescaleAndEqual(cipher);
    scheme.addConstAndEqual(cipher, 1);

    NTL::RR target_scale = cipher.scale;

    // First mapping: x \in [-10, 0] -> [-1, 1]
    // Evaluate the Chebyshev polynomial for exp(5(x-1))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, exp_cheby_poly_pool);
}

// CDF function: 0.5 * (1 + erf(x / sqrt(2)))
void Attention::evalCDF(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 1./softmax_x_max);
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, CDF_cheby_poly_pool);
}

// Sigmoid function: exp(x) / (1 + exp(x))
void Attention::evalSigmoid(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 0.2);
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, sigmoid_cheby_poly_pool);
}

void Attention::evalGeLU(Ciphertext& cipher)
{
    Ciphertext* mult_buffer = scheme_algo.chebyshev_tree_pool[0];
    *mult_buffer = cipher;
    evalCDF(cipher);
    scheme.multAndEqual_23(cipher, *mult_buffer);
    scheme.rescaleAndEqual(cipher);
}

void Attention::evalSiLU(Ciphertext& cipher)
{
    Ciphertext* mult_buffer = scheme_algo.chebyshev_tree_pool[0];
    *mult_buffer = cipher;
    evalSigmoid(cipher);
    scheme.multAndEqual_23(cipher, *mult_buffer);
    scheme.rescaleAndEqual(cipher);
}

// only in [0, 1]
void Attention::evalInv(Ciphertext& cipher, SecretKey& sk, double upper_bound)
{
    // scale cipher into [-1,1]
    double scale = 1.0 / upper_bound;
    scheme.multConstAndEqual(cipher, scale);
    if(abs(scale - int(scale)) >= 1e-6) scheme.rescaleAndEqual(cipher);
    

    Ciphertext* bx = scheme_algo.chebyshev_tree_pool[0];
    Ciphertext* ax = &cipher;
    Ciphertext* temp = scheme_algo.chebyshev_tree_pool[1];
    temp->l = cipher.l;
    temp->scale = cipher.scale;
    bx->l = cipher.l;
    bx->scale = cipher.scale;

    NTL::RR target_scale = cipher.scale;

    
    // b = (2 - k0*a)
    ax->scale = ax->scale / K_inv[0];
    scheme.constSub(*bx, *ax, 2);
    ax->scale = ax->scale * K_inv[0];
    // a = a * (2 - k0*a)
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);
    // a = k0 * a * (2 - k0*a)
    ax->scale = ax->scale / K_inv[0];
    // b = k0 * b * (2 - k0*a)
    bx->scale = bx->scale / K_inv[0];
    // cout<<"a.scale: "<<ax->scale<<"  b.scale: "<<bx->scale<<endl;

    NTL::RR threshold(pow(double(context.precision), 1.75));
    cout<<"threshold: "<<threshold<<endl;


    for(int i = 1; i < K_inv.size(); i++){

        // temp = (2 - k0*a)
        ax->scale = ax->scale / K_inv[i];
        scheme.constSub(*temp, *ax, 2);
        ax->scale = ax->scale * K_inv[i];

        // a = a * (2 - k0*a)
        scheme.multAndEqual_23(*ax, *temp);
        if(ax->scale > threshold){
            scheme.rescaleAndEqual(*ax);
            // cout<<"111"<<endl;
        }
        // a = k0 * a * (2 - k0*a)
        ax->scale = ax->scale / K_inv[i];

        // b = k0 * b * (2 - k0*a)
        scheme.multAndEqual_23(*bx, *temp);
        if(bx->scale > threshold){
            scheme.rescaleAndEqual(*bx);
            // cout<<"222"<<endl;
        }
        bx->scale = bx->scale / K_inv[i];

        // cout<<"a.scale: "<<ax->scale<<"  b.scale: "<<bx->scale<<endl;

        // scheme.decrypt_display(sk, *temp, "temp");
        // scheme.decrypt_display(sk, *ax, "ax");
        // scheme.decrypt_display(sk, *bx, "bx");
    }
    // while(bx->scale > target_scale*target_scale){
    //     scheme.rescaleAndEqual(*bx);
    // }
    // NTL::RR scale = bx->scale / target_scale;
    
    // scheme.multConstAndEqual(, scale);
    cipher = *bx;
}

void Attention::evalSoftMax(Ciphertext& cipher)
{
    // e^(x_i - x_max)
    scheme.addConstAndEqual(cipher, -softmax_x_max);
    evalExp(cipher);

    // reduce sum and repeat
    // reduce sum
    int n = token_len;
    for(int i = log(n) + 1; i >= 0; i--){
        scheme.leftRotateAddSelf_23(cipher, 1 << i);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    scheme.multConstAndEqual(cipher, *column_mask[0]);
    scheme.rescaleAndEqual(cipher);

    // repeate
    for(int i = 0; i < log(n) + 1; i++){
        scheme.leftRotateAddSelf_23(cipher, 32768 - (1 << i));
    }
}

// SoftMax function: exp(x_i) / sum(exp(x_j))
void Attention::LayerNorm(Ciphertext& cipher)
{

}