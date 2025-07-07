#pragma once

#include "../include/advanced/SchemeAlgo.cuh"
#include "../include/Context_23.cuh"
#include "../include/Scheme_23.cuh"


class Attention {
public:
    Context_23& context;
    Scheme_23& scheme;
    SchemeAlgo& scheme_algo;
    
    Attention(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, int token_len, int head_num, int d);

    
    vector<double> exp_cheby_coeffs;
    vector<Chebyshev_Polynomial*> exp_cheby_poly_pool;
    
    vector<double> sigmoid_cheby_coeffs;
    vector<Chebyshev_Polynomial*> sigmoid_cheby_poly_pool;
    
    vector<double> CDF_cheby_coeffs;
    vector<Chebyshev_Polynomial*> CDF_cheby_poly_pool;

    int head_num;
    int d;
    int token_len;
    double softmax_x_max;
    double activate_x_max;

    vector<Plaintext*> column_mask_reduce;
    vector<Plaintext*> column_mask_ccmm;
    void prepareMask(cuDoubleComplex* mask_host, vector<int> idx_vector);

    void addKey(SecretKey& sk);


    /********************************Single-Input Non-Linear Functions*******************************/
    // Exponential function: exp(x) = e^x
    void evalExp(Ciphertext& cipher);

    // CDF function: 0.5 * (1 + erf(x / sqrt(2)))
    void evalCDF(Ciphertext& cipher);
    // GeLU = x * CDF(x)
    void evalGeLU(Ciphertext& cipher);  

    // Sigmoid function: exp(x) / (1 + exp(x))
    void evalSigmoid(Ciphertext& cipher);
    // SiLU = x * Sigmoid(x)
    void evalSiLU(Ciphertext& cipher);

    /********************************Multi-Input Non-Linear Functions*******************************/
    // eval 1/x
    vector<double> K_inv;
    void evalInv(Ciphertext& cipher, double upper_bound = 1.0);

    // eval 1/\sqrt(x)
    vector<double> K_sqrt_inv;
    void evalSqrtInv(Ciphertext& cipher, SecretKey& sk, double upper_bound = 1.0);

    // SoftMax function: exp(x_i) / sum(exp(x_j))
    void evalSoftMax(vector<Ciphertext*>& cipher_P);

    // Layer Normalization: normalize each row of the matrix
    void LayerNorm(Ciphertext& cipher);

    vector<Ciphertext*> nonlieanr_buffer;

    /********************************CCMM for Multi-head Attention*******************************/

    // Q and K are the ciphertext in colomn packing
    // compute Q * K ^ T
    // example: Q 128*64 * K 128*64 -> O 128*128
    Ciphertext** tmpcipher_buffer;
    Ciphertext* leafnode;
    Ciphertext* tmp_shift_K;
    cuDoubleComplex* rot_diag;
    cuDoubleComplex* device_diag;
    vector<Plaintext> plain_tau_diag = vector<Plaintext>(64);
    vector<Plaintext> mask_ccmm_left = vector<Plaintext>(64);
    vector<Plaintext> mask_ccmm_right = vector<Plaintext>(64);
    void CCMM_QK(Ciphertext& Q, Ciphertext& K, Ciphertext& O1, Ciphertext& O2);
    void Recursive_CCMM_reduce(Ciphertext& Q, Ciphertext& K, int layer, int max_layer, int seq, int column_num, Ciphertext** tmpcipher_buffer);
    void CCMM_QK_splited_heads(vector<Ciphertext *>& Q, vector<Ciphertext *>& K,vector<Ciphertext *>& O, int column_each_head);
    void TauAndEqual(Ciphertext& A);
    void CCMM_V(Ciphertext& sigma_O1, Ciphertext& sigma_O2, Ciphertext& tau_V, Ciphertext& O);
};