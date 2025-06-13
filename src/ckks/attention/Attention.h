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

    vector<Plaintext*> column_mask;
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
    void evalInv(Ciphertext& cipher, SecretKey& sk, double upper_bound = 1.0);

    // eval 1/\sqrt(x)
    vector<double> K_square_inv;
    void evalSquareInv(Ciphertext& cipher, double upper_bound = 1.0);

    // SoftMax function: exp(x_i) / sum(exp(x_j))
    void evalSoftMax(Ciphertext& cipher);

    // Layer Normalization: normalize each row of the matrix
    void LayerNorm(Ciphertext& cipher);
};