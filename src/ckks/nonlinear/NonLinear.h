#pragma once

#include "../include/advanced/SchemeAlgo.cuh"
#include "../include/Context_23.cuh"
#include "../include/Scheme_23.cuh"


class NonLinear {
public:
    Context_23& context;
    Scheme_23& scheme;
    SchemeAlgo& scheme_algo;
    
    NonLinear(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo);



    vector<double> exp_cheby_coeffs;
    vector<Chebyshev_Polynomial*> exp_cheby_poly_pool;
    void evalExp(Ciphertext& cipher);


    vector<double> CDF_cheby_coeffs;
    vector<Chebyshev_Polynomial*> CDF_cheby_poly_pool;
    // CDF function: 0.5 * (1 + erf(x / sqrt(2)))
    void evalCDF(Ciphertext& cipher);

    vector<double> sigmoid_cheby_coeffs;
    vector<Chebyshev_Polynomial*> sigmoid_cheby_poly_pool;
    // Sigmoid function: exp(x) / (1 + exp(x))
    void evalSigmoid(Ciphertext& cipher);
};