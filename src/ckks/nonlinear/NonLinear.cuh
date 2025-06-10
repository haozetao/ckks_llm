#pragma once

#include "NonLinear.h"
#include "../include/advanced/SchemeAlgo.cuh"

NonLinear::NonLinear(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo)
    : context(context), scheme(scheme), scheme_algo(scheme_algo)
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
    
}

void NonLinear::evalExp(Ciphertext& cipher)
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
void NonLinear::evalCDF(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 0.2);
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, CDF_cheby_poly_pool);
}

// Sigmoid function: exp(x) / (1 + exp(x))
void NonLinear::evalSigmoid(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 0.2);
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, sigmoid_cheby_poly_pool);
}