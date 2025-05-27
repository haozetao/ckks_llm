#pragma once

#include "unistd.h"

#include "../Scheme_23.h"
#include "../Context_23.h"
#include "SchemeAlgo.cuh"
#include "../Plaintext.cuh"

struct Chebyshev_Polynomial{
	Chebyshev_Polynomial(){}
	Chebyshev_Polynomial(vector<double> coeffs, int maxDegree): coeffs(coeffs), maxDegree(maxDegree){}
	int degree(){return coeffs.size() - 1;}
	int maxDegree;
	vector<double> coeffs;
};

class SchemeAlgo
{
public:
    Context_23& context;
    Scheme_23& scheme;
	SecretKey& secretkey;
	SchemeAlgo(Context_23& context, Scheme_23& scheme, SecretKey& secretkey);

	int N;
	int slots;
	int logN;
	int logslots;
	int p_num;
	int q_num;
	int maxLevel;
	long precision;


	// // N * t_num * Qj_blockNum
	// uint64_tt* IP_input_temp;
	// // N * t_num * Ri_blockNum * 2
	// uint64_tt* IP_output_temp;
	// // N * (L+1)
	// uint64_tt* axbx1_mul;
	// // N * (L+1)
	// uint64_tt* bxbx_mul;

	vector<Ciphertext*> chebyshev_tree_pool;
	uint64_tt* chebyshev_tree_cipher_pool;
	vector<Ciphertext*> eval_sine_poly_pool;
	vector<Chebyshev_Polynomial*> chebyshev_poly_coeff_tree_pool;

	// vector<uint64_tt> add_const_copy_vec;
	// uint64_tt* add_const_buffer;

	vector<bool> eval_sine_poly_pool_computed;
	Plaintext* plain_buffer;
	cuDoubleComplex* complex_vals;

	void malloc_bsgs_buffer(int sine_degree);

    // void evalLinearTransformAndEqual(Ciphertext &cipher, MatrixDiag* matrixDiag);
	
	void evalPolynomialChebyshev(Ciphertext &cipher, NTL::RR target_scale);

	void computePowerBasis(int idx, NTL::RR target_scale);

	void evalRecurse(NTL::RR target_scale, int logSplit, int logDegree, int tree_idx);
	void evalIteration(NTL::RR target_scale, int logDegree);
	// void evalIterationBatch(NTL::RR target_scale, int logDegree);

	void prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx);
	void evalFromPowerBasis(NTL::RR target_scale, int tree_idx);

	void square_double_add_const_rescale(Ciphertext& cipher, double cnst);
};