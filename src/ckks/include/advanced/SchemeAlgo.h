#pragma once

#include "unistd.h"

#include "../Scheme_23.h"
#include "../Context_23.h"
#include "SchemeAlgo.cuh"
#include "../Plaintext.cuh"

struct Chebyshev_Polynomial{
	Chebyshev_Polynomial(){}
	Chebyshev_Polynomial(vector<double> coeffs, int maxDegree): coeffs(coeffs), maxDegree(maxDegree){}
	~Chebyshev_Polynomial(){}
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


	vector<Ciphertext*> chebyshev_tree_pool;
	uint64_tt* chebyshev_tree_cipher_pool;
	vector<Ciphertext*> cheby_basis_pool;

	void malloc_bsgs_buffer(int cheby_degree);
	
	void evalPolynomialChebyshev(Ciphertext &cipher, NTL::RR target_scale, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);

	// void computePowerBasis(int idx, NTL::RR target_scale);

	void evalRecurse(NTL::RR target_scale, int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);
	// void evalIteration(NTL::RR target_scale, int logDegree, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);
	// void evalIterationBatch(NTL::RR target_scale, int logDegree);

	void call_prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);
	void prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);

	void evalFromPowerBasis(NTL::RR target_scale, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector);

	void square_double_add_const_rescale(Ciphertext& cipher, double cnst);
};