#pragma once

#include "../Scheme_23.h"
#include "../Context_23.h"
#include "../advanced/SchemeAlgo.cuh"

// #include "Bootstrapping_encoding.h"
#include "Bootstrapping_encoding.cuh"
#include "Bootstrapping_C2S_S2C_impl.cuh"

class Bootstrapper
{
public:
    Context_23& context;
    Scheme_23& scheme;
	SchemeAlgo& scheme_algo;
	Bootstrapper(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, SecretKey& secretkey, EncodingMatrix& encodingMatrix, int is_STC_first, int logradix = 4);

	// scale the cipher_Q0 to cipher_QL
	int N;
	int slots;
	int logN;
	int logslots;
	int p_num;
	int q_num;
	int t_num;
	int maxLevel;
	long precision;

	// size = qnum
	vector<uint64_tt> inv_Ndiv2slots;
	uint64_tt* inv_Ndiv2slots_device;
	
	cuDoubleComplex* diag_inv_idx_buffer;
	cuDoubleComplex* diag_idx_buffer;

	int radix;
    int logradix;

	int c2s_cost_level;
	int s2c_cost_level;
	int sine_cost_level;
	int double_angle_cost_level;
	int bootstrapping_cost_level;

	// interpolation in the range (-K, K)
	int eval_sine_K;
	double sine_A, sine_B;
	// 2^double_angle_level
	double sine_factor;
	NTL::RR eval_sine_scaling_factor;
	NTL::RR qDiff;
	NTL::RR sqrt2Pi;
	vector<double> eval_sine_chebyshev_coeff;

	// |m|
	double message_ratio;
	// sine degree of evalmod1
	int sine_degree;
	// arcsine degree of evalmod1
	int arcsine_degree;

	// if generated boot_key
	int boot_key_flag = 0;

	// if the first step of boot is STC
	int is_STC_first = 0;

    Ciphertext *ctReal = nullptr, *ctImag = nullptr;
	Ciphertext *ctImag_tmp = nullptr;

	EncodingMatrix& encodingMatrix;
	void prepare_sine_chebyshev_poly();

	void addBootstrappingKey(SecretKey& secretKey, cudaStream_t stream = 0);

	void PrefetchRotKeys(vector<int> rotIdx);

	void resetScale(Ciphertext& cipher);

    void modUpQ0toQL(Ciphertext& cipher);

	void subAndSum(Ciphertext& cipher);

	void coeffToSlot(Ciphertext& cipher, Ciphertext& cipherReal, Ciphertext& cipherImag);

	void EvalModAndEqual(Ciphertext& cipher);

	void slotToCoeff(Ciphertext& cipher, Ciphertext& cipherReal, Ciphertext& cipherImag);

	void FirstModUpBootstrapping(Ciphertext& cipher);
	
	void FirstSTCBootstrapping(Ciphertext& cipher);

	void newResetScale(Ciphertext& cipher);

	void Bootstrapping(Ciphertext& cipher);
};