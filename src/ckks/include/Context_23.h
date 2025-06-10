#pragma once

#include <vector>
#include <sys/time.h>
#include <complex>
#include <map>

using namespace std;

#include "fft_special.cuh"
#include "Utils.cuh"
#include "poly_arithmetic.cuh"
#include "Sampler.cuh"
#include "RNG.cuh"
#include "poly_arithmetic.cuh"
#include "Plaintext.cuh"
#include "ntt_60bit.cuh"

class Context_23{
    public:
    Context_23(long logN, long logslots, long h = 64, double sigma = dstdev);

	void getPrimeCKKS(int hamming_weight);
	void preComputeOnCPU();
	void copyMemoryToGPU();

	__host__ void rescaleAndEqual(uint64_tt* device_a, int l);

	__host__ void encode(cuDoubleComplex* vals, Plaintext& msg);
	__host__ void decode(Plaintext& msg, cuDoubleComplex* vals);
	__host__ void encode_coeffs(double* vals, Plaintext& msg);
	__host__ void decode_coeffs(Plaintext& msg, double* vals);


	__host__ void PolyToBigintLvl(int level, uint64_tt* p1, int gap, std::vector<NTL::ZZ>& coeffsBigint);
	__host__ void encode_T(cuDoubleComplex* vals, PlaintextT& msg_PQl, NTL::RR scale);

	//old_ntt
	__host__ void forwardNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num);
	__host__ void inverseNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num);
	//new_ntt
	__host__ void FromNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int num);
	__host__ void FromNTTInplace_for_externalProduct(uint64_tt* device_a, int start_mod_idx, int cipher_mod_len, int poly_mod_num, int block_mod_num, int poly_mod_len, int cipher_mod_num, int batch_size);
	__host__ void ToNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int num);
	__host__ void ToNTTInplace_for_externalProduct(uint64_tt* device_a, int start_mod_idx, int cipher_mod_len, int poly_mod_num, int block_mod_num, int poly_mod_len, int cipher_mod_num, int batch_size);

	__host__ void divByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num);
	__host__ void mulByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num);
	__host__ void poly_add_complex_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_buffer, int idx_a, int idx_mod, int mod_num);
	__host__ void poly_mul_const_batch_device(uint64_tt* device_a, uint64_tt* const_real, int idx_mod, int mod_num);
	__host__ void poly_mul_const_add_cipher_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, uint64_tt target_scale, int idx_mod, int mod_num);
	
	// for ckks-23 fast-ksw decomposition
	__host__ void modUpPQtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size);
	__host__ void modUpQjtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size);
	// __host__ void modDownTtoRi_23(uint64_tt* output, uint64_tt* modUp_QjtoT_temp, uint64_tt* exProduct_T_temp, int l);
	__host__ void modUpTtoPQl_23(uint64_tt* modUp_TtoQj_buffer, uint64_tt* exProduct_T_temp, int l, int batch_size);
	__host__ void modDownPQltoQl_23(uint64_tt* output, uint64_tt* modUp_TtoQj_buffer, int l, int batch_size);

	// for ckks-23 fast-ksw external product
	__host__ void external_product_T(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l);
	__host__ void external_product_T_swk_reuse(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l, int batch_size);
	__host__ void mult_PlaintextT(uint64_tt* cipher_modUp_QjtoT, PlaintextT& plain_T, int l);
	__host__ void mult_PlaintextT_permuted(uint64_tt* cipher_modUp_QjtoT, PlaintextT& plain_T, int rotSlots, int l);

	// Encryption parameters
	int logN; ///< Logarithm of Ring Dimension
	int logNh; ///< Logarithm of Ring Dimension - 1
	int logslots;
	int slots;
	int L; ///< Maximum Level that we want to support
	int q_num; ///< num of q (usually L + 1)
	int K; ///< The number of special modulus (usually (L + 1) / dnum)
	int dnum;
	int alpha;

	int p_num;
	int t_num;
	int mod_num;
	int gamma; // gamma stands for tilde_r
	int Ri_blockNum; // blockNum = ceil((p_num + q_num) / gamma)
	int Qj_blockNum;

	long N;
	long M;
	long Nh;

	long logp; 
	long precision;

	long h;
	double sigma;

	vector<uint64_tt> qVec;
	vector<uint64_tt> pVec;
	vector<uint64_tt> tVec;

	vector<uint128_tt> qMuVec; // Barrett reduction
	vector<uint128_tt> pMuVec; // Barrett reduction
	vector<uint128_tt> tMuVec;

	vector<uint64_tt> qPsi; // psi q
	vector<uint64_tt> pPsi; // psi p
	vector<uint64_tt> tPsi;	// psi t
	vector<uint64_tt> qPsiInv; // inv psi q
	vector<uint64_tt> pPsiInv; // inv psi p
	vector<uint64_tt> tPsiInv; // inv psi t

	vector<uint64_tt> pqtVec; // pqt
	vector<uint64_tt> pqt2Vec;
	vector<uint128_tt> pqtMuVec; // pqt Barrett reduction
	vector<uint64_tt> pqtMuVec_high;
	vector<uint64_tt> pqtMuVec_low;

	vector<uint64_tt> pqtPsi; // pqt Psi
	vector<uint64_tt> pqtPsiInv; // pqt Psi inv
	// psi powers pq
	uint64_tt* pqtPsiTable_device; // ok
	// inv psi powers pq
	uint64_tt* pqtPsiInvTable_device;  // ok

	vector<double> eval_sine_chebyshev_coeff;
	int double_angle_cost_level;
	int eval_sine_K;

	/************************************base convert from PQl to Ql****************************************/
		// P/pk					
		// [P/p0 P/p1 ... P/pk] mod qi
		// P/pk mod qi
		// size = (L + 1) * K
		// ok
		vector<vector<uint64_tt>> pHatVecModq_23;
		uint64_tt* pHatVecModq_23_device;

		// pk/P
		// inv[p012...k/p0] inv[p012...k/p1] ... inv[p012...k/pk]
		// pk/P mod pk
		// size = K
		// ok
		vector<uint64_tt> pHatInvVecModp_23;
		vector<uint64_tt> pHatInvVecModp_23_shoup;
		uint64_tt* pHatInvVecModp_23_device;

		// P{Q-Ql} mod qi
		vector<vector<uint64_tt>> PQ_inv_mod_qi_better;
		// size = (L+1)*(L+1)
		uint64_tt* PQ_inv_mod_qi_better_device;

		vector<vector<uint64_tt>> PQ_div_Qj_modqi;
		// size = (L+1)*(L+1)
		uint64_tt* PQ_div_Qj_modqi_device;

	/************************************base convert from Ri to T******************************************/
		// r_ij/Ri mod r_ij
		// {inv[R0/r_00] inv[R0/r_01] ... inv[R0/r_{0,gamma-1}]} ... {inv[R_{blockNum-1}/r_{blockNum-1,gamma-1}]}
		// {r_00/R0 mod r_00 ... r_{gamma-1}0/R0 mod r_{gamma-1}0} ...... {r_0{blockNum-1}/R_{blockNum-1} mod r_0{blockNum-1} ... r_{blockNum-1}{gamma-1}/R0 mod r_{blockNum-1}{blockNum-1}}
		// size = gamma * blockNum
		// ok
		vector<vector<uint64_tt>> RiHatInvVecModRi_23;
		vector<vector<uint64_tt>> RiHatInvVecModRi_23_shoup;
		// gamma * blockNum
		uint64_tt* 	RiHatInvVecModRi_23_device;
		uint64_tt* 	RiHatInvVecModRi_23_shoup_device;

		// Ri/r_ij mod t_k
		// {[R0/r_00] [R0/r_01] ... [R0/r_{0,gamma-1}]} ... {[R_{blockNum-1}/r_{blockNum-1,gamma-1}]}
		// {r_00/R0 mod r_00 ... r_{gamma-1}0/R0 mod t_k} ...... {r_0{blockNum-1}/R_{blockNum-1} mod r_0{blockNum-1} ... r_{blockNum-1}{gamma-1}/R0 mod r_{blockNum-1}{blockNum-1}}
		// size = gamma * t_num * blockNum
		// ok
		vector<vector<vector<uint64_tt>>> RiHatVecModT_23;
		// gamma * t_num * blockNum
		uint64_tt* RiHatVecModT_23_device;
		// Ri mod ti
		// {R0 ... R_{blockNum-1}} mod t0 ... {R0 ... R_{blockNum-1}} mod t_{t_num-1}
		vector<vector<uint64_tt>> Rimodti;


	/************************************base convert from Qj to T******************************************/
		// q_ij/Qi mod q_ij
		// {inv[Q0/q_00] inv[Q0/q_01] ... inv[Q0/r_{0,gamma-1}]} ... {inv[Q_{blockNum-1}/q_{blockNum-1,gamma-1}]}
		// 
		// size = p_num * blockNum * (L+1)
		// 
		vector<vector<vector<uint64_tt>>> QjHatInvVecModQj_23;
		vector<vector<vector<uint64_tt>>> QjHatInvVecModQj_23_shoup;
		uint64_tt* QjHatInvVecModQj_23_device;
		uint64_tt* QjHatInvVecModQj_23_shoup_device;

		// size = p_num * blockNum
		// 
		vector<vector<vector<vector<uint64_tt>>>> QjHatVecModT_23;
		uint64_tt* QjHatVecModT_23_device;
		// Qj mod ti
		// {Q0 ... Q_{blockNum-1}} mod t0 ... {Q0 ... Q_{blockNum-1}} mod t_{t_num-1}
		vector<vector<uint64_tt>> Qjmodti;
		uint64_tt* Qjmodti_device;

	/************************************base convert from T to Ri******************************************/
		// ti / T mod ti
		// {inv[T/t0] inv[T/t1] ... inv[T/t_{t_num-1}]}
		// size = t_num
		// ok
		vector<uint64_tt> THatInvVecModti_23;
		vector<uint64_tt> THatInvVecModti_23_shoup;
		uint64_tt* THatInvVecModti_23_device;
		uint64_tt* THatInvVecModti_23_shoup_device;

		// T / ti mod Ri
		// {[T/t0] [T/t1] ... [T/t_{t_num-1}] mod r_i}
		// size = t_num * (p_num + q_num)
		//
		vector<vector<uint64_tt>> THatVecModRi_23;
		uint64_tt* THatVecModRi_23_device;
		// T mod pqi
		// {T mod pq0 ... T mod pq_{n-1}}
		vector<uint64_tt> Tmodpqi;

	/************************************************rescale************************************************/
	// qi mod qj
	// inv[q1]mod qi inv[q2]mod qi inv[q3]mod qi inv[q4]mod qi ... inv[qL]mod qi
	// ql mod qi [l(l-1)/2 + i]
	// ok
	vector<vector<uint64_tt>> qiInvVecModql;
	vector<vector<uint64_tt>> qiInvVecModql_shoup;
	uint64_tt* qiInvVecModql_device;
	uint64_tt* qiInvVecModql_shoup_device;

	/************************************************decode*************************************************/
	vector<vector<uint64_tt>> QlInvVecModqi;
	uint64_tt* QlInvVecModqi_device;
	vector<vector<uint64_tt>> QlHatVecModt0;
	uint64_tt* QlHatVecModt0_device;

	/************************************copy to constant memory********************************************/
	vector<uint64_tt> halfTmodpqti; 	// T//2 mod pqti
	vector<uint64_tt> PModqt;			// P mod q
	vector<uint64_tt> PModqt_shoup;
	vector<uint64_tt> PinvModq;			// P^-1 mod q
	vector<uint64_tt> PinvModq_shoup;

	long randomArray_len;
	// random array for key gen
	// only sk pk relk
	// ! not ok
    uint8_tt* randomArray_device;
	// = randomArray_device
	uint8_tt* randomArray_sk_device;
	// = randomArray_device + N 
	uint8_tt* randomArray_pk_device;
	// = randomArray_device + N + (L + 1 + K) * N * sizeof(uint64_tt) / sizeof(uint8_tt)
	uint8_tt* randomArray_e_pk_device;
	uint8_tt* randomArray_swk_device;
	uint8_tt* randomArray_e_swk_device;

	// precomputed rotation group indexes
	uint64_tt* rotGroups_device;
	// // precomputed ksi powers
	cuDoubleComplex* ksiPows_device;


	// N * (L+1)
	uint64_tt* decode_buffer_device;
	uint64_tt* decode_buffer_host;
	cuDoubleComplex* encode_buffer;
	double* encode_coeffs_buffer;

	//new_ntt_param
	vector<uint64_tt> n_inv_host;
    vector<uint64_tt> n_inv_shoup_host;

	uint64_tt* psi_table_device;
	uint64_tt* psiinv_table_device;

	uint64_tt* n_inv_device;
	uint64_tt* n_inv_shoup_device;
	
	uint64_tt* psi_shoup_table_device;
	uint64_tt* psiinv_shoup_table_device;
};