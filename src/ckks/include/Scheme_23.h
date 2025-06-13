#pragma once

#include "Context_23.h"
#include "Key.cuh"
#include "Key_decomp.cuh"
#include "SecretKey.cuh"
#include "Plaintext.cuh"
#include "Ciphertext.cuh"
// #include "./advanced/MatrixDiag.cuh"


class Scheme_23
{
public:
    Context_23& context;
	Scheme_23(Context_23& context):context(context){mallocMemory();}

	// L+1
    uint64_tt *axbx1_mul, *axbx2_mul, *bxbx_mul;
	// K+L+1
	uint64_tt *sxsx, *sx_coeff;
	// K+L+1
    uint64_tt *axax_mul, *temp_mul, *ex_swk;
	// t_num * Ri_blockNum
	uint64_tt *modUp_RitoT_temp;
	// t_num * Qj_blockNum
	uint64_tt* modUp_QjtoT_temp;
	// t_num * Ri_blockNum * 2
	uint64_tt* exProduct_T_temp;
	// N * t_num * Ri_blockNum * 2;
	uint64_tt* modUp_TtoQj_buffer;

	// N * (L+1) * 2
	uint64_tt* rescale_buffer;
	Ciphertext* cipher_temp_pool;
	// K+L+1
	uint64_tt *ex_enc,*vx_enc;
	uint8_tt *in_enc;
	uint64_tt random_len_for_enc;

	vector<uint64_tt> add_const_copy_vec;
	uint64_tt* add_const_buffer;
	
	uint64_tt** rotKey_pointer_device;
	int* rotSlots_device;

	// map<long, Key*> keyMap; ///< contain Encryption, Multiplication and Conjugation keys, if generated
	// map<long, Key*> leftRotKeyMap; ///< contain left rotation keys, if generated
    
	Key* publicKey = nullptr;
	// rlk_23_i = (b = -as + P s^2 + e, a)
	Key_decomp* rlk_23 = nullptr;
	// Key* RotKey;
	vector<Key_decomp*> rotKey_vec_23;
	// Conjugate
	Key_decomp* ConjKey_23 = nullptr;
	// for bootstrapping subAndSum
	vector<Key_decomp*> autoKey_vec_23;

	// map<long, vector<Key*>> keyMap_better; ///< contain Encryption, Multiplication and Conjugation keys, if generated
	// map<long, vector<Key*>> leftRotKeyMap_better; ///< contain left rotation keys, if generated
	// vector<Key*> mult_key;

	// map<long, uint64_tt*> ntt_gadget; ///< ntt key switch gadget on PQ

	void mallocMemory();
	
	void decrypt_display(SecretKey& sk, Ciphertext& cipher, char* s, int row_num = 32768);

	/**
	 * generates key for public encryption (key is stored in publicKey)
	 */
	void addEncKey(SecretKey& secretKey, cudaStream_t stream = 0);

			/**
			 * generates key for key switch (key is stored in xxx)
			 */
			Key_decomp* addSWKey_23(SecretKey& secretKey, uint64_tt* s2, cudaStream_t stream = 0);

			/**
			 * generates key for multiplication (key is stored in rlk_23)
			 */
			void addMultKey_23(SecretKey& secretKey, cudaStream_t stream = 0);

			/**
			 * generates key for conjugation (key is stored in ConjKey_23)
			 */
			void addConjKey_23(SecretKey& secretKey, cudaStream_t steam = 0);

			/**
			 * generates key for left rotation <Hoisting Rotation> (key is stored in rotKey_vec_23)
			 */
			void addLeftRotKey_23(SecretKey& secretkey, long rot_num, cudaStream_t stream = 0);

			/**
			 * generates key for automorphism (key is stored in autoKey_vec_23)
			 */
			void addAutoKey_23(SecretKey& secretkey, int d, cudaStream_t stream = 0);


	// void addLeftRotKeys(SecretKey& secretKey);
	// void addRightRotKeys(SecretKey& secretKey);

    // // l+1 = num of q
	// Plaintext encode(double* vals, long slots, long l, cudaStream_t stream = 0);
    // // l+1 = num of q
	// Plaintext encode(cuDoubleComplex* vals, long slots, long l, cudaStream_t stream = 0); 

	// cuDoubleComplex* decode(Plaintext& msg, cudaStream_t stream = 0);


    // using keyMap[ENCRYPTION] as pk
	void encryptZero(Ciphertext& cipher, int l, int slots, cudaStream_t stream = 0);
	void encryptMsg(Ciphertext& c, Plaintext& message, cudaStream_t stream = 0);

	void decryptMsg(Plaintext& m, SecretKey& secretKey, Ciphertext& cipher, cudaStream_t stream = 0);


    // Homomorphic Addition
	void add(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2);
	void addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void addConstAndEqual(Ciphertext& cipher, Plaintext& cnst);
	void addConstAndEqual(Ciphertext& cipher, double cnst);
	void addConstAndEqual(Ciphertext& cipher, cuDoubleComplex cnst);

	// Homomorphic Substraction
	void sub(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2);
	void subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);
	void constSub(Ciphertext& cipher_res, Ciphertext& cipher, double cnst);
	void constSubAndEqual(Ciphertext& cipher, double cnst);

	// Homomorphic Negation
	void negate(Ciphertext& cipher_res, Ciphertext& cipher);
	void negateAndEqual(Ciphertext& cipher);

	// cipher = cipher * -i
	void divByiAndEqual(Ciphertext& cipher);
	void mulByiAndEqual(Ciphertext& cipher);

/*******************************************************key switch***********************************************************/
	// Homomorphic Multiplication
	void multAndEqual_23(Ciphertext& cipher1, Ciphertext& cipher2);
	void mult_23(Ciphertext &cipher_res, Ciphertext &cipher1, Ciphertext &cipher2);

	void multAndEqual_beforeIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_input, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch);
	void multAndEqual_afterIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_output, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch);


	void multConstAndEqual(Ciphertext& cipher, Plaintext& cnst);
	void multConstAndEqual(Ciphertext& cipher, double cnst);
	// void multConstAndEqual(Ciphertext& cipher, cuDoubleComplex cnst);

	void multConstAndAddCipherEqual(Ciphertext& c1, Ciphertext& c2, double cnst, NTL::RR target_scale);

	void rescaleAndEqual(Ciphertext& cipher);
	void rescaleAndEqual(Ciphertext& cipher, NTL::RR target_scale);
	void rescaleAndEqual_noNTT(Ciphertext& cipher);

	void rescaleToAndEqual(Ciphertext& cipher, int level);

	void conjugateAndEqual_23(Ciphertext& cipher); 
	void conjugate_23(Ciphertext& cipher_res, Ciphertext& cipher);

	//Homomorphic <Hoisting Rotation> Rotate
	void leftRotateAndEqual_23(Ciphertext& cipher, long rotSlots);
	void leftRotateAddSelf_23(Ciphertext& cipher, long rotSlots);
	void leftRotate_23(Ciphertext& cipher_res, Ciphertext& cipher, long rotSlots);
	// void leftRotateMany_23(Ciphertext& cipher, Ciphertext* cipher_res, vector<int> rotSlotsVec);
	void rightRotateAndEqual_23(Ciphertext& cipher, long rotSlots);
	// void rightRotate_23(Ciphertext& cipher, Ciphertext& cipher_res, long rotSlots);

	// f(X) -> f(X) + f(X^(2^d+1))
	void automorphismAndAdd(Ciphertext& cipher, int d);

	// Homomorphic Squaring
	void square(Ciphertext &cipher1, Ciphertext& cipher2);
	void squareAndEqual(Ciphertext& cipher);
	void divByPo2AndEqual(Ciphertext& cipher);

	void square_double_add_const_rescale(Ciphertext& cipher1, Ciphertext& cipher2, double cnst);
	void squareAndEqual_double_add_const_rescale(Ciphertext& cipher, double cnst);
};