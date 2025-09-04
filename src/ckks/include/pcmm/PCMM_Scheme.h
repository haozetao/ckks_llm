#pragma once

#include "../uint128.cuh"
#include "../Utils.cuh"
#include "../SecretKey.cuh"
#include "../Scheme_23.cuh"
#include "../Key.cuh"
#include "MLWESecretKey.cuh"

#include "../bootstrapping/Bootstrapper.cuh"

class PCMM_Scheme{
public:
    PCMM_Context& pcmm_context;
    Scheme_23& scheme;
    Context_23& context;
    Bootstrapper& bootstrapper;
    SecretKey& sk;
    PCMM_Scheme(PCMM_Context& pcmm_context, Scheme_23& scheme, Bootstrapper& bootstrapper, SecretKey& sk);

    uint64_tt* embeded_mlwe_buffer;
    vector<Key*> repackingKeys;

    vector<uint64_tt*> repacking_cipher_pointer;
    uint64_tt** repacking_cipher_pointer_device;

    uint64_tt* ppmm_output;
    Plaintext* plain;


    __host__ void convertMLWESKfromRLWESK(MLWESecretKey& mlwe_sk, SecretKey& rlwe_sk);

    __host__ void mlweDecrypt(MLWECiphertext& mlwe_cipher, MLWESecretKey& mlwe_sk, MLWEPlaintext& mlwe_plain);

    __host__ void rlweCipherDecompose(Ciphertext& rlwe_cipher, vector<MLWECiphertext*> mlwe_cipher_decomposed, uint64_tt mlwe_num, uint64_tt offset);

    // modpacking algorithm
    __host__ void addRepakcingKey(MLWESecretKey& mlwe_sk, SecretKey& rlwe_sk);

    // modpacking algorithm
    __host__ void mlweCipherPacking(Ciphertext& rlwe_cipher, vector<MLWECiphertext*> mlwe_cipher_decomposed, uint64_tt mlwe_num, uint64_tt offset);

    __host__ void PPMM(float* plain_mat, vector<MLWECiphertext*> mlwe_cipher_decomposed, int mat_M, int mat_N, int mat_K);

    __host__ void coeffBitRev(Ciphertext& rlwe_cipher);

    __host__ void PCMM_Boot(float* plain_mat, Ciphertext& rlwe_cipher, vector<MLWECiphertext*>& mlwe_cipher_decomposed, int mat_M, int mat_N, int mat_K);
};