#pragma once

#include "../include/uint128.cuh"
#include "../include/Utils.cuh"
#include "../include/SecretKey.cuh"
#include "../include/Scheme_23.cuh"
#include "../include/Key.cuh"
#include "../include/advanced/SchemeAlgo.cuh"


class CCMM_Scheme{
public:
    Scheme_23& scheme;
    Context_23& context;
    SchemeAlgo& scheme_algo;
    CCMM_Scheme(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, int d, int head_num, int token_len);

    vector<Ciphertext*> mult_buffer;
    Ciphertext* rot_buffer;

    void addKey(SecretKey& sk);

    // Q and K are the ciphertext in colomn packing
    // compute Q * K ^ T
    // example: Q 128*64 * K 128*64 -> O 128*128
    void CCMM_QK(Ciphertext& Q, Ciphertext& K, Ciphertext& O);

    // Q and K are the ciphertext in colomn packing
    // compute O * V
    // example: O 128*128 * V 128*64 -> S 128*64
    void CCMM_OV(Ciphertext& O, Ciphertext& V, Ciphertext& S);

    void multConstDiagAndEqual(Ciphertext& cipher, Plaintext& cnst, int offset = 0);

    // n
    int token_len;
    // matrix size of each head
    // 768 in bert
    int d;
    // head_num
    // 12 in bert
    int head_num;


    cuDoubleComplex* column_mask_buffer;
    cuDoubleComplex* column_mask_buffer_device;
    Plaintext* column_mask;
    vector<Plaintext*> diag_mask;
    vector<Plaintext*> diag_mask_2;

};