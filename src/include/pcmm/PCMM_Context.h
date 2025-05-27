#pragma once

#include "../uint128.cuh"
#include "../Utils.cuh"
#include "../Context_23.cuh"
#include "MLWECiphertext.cuh"
#include "MLWEPlaintext.cuh"


__constant__ uint64_tt ringpack_pq_cons[4];
__constant__ uint64_tt ringpack_pq_mu_low_cons[4];
__constant__ uint64_tt ringpack_pq_mu_high_cons[4];

class PCMM_Context{
public:
    Context_23& context;
    PCMM_Context(int N1, int mlwe_rank, vector<uint64_tt> p_ringpack, vector<uint64_tt> q_ringpack, Context_23& context);

    void preComputeOnCPU();
    void copyMemoryToGPU();

    // NTT 256 point
    __host__ void ToNTTInplace(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size);
    __host__ void FromNTTInplace(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size);

    // // decompose 1 R_N -> k R_N1
    // __host__ void ringDown(uint64_tt* bigRing_polys, uint64_tt* smallRing_poly, int mod_num);
    // // packing k R_N1 -> 1 R_N
    // __host__ void ringUp(uint64_tt* smallRing_poly, uint64_tt* bigRing_polys, int mod_num);

    // encoding to coeffs
    __host__ void encodeCoeffs(MLWEPlaintext& cipher, double* vals);
    __host__ void decodeCoeffs(MLWEPlaintext& cipher, double* vals);

    // MLWE poly dim
    int N1;
    // MLWE poly rank
    int mlwe_rank;
    // modulus
    int ringpack_p_count;
    int ringpack_q_count;
    int ringpack_pq_count;

    vector<uint64_tt> p_ringpack;
    vector<uint64_tt> q_ringpack;

    vector<uint64_tt> pq_ringpack;
    vector<uint128_tt> pq_ringpack_mu;
    vector<uint64_tt> pq_ringpack_mu_high;
    vector<uint64_tt> pq_ringpack_mu_low;
    
    vector<uint64_tt> p_inv_mod_qi_host;
    vector<uint64_tt> p_inv_mod_qi_shoup_host;

    uint64_tt* p_inv_mod_qi;
    uint64_tt* p_inv_mod_qi_shoup;


    vector<uint64_tt> psi_pq_ringpack;
    vector<uint64_tt> psi_inv_pq_ringpack;

	vector<uint64_tt> N1_inv_host;
    vector<uint64_tt> N1_inv_shoup_host;
    uint64_tt* N1_inv_device;
    uint64_tt* N1_inv_shoup_device;


    uint64_tt* N1_psi_table_device;
    uint64_tt* N1_psi_shoup_table_device;

    uint64_tt* N1_psiinv_table_device;
    uint64_tt* N1_psiinv_shoup_table_device;

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
};