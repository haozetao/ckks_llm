//
// Created by cqd on 24-7-12.
//

#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include <map>
#include <algorithm>

#include "../Scheme_23.cuh"
#include "../SecretKey.cuh"

// #include "Bootstrapping_encoding.cuh"
// #include "Bootstrapping_C2S_S2C_impl.cuh"

using namespace std;

class EncodingMatrix {
public:

    EncodingMatrix(SecretKey& secretKey, Scheme_23& scheme, int levelS2C, int levelC2S, int is_STC_first = 0);

    complex<double>* Rotate(const vector<complex<double>>& a, int index, bool tr);

    //--------------------OpenFHE的矩阵生成函数----------------------------

    // computes all powers of a primitive root of unity exp(2 * M_PI/m)
    vector<complex<double>> ComputeRoots(int N,bool a = true);

    //生成矩阵分解的参数
    vector<int32_t> GetCollapsedFFTParams(uint32_tt slots, uint32_tt levelBudget, uint32_tt dim1);

    /**
     * Computes parameters to ensure the encoding and decoding computations take exactly the
     * specified number of levels. More specifically, it returns a vector than contains
      * layers (the number of layers to collapse in one level), rows (how many such levels),
     * rem (the number of layers remaining to be collapsed in one level)
     *
     * @param logSlots the base 2 logarithm of the number of slots.
     * @param budget the allocated level budget for the computation.
     */
    vector<uint32_tt> SelectLayers(uint32_tt logSlots, uint32_tt budget);

    /**
     * Computes the coefficients for the FFT encoding for CoeffEncodingCollapse such that every
     * iteration occupies one level.
     *
     * @param pows vector of roots of unity powers.
     * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
     * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
     */
    vector<vector<complex<double>>> CoeffEncodingOneLevel(vector<complex<double>> pows,
                                                                         vector<uint32_tt> rotGroup,
                                                                         bool flag_i);
    /**
     * Computes the coefficients for the FFT decoding for CoeffDecodingCollapse such that every
     * iteration occupies one level.
     *
     * @param pows vector of roots of unity powers.
     * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
     * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
     */
    vector<vector<complex<double>>> CoeffDecodingOneLevel(vector<complex<double>> pows,
                                                                         vector<uint32_tt> rotGroup,
                                                                         bool flag_i);

    /**
     * Computes the coefficients for the given level budget for the FFT encoding. Needed in
     * EvalLTFFTPrecomputeEncoding.
     *
     * @param pows vector of roots of unity powers.
     * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
     * @param levelBudget the user specified budget for levels.
     * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
     */
    vector<vector<vector<complex<double>>>> CoeffEncodingCollapse(
            vector<complex<double>> pows, vector<uint32_tt> rotGroup, uint32_tt levelBudget,
            bool flag_i);

    /**
     * Computes the coefficients for the given level budget for the FFT decoding. Needed in
     * EvalLTFFTPrecomputeDecoding.
     *
     * @param pows vector of roots of unity powers.
     * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute FFT.
     * @param levelBudget the user specified budget for levels.
     * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
     */
    vector<vector<vector<complex<double>>>> CoeffDecodingCollapse(
            vector<complex<double>> pows, vector<uint32_tt> rotGroup, uint32_tt levelBudget,
            bool flag_i);

    vector<vector<PlaintextT*>> EvalCoeffsToSlotsPrecompute(const vector<complex<double>>& A,
                                                                    const vector<uint32_tt>& rotGroup,
                                                                    bool flag_i, NTL::RR scale = NTL::RR(1), uint32_tt target_level = 0);
    vector<vector<PlaintextT*>> EvalSlotsToCoeffsPrecompute(const vector<complex<double>>& A,
                                                                    const vector<uint32_tt>& rotGroup,
                                                                    bool flag_i, NTL::RR scale, uint32_tt target_level = 0);

    uint32_tt ReduceRotation(int index, int slots);


    Scheme_23& scheme;
    SecretKey& secretKey;

    int levelBudgetEnc;
    int levelBudgetDec;
    int is_STC_first;

    vector<vector<PlaintextT*>> m_U0hatTPreFFT; //CTS预处理矩阵
    vector<vector<PlaintextT*>> m_U0PreFFT; //STC预处理矩阵

    vector<int> m_paramsEnc;//CTS参数
    vector<int> m_paramsDec;//CTS参数

    vector<cuDoubleComplex> rotateTemp_host;
    cuDoubleComplex* rotateTemp_device;
    
    cudaStream_t stream_prefetch;

    // N * (p_num + q_num ) * 2
    uint64_tt* cipher_buffer_PQ;
    // N * t_num * Ri_blockNum * 2
    uint64_tt* PQ_to_T_temp;
	// N * t_num * Ri_blockNum * 2 * gs
	uint64_tt* gs_cipher_T;
    // N * t_num * Ri_blockNum * 2
    uint64_tt* T_to_PQ_temp;
    // N * t_num * Ri_blockNum * 2 * bs
    uint64_tt* bs_cipher_T;

    vector<vector<int>> rot_in_C2S;
    vector<vector<int>> rot_out_C2S;

    vector<vector<int>> rot_in_S2C;
    vector<vector<int>> rot_out_S2C;

    void Precompute_rot_in_out_C2S();
    void Precompute_rot_in_out_S2C();
    void addBootstrappingKey();
    void addC2SKey();
    void addS2CKey();


    void EvalCoeffsToSlots(vector<vector<PlaintextT*>>&A, Ciphertext&cipher);

    void EvalSlotsToCoeffs(vector<vector<PlaintextT*>>&A, Ciphertext&cipher);

    vector<int> rotIdx_C2S;
    vector<int> rotIdx_S2C;
    
    vector<uint64_tt*> rotKey_pointer;
    uint64_tt** rotKey_pointer_device;
        
    vector<int> rotSlots;
    int* rotSlots_device;

    vector<uint64_tt*> plaintextT_pointer;
    uint64_tt** plaintextT_pointer_device;
    vector<int> accNum_vec;
    int* accNum_vec_device;

    void Giant_Rotate_And_Mult_PlaintT(uint64_tt* bs_cipher_T, uint64_tt* PQ_to_T_temp, int gs_here, uint64_tt* modUp_QjtoT_temp, vector<uint64_tt*> rotKey_pointer, vector<int> rotSlots_vec, 
                                           vector<uint64_tt*> plaintextT_pointer, vector<int> accNum_vec, int bs, int gs, int l);
    void Baby_Rotate(uint64_tt* output, uint64_tt* modUp_QjtoT_temp, uint64_tt* input, int rotNum_here, vector<uint64_tt*> rotKey_pointer, vector<int> rotSlots_vec, int l);
};

//     //coeffstoslots  

//      //DCRT
//     typedef vector<vector<uint64_t>> DCRTPoly;
      

//     void MulScalarP(uint64_t* res, uint64_t* a, long l, long k = 0);

//     void MulScalarPInv(uint64_t* res, uint64_t* a, long l, long k = 0);

//     vector<uint64_t* > EvalFastRotationPrecompute(Ciphertext &cipher);

//     void EvalMultExt(Ciphertext &result, Ciphertext &cipher, Plaintext pt);
//     Ciphertext EvalMultExt(Ciphertext &cipher, Plaintext pt);

//     void EvalAddExtInPlace(Ciphertext &cipher, Ciphertext &ext);

//     uint64_t* InnerProduct(int levelQ, vector<uint64_t* > aPolyTs, vector<vector<vector<uint64_t>>> key);//return to PQ

//     void AddAndEqual(uint64_t* a, uint64_t* b, long l, long k);

//     void NTTAndEqual(uint64_t* a, long l, long k);

//     void PermuteNTTWithIndex(uint64_t* polOut, uint64_t *polIn, uint64_t* index, int l, int k);

//     // Ciphertext EvalFastRotationExt(Ciphertext cipher, int32_t index, vector<uint64_t* > digits, bool addFirst);
//     // Ciphertext EvalFastRotationExt(Ciphertext cipher, int32_t index, bool addFirst);
//     void EvalFastRotationExt(Ciphertext &result, Ciphertext &cipher, vector<uint64_t* >  digits, int32_t index, bool addFirst);
//     //对第一个元素进行模降QP-->Q
//     //input   cipher
//     //output  vector<vector<uint64>>
// //    uint64_t* KeySwitchDownFirstElement(Ciphertext &cipher);
//     //对密文进行模降QP-->Q
//     //input   cipher(mod QP)
//     //output  cipher(mod Q)
//     Ciphertext KeySwitchDown(Ciphertext &cipher);

//     Ciphertext KeySwitchDownFirst(Ciphertext &cipher);
//     // void KeySwitchDown(Ciphertext &cipher);

//     //对密文进行模升Q-->QP
//     //input   cipher(mod Q)
//     //output  cipher(mod QP)
//     void KeySwitchExt(Ciphertext& result, Ciphertext cipher);

//     void PQDecrypt( SecretKey &secretKey, Ciphertext &cipher, string s);

//     void PQRescaleAndDecrypt(SecretKey &secretKey, Ciphertext &cipher, string s);

//     void QDecrypt(SecretKey &secretKey, Ciphertext &cipher, string s);

//     void QRescaleAndDecrypt(SecretKey &secretKey, Ciphertext &cipher, string s);
