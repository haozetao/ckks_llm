#pragma once

#include <string>

using namespace std;

#include "Load_bert.cuh"
#include "../ckks/attention/Attention.cuh"
#include "../ckks/include/bootstrapping/Bootstrapper.cuh"
#include "../ckks/include/pcmm/PCMM_Context.cuh"
#include "../ckks/include/pcmm/PCMM_Scheme.cuh"
#include "../ckks/attention/Attention.cuh"


class Bert{
public:
    Context_23 &context;
    Scheme_23 &scheme;
    SchemeAlgo &scheme_algo;
    Attention &attention_scheme;
    SecretKey &sk;
    Bert_model_weights model_weights;
    Bootstrapper &bootstrapper;
    PCMM_Scheme& pcmm_scheme;
    Bert(string model_catalog, Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, Attention& attention_scheme, Bootstrapper &bootstrapper, PCMM_Scheme& pcmm_scheme, SecretKey &sk);
    Ciphertext** tmpcipher_buffer;
    vector<MLWECiphertext*> mlwe_cipher_buffer;
    int PCMM_N1 = 128;
    int q_ringpack_count;
    int mlwe_rank;
    int mat_M = 768, mat_N = 768;
    int N;
    int L;
    int slots;

    int cipher_num;
    int column_num;
    double tt = 0.00038907124; // upper_bound = 10000
    vector<Plaintext*> attn_LayerNorm_gamma[12];
    vector<Plaintext*> attn_LayerNorm_beta[12];
    vector<Plaintext*> layer_output_LayerNorm_gamma[12];
    vector<Plaintext*> layer_output_LayerNorm_beta[12];
    vector<Plaintext*> layer_query_bias[12];
    vector<Plaintext*> layer_key_bias[12];
    vector<Plaintext*> layer_value_bias[12];

    
    

    // pcmm
    void mul_W_QKV(vector<Ciphertext*> &X, Bert_model_weights &model_weights, vector<Ciphertext*> &resQ, vector<Ciphertext*> &resK, vector<Ciphertext*> &resV, int layer);
    // ccmm Q*K^T
    void attn_QK(vector<Ciphertext*> &Q, vector<Ciphertext*> &K, vector<Ciphertext*> &O);
    // ccmm S*V
    void attn_mulV(vector<Ciphertext*> &S, vector<Ciphertext*> &V, vector<Ciphertext*> &O);
    void attn_Softmax_phase1(vector<Ciphertext*> &O);
    void attn_Softmax_phase2(vector<Ciphertext*> &O);
    // including a linear transformation and LayerNorm
    void attn_output(vector<Ciphertext*> &O, int layer);
    void intermediate(vector<Ciphertext*> &O, int layer, vector<Ciphertext*> &res);
    void layer_output(vector<Ciphertext*> &O, int layer, vector<Ciphertext*> &layer_res);
    void boot(vector<Ciphertext*> &O);
    void PCMM_Boot_768_768(float* plain_mat, vector<Ciphertext*>& rlwe_cipher, vector<Ciphertext*>& res_cipher, int target_level, int do_s2c);
    void test_PCMM_Boot_768_768(float* plain_mat, vector<Ciphertext*>& rlwe_cipher, vector<Ciphertext*>& res_cipher, int target_level, int do_s2c);
    
    void infer(vector<Ciphertext*> &encrypted_token);

};