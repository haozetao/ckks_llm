#pragma once

#include "Bert.h"
#include "../ckks/include/TimeUtils.cuh"


Bert::Bert(string model_catalog, Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, Attention& attention_scheme, 
    Bootstrapper &bootstrapper, PCMM_Scheme& pcmm_scheme, SecretKey &sk)
    : context(context), scheme(scheme), scheme_algo(scheme_algo), attention_scheme(attention_scheme),
     sk(sk),model_weights(model_catalog), bootstrapper(bootstrapper),pcmm_scheme(pcmm_scheme)
{
    N = context.N;
    L = context.L;
    slots = context.slots;
    tmpcipher_buffer = new Ciphertext*[20];
    for (int i = 0; i < 20; ++i) {
        tmpcipher_buffer[i] = new Ciphertext(context.N, context.L, 1, context.slots, NTL::RR(context.precision));
    }

    vector<uint64_tt> p_ringpack = {context.pVec[context.p_num - 1]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();
    int mlwe_rank = context.N / PCMM_N1;
    // int decomp_num = mlwe_rank / 2;
    int decomp_num = 3072;
    for(int i = 0; i < decomp_num; i++){
        MLWECiphertext* mlwe_cipher = new MLWECiphertext(PCMM_N1, q_ringpack_count - 1, q_ringpack_count - 1, mlwe_rank, NTL::RR(context.precision));
        mlwe_cipher_buffer.push_back(mlwe_cipher);
    }
    column_num = slots/attention_scheme.token_len;
    cipher_num = attention_scheme.head_num / (column_num / attention_scheme.d);
    // prepare for LayerNorm gamma and beta
    // gamma
    cuDoubleComplex* gamma_buffer_host = new cuDoubleComplex[slots];
    cuDoubleComplex* gamma_buffer_device;
    cudaMalloc(&gamma_buffer_device, sizeof(cuDoubleComplex) * slots);
    float* tmp_gamma = (float *)malloc(sizeof(float)*column_num*cipher_num);
    
    double sqrt_n = sqrt(cipher_num * column_num);
    
    for (int layer = 0; layer < attention_scheme.head_num; layer++){
        cudaMemcpy(tmp_gamma, model_weights.attention_output_LayerNorm_gamma[layer], sizeof(float)*column_num*cipher_num, cudaMemcpyDeviceToHost);
        for (int i = 0; i < cipher_num; i++){
            Plaintext* gamma = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
            for (int j = 0; j < attention_scheme.token_len; j++){
                for (int k = 0; k < column_num; k++){
                    gamma_buffer_host[j*column_num+k].x = tmp_gamma[k + i*column_num] * tt * sqrt_n;
                    gamma_buffer_host[j*column_num+k].y = 0;
                }
            }
            cudaMemcpy(gamma_buffer_device, gamma_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(gamma_buffer_device, *gamma);
            attn_LayerNorm_gamma[layer].push_back(gamma);
        }
    }
    // beta
    for (int layer = 0; layer < 12; layer++){
        cudaMemcpy(tmp_gamma, model_weights.attention_output_LayerNorm_beta[layer], sizeof(float)*column_num*cipher_num, cudaMemcpyDeviceToHost);
        for (int i = 0; i < cipher_num; i++){
            Plaintext* beta = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
            for (int j = 0; j < attention_scheme.token_len; j++){
                for (int k = 0; k < column_num; k++){
                    gamma_buffer_host[j*column_num+k].x = tmp_gamma[k + i*column_num];
                    gamma_buffer_host[j*column_num+k].y = 0;
                }
            }
            cudaMemcpy(gamma_buffer_device, gamma_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(gamma_buffer_device, *beta);
            attn_LayerNorm_beta[layer].push_back(beta);
        }
    }

    // prepare for layer_output LayerNorm gamma and beta
    // gamma
    for (int layer = 0; layer < attention_scheme.head_num; layer++){
        cudaMemcpy(tmp_gamma, model_weights.output_LayerNorm_gamma[layer], sizeof(float)*column_num*cipher_num, cudaMemcpyDeviceToHost);
        for (int i = 0; i < cipher_num; i++){
            Plaintext* gamma = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
            for (int j = 0; j < attention_scheme.token_len; j++){
                for (int k = 0; k < column_num; k++){
                    gamma_buffer_host[j*column_num+k].x = tmp_gamma[k + i*column_num] * tt * sqrt_n;
                    gamma_buffer_host[j*column_num+k].y = 0;
                }
            }
            cudaMemcpy(gamma_buffer_device, gamma_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(gamma_buffer_device, *gamma);
            layer_output_LayerNorm_gamma[layer].push_back(gamma);
        }
    }
    // beta
    for (int layer = 0; layer < 12; layer++){
        cudaMemcpy(tmp_gamma, model_weights.output_LayerNorm_beta[layer], sizeof(float)*column_num*cipher_num, cudaMemcpyDeviceToHost);
        for (int i = 0; i < cipher_num; i++){
            Plaintext* beta = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
            for (int j = 0; j < attention_scheme.token_len; j++){
                for (int k = 0; k < column_num; k++){
                    gamma_buffer_host[j*column_num+k].x = tmp_gamma[k + i*column_num];
                    gamma_buffer_host[j*column_num+k].y = 0;
                }
            }
            cudaMemcpy(gamma_buffer_device, gamma_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(gamma_buffer_device, *beta);
            layer_output_LayerNorm_beta[layer].push_back(beta);
        }
    }
}

void Bert::mul_W_QKV(vector<Ciphertext*> &X, Bert_model_weights &model_weights, vector<Ciphertext*> &resQ, vector<Ciphertext*> &resK, vector<Ciphertext*> &resV, int layer){
    cout << "(*X[0]).L " << (*X[0]).L << endl;
    PCMM_Boot_768_768(model_weights.attention_self_query_weight[layer], X, resQ, (*X[0]).L-2, 0);
    cout << "111" << endl;
    PCMM_Boot_768_768(model_weights.attention_self_key_weight[layer], X, resK, (*X[0]).L-2, 0);
    cout << "222" << endl;
    PCMM_Boot_768_768(model_weights.attention_self_value_weight[layer], X, resV, 18, 0);
    cout << "333" << endl;
}

void Bert::attn_QK(vector<Ciphertext*> &Q, vector<Ciphertext*> &K, vector<Ciphertext*> &O){
    for (int i = 0; i < Q.size(); i++){
        attention_scheme.CCMM_QK(*Q[i], *K[i], *O[2*i], *O[2*i+1]);
    }
}

void Bert::attn_Softmax_phase1(vector<Ciphertext*> &O){
    for (int i = 0; i < O.size()/2; i++){
        vector<Ciphertext *> Softmax_input = {O[2*i], O[2*i+1]};
        double sqrt_d = sqrt(attention_scheme.d);
        (*O[2*i]).scale = (*O[2*i]).scale * sqrt_d;
        (*O[2*i+1]).scale = (*O[2*i+1]).scale * sqrt_d;
        attention_scheme.evalSoftMax_phase1(Softmax_input);
    }
}

void Bert::attn_Softmax_phase2(vector<Ciphertext*> &O){
    for (int i = 0; i < O.size()/2; i++){
        vector<Ciphertext *> Softmax_input = {O[2*i], O[2*i+1]};
        cout << "before evalSoftMax_phase2 scale " << O[2*i]->scale << endl;
        attention_scheme.evalSoftMax_phase2(Softmax_input);
        cout << "after evalSoftMax_phase2 scale " << O[2*i]->scale << endl;
    }
}

void Bert::attn_output(vector<Ciphertext*> &O, int layer){
    cout<<"O.level before attn_output PCMM: "<<(*O[0]).l<<endl;
    cout<<"O.level before attn_output PCMM: "<<(*O[1]).l<<endl;
    cout<<"O.level before attn_output PCMM: "<<(*O[2]).l<<endl;
    // dense
    CUDATimer cuTimer;
    cuTimer.start();
    PCMM_Boot_768_768(model_weights.attention_output_dense_weight[layer], O, O, (*O[0]).L, 1);
    cout << "attn_output pcmm time "  << cuTimer.stop() << endl;

    cout<<"O.level before LayerNorm phase 1: "<<(*O[0]).l<<endl;
    cout<<"O.scale before LayerNorm phase 1: "<<(*O[0]).scale<<endl;
    // LayerNorm
    cuTimer.start();
    attention_scheme.LayerNorm_Bert(O, sk);
    cout<<"O.level after LayerNorm phase 1: "<<(*O[0]).l<<endl;
    cout<<"O.scale after LayerNorm phase 1: "<<(*O[0]).scale<<endl;
    for (int i = 0; i < O.size(); i++){
        scheme.multConstAndEqual(*O[i], *attn_LayerNorm_gamma[layer][i]);
        scheme.rescaleAndEqual(*O[i]);
        scheme.addConstAndEqual(*O[i], *attn_LayerNorm_beta[layer][i]);
    }
    cout << "attn_output_layernorm time "  << cuTimer.stop() << endl;
    cout<<"O.level after LayerNorm phase 2: "<<(*O[0]).l<<endl;
    cout<<"O.scale after LayerNorm phase 2: "<<(*O[0]).scale<<endl;
}

void Bert::intermediate(vector<Ciphertext*> &O, int layer, vector<Ciphertext*> &res){
    CUDATimer cuTimer;
    cuTimer.start();
    EncodingMatrix& encodingMatrix = bootstrapper.encodingMatrix;
    int target_level = (*O[0]).L-4;
    int mat_M = 3072;
    int mat_N = 768;
    
    scheme.mulByiAndEqual(*O[1]);
    scheme.addAndEqual(*O[0], *O[1]);

    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *O[0]);
    pcmm_scheme.rlweCipherDecompose(*O[0], mlwe_cipher_buffer, 512, 0);

    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *O[2]);
    pcmm_scheme.rlweCipherDecompose(*O[2], mlwe_cipher_buffer, 256, 512);

    pcmm_scheme.PPMM(model_weights.intermediate_dense_weight[layer], mlwe_cipher_buffer, mat_M, mat_N, 128);

    for (int i = 0; i < mat_M / 512; i++){
        pcmm_scheme.mlweCipherPacking(*res[2*i], mlwe_cipher_buffer, 512, i*512);
        bootstrapper.modUpQ0toQL(*res[2*i], target_level);
        encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, *res[2*i]);
        scheme.conjugate_23(*bootstrapper.ctReal, *res[2*i]);
        scheme.sub(*bootstrapper.ctImag, *res[2*i], *bootstrapper.ctReal);
        scheme.addAndEqual(*bootstrapper.ctReal, *res[2*i]);

        scheme.divByiAndEqual(*bootstrapper.ctImag);
        bootstrapper.EvalModAndEqual(*bootstrapper.ctReal);
        bootstrapper.EvalModAndEqual(*bootstrapper.ctImag);

        scheme.multConstAndEqual(*bootstrapper.ctReal, 256./16*16);
        scheme.multConstAndEqual(*bootstrapper.ctImag, 256./16*16);

        *res[i*2] = *bootstrapper.ctReal;
        *res[i*2+1] = *bootstrapper.ctImag;
    }
    cout << "intermediate pcmm time "  << cuTimer.stop() << endl;
    cout <<"res[0].scale before evalGeLU: "<<(*res[0]).scale<<endl;
    cuTimer.start();
    for (int i = 0; i < res.size(); i++){
        attention_scheme.evalGeLU(*res[i]);
    }
    cout << "intermediate evalGeLU time "  << cuTimer.stop() << endl;
    cout <<"res[0].scale after evalGeLU: "<<(*res[0]).scale<<endl;
}

void Bert::layer_output(vector<Ciphertext*> &O, int layer, vector<Ciphertext*> &layer_res){
    CUDATimer cuTimer;
    cuTimer.start();
    EncodingMatrix& encodingMatrix = bootstrapper.encodingMatrix;
    int target_level = (*O[0]).L;
    int mat_M = 768;
    int mat_N = 3072;
    for (int i = 0; i < mat_N / 512; i++){
        scheme.mulByiAndEqual(*O[i*2+1]);
        scheme.addAndEqual(*O[i*2], *O[i*2+1]);

        encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *O[i*2]);
        pcmm_scheme.rlweCipherDecompose(*O[i*2], mlwe_cipher_buffer, 512, i*512);
    }

    pcmm_scheme.PPMM(model_weights.output_dense_weight[layer], mlwe_cipher_buffer, mat_M, mat_N, 128);

    pcmm_scheme.mlweCipherPacking(*O[0], mlwe_cipher_buffer, 512, 0);
    pcmm_scheme.mlweCipherPacking(*O[2], mlwe_cipher_buffer, 256, 512);

    bootstrapper.modUpQ0toQL(*O[0], target_level);

    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, *O[0]);

    // real = a-bi
    scheme.conjugate_23(*bootstrapper.ctReal, *O[0]);

    // imag = cipher - real = 2bi
    scheme.sub(*bootstrapper.ctImag, *O[0], *bootstrapper.ctReal);
    // real = real + cipher = 2a
    scheme.addAndEqual(*bootstrapper.ctReal, *O[0]);

    scheme.divByiAndEqual(*bootstrapper.ctImag);
    bootstrapper.EvalModAndEqual(*bootstrapper.ctReal);
    bootstrapper.EvalModAndEqual(*bootstrapper.ctImag);

    scheme.multConstAndEqual(*bootstrapper.ctReal, 256./16*16);
    scheme.multConstAndEqual(*bootstrapper.ctImag, 256./16*16);

    *O[0] = *bootstrapper.ctReal;
    *O[1] = *bootstrapper.ctImag;


    bootstrapper.modUpQ0toQL(*O[2], target_level);
    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, *O[2]);
    bootstrapper.EvalModAndEqual(*O[2]);
    scheme.multConstAndEqual(*O[2], 256./16*16);
    cout << "layer_output pcmm time "  << cuTimer.stop() << endl;
    cuTimer.start();
    vector<Ciphertext*> LayerNorm_input = {O[0], O[1], O[2]};
    
    cout << "before LayerNorm_phase1 LayerNorm_input[0] scale " << LayerNorm_input[0]->scale << endl;
    // LayerNorm
    attention_scheme.LayerNorm_Bert(LayerNorm_input, sk);
    cout << "after LayerNorm_phase1 LayerNorm_input[0] scale " << LayerNorm_input[0]->scale << endl;
    for (int i = 0; i < 3; i++){
        scheme.multConstAndEqual(*LayerNorm_input[i], *layer_output_LayerNorm_gamma[layer][i]);
        scheme.rescaleAndEqual(*LayerNorm_input[i]);
        scheme.addConstAndEqual(*LayerNorm_input[i], *layer_output_LayerNorm_beta[layer][i]);
    }
    cout << "layer_output LayerNorm time "  << cuTimer.stop() << endl;
    cout << "after LayerNorm_phase2 LayerNorm_input[0] scale " << LayerNorm_input[0]->scale << endl;

    scheme.mulByiAndEqual(*LayerNorm_input[1]);
    scheme.addAndEqual(*LayerNorm_input[0], *LayerNorm_input[1]);
    *layer_res[0] = *LayerNorm_input[0];
    *layer_res[1] = *LayerNorm_input[2];
    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *layer_res[0]);
    encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *layer_res[1]);
    
    cout << "after layer_output LayerNorm and s2c level " << layer_res[0]->l << endl;
    cout << "after layer scale " << layer_res[0]->scale << endl;
}

void Bert::attn_mulV(vector<Ciphertext*> &S, vector<Ciphertext*> &V, vector<Ciphertext*> &O){
    for (int i = 0; i < V.size(); i++){
        attention_scheme.TauAndEqual(*V[i]);
        attention_scheme.CCMM_V(*S[2*i], *S[2*i+1], *V[i], *O[i]);
    }
}

void Bert::boot(vector<Ciphertext*> &O){
    for (int i = 0; i < O.size(); i++){
        bootstrapper.Bootstrapping(*O[i]);
    }
}

void Bert::PCMM_Boot_768_768(float* plain_mat, vector<Ciphertext*>& rlwe_cipher, vector<Ciphertext*>& res_cipher, int target_level, int do_s2c)
{
    int K = scheme.context.K;
    int L = scheme.context.L;
    int N = scheme.context.N;
    
    int N1 = pcmm_scheme.pcmm_context.N1;
    int mlwe_rank = pcmm_scheme.pcmm_context.mlwe_rank;
    int ringpack_p_count = pcmm_scheme.pcmm_context.ringpack_p_count;
    int ringpack_q_count = pcmm_scheme.pcmm_context.ringpack_q_count;
    int ringpack_pq_count = pcmm_scheme.pcmm_context.ringpack_pq_count;

    int mlwe_num = mlwe_cipher_buffer.size();
    cout << "mlwe_num = " << mlwe_num << endl;
    
    EncodingMatrix& encodingMatrix = bootstrapper.encodingMatrix;

    if(do_s2c == 1){
        *res_cipher[0] = *rlwe_cipher[0];
        *res_cipher[1] = *rlwe_cipher[1];
        *res_cipher[2] = *rlwe_cipher[2];
        cout << "pcmm input level " << res_cipher[0]->l << endl;
        
        scheme.mulByiAndEqual(*res_cipher[1]);
        scheme.addAndEqual(*res_cipher[0], *res_cipher[1]);

        encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *res_cipher[0]);
        pcmm_scheme.rlweCipherDecompose(*res_cipher[0], mlwe_cipher_buffer, 512, 0);

        encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, *res_cipher[2]);
        pcmm_scheme.rlweCipherDecompose(*res_cipher[2], mlwe_cipher_buffer, 256, 512);
    }
    else{
        pcmm_scheme.rlweCipherDecompose(*rlwe_cipher[0], mlwe_cipher_buffer, 512, 0);
        pcmm_scheme.rlweCipherDecompose(*rlwe_cipher[1], mlwe_cipher_buffer, 256, 512);
    }
    CUDATimer cuTimer;
    cuTimer.start();
    
    pcmm_scheme.PPMM(plain_mat, mlwe_cipher_buffer, 768, 768, 128);
    cout << "ppmm time "  << cuTimer.stop() << endl;

    pcmm_scheme.mlweCipherPacking(*res_cipher[0], mlwe_cipher_buffer, 512, 0);

    pcmm_scheme.mlweCipherPacking(*res_cipher[2], mlwe_cipher_buffer, 256, 512);

    // // scheme.decrypt_display(scheme_algo.secretkey, cipher, "ctReal after stc ");
	// // Step 1: scale to q0/|m|
    // // q0 / message_ratio = q0 / 4.0 == input.scale
	// // Step 2 : Extend the basis from q to Q
    bootstrapper.modUpQ0toQL(*res_cipher[0], target_level);

    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, *res_cipher[0]);

    // real = a-bi
    scheme.conjugate_23(*bootstrapper.ctReal, *res_cipher[0]);

    // imag = cipher - real = 2bi
    scheme.sub(*bootstrapper.ctImag, *res_cipher[0], *bootstrapper.ctReal);
    // real = real + cipher = 2a
    scheme.addAndEqual(*bootstrapper.ctReal, *res_cipher[0]);

    scheme.divByiAndEqual(*bootstrapper.ctImag);
    bootstrapper.EvalModAndEqual(*bootstrapper.ctReal);
    bootstrapper.EvalModAndEqual(*bootstrapper.ctImag);

    scheme.multConstAndEqual(*bootstrapper.ctReal, 256./16*16);
    scheme.multConstAndEqual(*bootstrapper.ctImag, 256./16*16);

    *res_cipher[0] = *bootstrapper.ctReal;
    *res_cipher[1] = *bootstrapper.ctImag;


    bootstrapper.modUpQ0toQL(*res_cipher[2], target_level);
    cout << "after modUpQ0toQL level " << res_cipher[2]->l << endl;
    encodingMatrix.EvalCoeffsToSlots(encodingMatrix.m_U0hatTPreFFT, *res_cipher[2]);
    bootstrapper.EvalModAndEqual(*res_cipher[2]);
    cout << "after EvalModAndEqual level " << res_cipher[2]->l << endl;
    scheme.multConstAndEqual(*res_cipher[2], 256./16*16);
    cout << "after multConstAndEqual level " << res_cipher[2]->l << endl;

    
    // // Real part * 2
    // scheme.addAndEqual(rlwe_cipher, *bootstrapper.ctReal);
    // // // c1.l -= 4;
    // bootstrapper.EvalModAndEqual(rlwe_cipher);

    // scheme.multConstAndEqual(rlwe_cipher, 256./16*16);
}

void Bert::infer(vector<Ciphertext*> &encrypted_token){
    cout << "begin infer" << endl;
    cudaEvent_t start1, end1;
    cudaEventCreate(&start1);
    cudaEventCreate(&end1);
    CUDATimer cuTimer1;
    

    float QKV = 10000, QmulK = 10000, softmax1 = 10000, boot_time = 10000, softmax2 = 10000, mulV = 10000;
    float attn_output_time = 10000, intermediate_time = 10000, layer_output_time = 10000;
    float temp1 = 0;
    for (int i = 0; i < 2; i++){
        *tmpcipher_buffer[i] = *encrypted_token[i];
    }
    
    int layer = 0;
    for (int layer = 0; layer < 12; layer++)
    {
        // QKV
        vector<Ciphertext*> X = {tmpcipher_buffer[0], tmpcipher_buffer[1]};
        vector<Ciphertext*> Q = {tmpcipher_buffer[2], tmpcipher_buffer[3], tmpcipher_buffer[4]};
        vector<Ciphertext*> K = {tmpcipher_buffer[5], tmpcipher_buffer[6], tmpcipher_buffer[7]};
        vector<Ciphertext*> V = {tmpcipher_buffer[8], tmpcipher_buffer[9], tmpcipher_buffer[10]};
        cout<<"X.level before qkv: "<<(*X[0]).l<<endl;
        // scheme.decrypt_display(sk, *X[0], "before mul_W_QKV *X[0]");
        cudaEventRecord(start1);
            mul_W_QKV(X, model_weights, Q, K, V, layer);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        QKV = min(QKV, temp1);
        // scheme.decrypt_display(sk, *Q[0], "after mul_W_QKV *Q[0]");
        // scheme.decrypt_display(sk, *K[0], "after mul_W_QKV *K[0]");
        // scheme.decrypt_display(sk, *V[0], "after mul_W_QKV *V[0]");
        cout<<"Q.level after qkv: "<<(*Q[0]).l<<endl;

        cout<<"Q.level before qk: "<<(*Q[0]).l<<endl;
        cout<<"Q.scale before qk: "<<(*Q[0]).scale<<endl;
        // Q*K^T
        vector<Ciphertext*> O = {tmpcipher_buffer[0], tmpcipher_buffer[1], tmpcipher_buffer[2], tmpcipher_buffer[5], tmpcipher_buffer[3], tmpcipher_buffer[6]};
        cudaEventRecord(start1);
            attn_QK(Q, K, O);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        QmulK = min(QmulK, temp1);
        cout<<"O.level after qk: "<<(*O[0]).l<<endl;
        cout<<"O.scale after qk: "<<(*O[0]).scale<<endl;
        // scheme.decrypt_display(sk, *O[0], "after attn_QK *O[0]");
        // scheme.decrypt_display(sk, *O[1], "after attn_QK *O[1]");
        // scheme.decrypt_display(sk, *O[2], "after attn_QK *O[2]");
        // scheme.decrypt_display(sk, *O[3], "after attn_QK *O[3]");
        // scheme.decrypt_display(sk, *O[4], "after attn_QK *O[4]");
        // scheme.decrypt_display(sk, *O[5], "after attn_QK *O[5]");
        
        cout<<"O.scale before attn_Softmax_phase1: "<<(*O[0]).scale<<endl;
        cudaEventRecord(start1);
            attn_Softmax_phase1(O);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        softmax1 = min(softmax1, temp1);
        cout<<"O.scale after attn_Softmax_phase1: "<<(*O[0]).scale<<endl;
        
        cout<<"O.level after softmax1: "<<(*O[0]).l<<endl;
        cudaEventRecord(start1);
            boot(O);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        boot_time = min(boot_time, temp1);
        
        cout<<"O.scale before softmax2: "<<(*O[0]).scale<<endl;
        cout<<"O.level before softmax2: "<<(*O[0]).l<<endl;
        cudaEventRecord(start1);
            attn_Softmax_phase2(O);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        softmax2 = min(softmax2, temp1);
        cout<<"O.scale after softmax2: "<<(*O[0]).scale<<endl;
        
        cout<<"O.level after softmax2: "<<(*O[0]).l<<endl;
        vector <Ciphertext*> mulV_res = {tmpcipher_buffer[4], tmpcipher_buffer[7], tmpcipher_buffer[11]};
        cout<<"O.level before mulV: "<<(*O[0]).l<<endl;
        cout<<"V.level before mulV: "<<(*V[0]).l<<endl;
        cout<<"O.scale before mulV: "<<(*O[0]).scale<<endl;
        cout<<"V.scale before mulV: "<<(*V[0]).scale<<endl;
        cudaEventRecord(start1);
            attn_mulV(O, V, mulV_res);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        mulV = min(mulV, temp1);

        cout<<"mulV_res.level after mulV: "<<(*mulV_res[0]).l<<endl;
        cout<<"mulV_res.scale after mulV: "<<(*mulV_res[0]).scale<<endl;
        cudaEventRecord(start1);
            attn_output(mulV_res,layer);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        attn_output_time = min(attn_output_time, temp1);
        
        vector <Ciphertext*> intermediate_res = {tmpcipher_buffer[2], tmpcipher_buffer[3], tmpcipher_buffer[5], tmpcipher_buffer[6], tmpcipher_buffer[8], tmpcipher_buffer[9],
            tmpcipher_buffer[10], tmpcipher_buffer[12], tmpcipher_buffer[13], tmpcipher_buffer[14], tmpcipher_buffer[15], tmpcipher_buffer[16]};
        cudaEventRecord(start1);
            intermediate(mulV_res, layer, intermediate_res);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        intermediate_time = min(intermediate_time, temp1);
        
        vector <Ciphertext*> layer_res = {tmpcipher_buffer[0], tmpcipher_buffer[1]};
        cudaEventRecord(start1);
            layer_output(intermediate_res, layer, layer_res);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        cudaEventElapsedTime(&temp1, start1, end1);
        layer_output_time = min(layer_output_time, temp1);

        cout << "\n\n\n\n\n" << endl;
    }
    printf("Time: QKV: %f ms\n", QKV);
    printf("Time: QmulK: %f ms\n", QmulK);
    printf("Time: softmax1: %f ms\n", softmax1);
    printf("Time: boot_time: %f ms\n", boot_time);
    printf("Time: softmax2: %f ms\n", softmax2);
    printf("Time: mulV: %f ms\n", mulV);
    printf("Time: attn_output_time: %f ms\n", attn_output_time);
    printf("Time: intermediate_time: %f ms\n", intermediate_time);
    printf("Time: layer_output_time: %f ms\n", layer_output_time);

        


}