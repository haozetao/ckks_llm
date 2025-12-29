#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include "cublas_v2.h"

using namespace std;

#include "src/ckks/include/Ciphertext.cuh"
#include "src/ckks/include/Context_23.cuh"
#include "src/ckks/include/Plaintext.cuh"
#include "src/ckks/include/Scheme_23.cuh"
#include "src/ckks/include/bootstrapping/Bootstrapper.cuh"
#include "src/ckks/include/TimeUtils.cuh"
#include "src/ckks/include/precision.cuh"
#include "src/ckks/attention/Attention.cuh"
#include "src/bert/Bert.cuh"


void read_matrix_from_file(const std::string& filename, cuDoubleComplex* mes, int size);
void load_bert_input(const std::string& filename, vector<cuDoubleComplex*>& mes, int token_len, int column_num, int cipher_num);
int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;
    int PCMM_N1 = 128;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    CUDATimer cuTimer;

    Context_23 context(logN, logslots, 192);
    cout<<"Generate Context OK"<<endl;

    // for(auto v : context.pqtVec)
    // {
    //     printf("%llu, ", v);
    // }

    int N = context.N;
    // int K = context.K;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = context.slots;

    int mlwe_rank = N / PCMM_N1;
    vector<uint64_tt> p_ringpack = {context.pVec[context.p_num - 1]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();

    Scheme_23 scheme(context);
    cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    cout<<"Generate sk OK"<<endl;

    scheme.mallocMemory();
    scheme.addEncKey(sk);
    cout<<"Generate pk OK"<<endl;

    

float gen_swk = 1000;
float enc = 1000, dec = 1000, resc = 10000000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
float hadd = 1000, cadd = 1000, hmult = 1000, cmult = 1000;
float rotate = 1000, conjugate = 1000;
float bootstrapping = 1000;
float ccmm_time = 1000;
float temp = 0;

        cudaEventRecord(start);
            scheme.addMultKey_23(sk);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        gen_swk = min(gen_swk, temp);
            scheme.addLeftRotKey_23(sk, 1);
            scheme.addConjKey_23(sk);


    cout<<"Generate rlk OK"<<endl;
    
    int is_STC_first = 1;
    SchemeAlgo scheme_algo(context, scheme, sk);
        scheme_algo.malloc_bsgs_buffer(context.eval_sine_chebyshev_coeff.size());
    EncodingMatrix encodingMatrix(sk, scheme, 4, 4, is_STC_first);
    Bootstrapper bootHelper(context, scheme, scheme_algo, sk, encodingMatrix, is_STC_first);
        bootHelper.addBootstrappingKey(sk);

    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme, bootHelper, sk);

    int pq_ringpack_count = pcmm_context.pq_ringpack.size();

    MLWESecretKey mlwe_sk(PCMM_N1, pq_ringpack_count, mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);
    // pcmm_scheme.addRepakcingKey(mlwe_sk, sk);

    Attention attention_scheme(context, scheme, scheme_algo, 128, 12, 64, sk);
    attention_scheme.addKey(sk);
    string model_catalog = "data/bert/";
    Bert bert(model_catalog, context, scheme, scheme_algo, attention_scheme, bootHelper, pcmm_scheme, sk);
    pcmm_scheme.addRepakcingKey(mlwe_sk, sk);
    
    int cipher_num = 3;
    vector<cuDoubleComplex*> message_host;
    vector<cuDoubleComplex*> message_device;
    
    // load_bert_input("hidden_states.bin", message_host, 128, 768, 3);
    for (int i = 0; i < cipher_num; i++){
        cuDoubleComplex* data = new cuDoubleComplex[slots];
        randomComplexArray(data, slots, -0.05, 0.1);
        message_host.push_back(data);
    }

    for (int i = 0; i < cipher_num; i++){
        cuDoubleComplex* mes_device;
        cudaMalloc(&mes_device, sizeof(cuDoubleComplex) * slots);
        cudaMemcpy(mes_device, message_host[i], sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        message_device.push_back(mes_device);
    }
    // float* plain_mat = (float *)malloc(sizeof(float)*768*768);
    // float* mul_res = (float *)malloc(sizeof(float)*128*768);
    // for (int i = 0; i < 768*768; i++){
    //     plain_mat[i] = 1.0;
    //     // plain_mat[i] = (i%100)/10000.0;
    // }
    // cudaMemcpy(plain_mat, bert.model_weights.attention_self_query_weight[0], sizeof(float)*768*768, cudaMemcpyDeviceToHost);
    // float* plain_mat_device;
    // cudaMalloc(&plain_mat_device, sizeof(float) * 768 * 768);
    // cudaMemcpy(plain_mat_device, plain_mat, sizeof(float) * 768 * 768, cudaMemcpyHostToDevice);
    // memset(mul_res, 0, sizeof(float)*128*768);
    // for (int i = 0; i < 128; i++){
    //     for (int j = 0; j < 768; j++){
    //         for (int k = 0; k < 768; k++){
    //             if (k < 256){
    //                 mul_res[i*768+j] += message_host[0][i*256+k].x * plain_mat[768*j+k];
    //             }
    //             else if (k < 512){
    //                 mul_res[i*768+j] += message_host[1][i*256+k-256].x * plain_mat[768*j+k];
    //             }
    //             else {
    //                 mul_res[i*768+j] += message_host[2][i*256+k-512].x * plain_mat[768*j+k];
    //             }
    //         }
    //     }
    // }
    // cout << "mul_res[0]= " << mul_res[0] << endl;
    // cout << "mul_res[1]= " << mul_res[1] << endl;
    // cout << "mul_res[256]= " << mul_res[256] << endl;
    // cout << "mul_res[512]= " << mul_res[512] << endl;
    
    
    /*************************************************************************** */

    // int target_level = is_STC_first ? bootHelper.encodingMatrix.levelBudgetDec : 0;
    // target_level = L - 3 - 9;
    int s2c_level_cost = 3, c2s_level_cost = 3;
    int target_level = 1 + s2c_level_cost;
    // for(target_level; target_level >= 0; target_level--)
    {
        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m2(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m3(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c3(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m1_dec(N, L, target_level, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec, * complex_msg_dec_host;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        
        for(int i = 0; i < 1; i++)
        {
            cudaEventRecord(start);
                context.encode(message_device[0], plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            ecd = min(ecd, temp);
                context.encode(message_device[1], plain_m2);
                context.encode(message_device[2], plain_m3);

            cudaEventRecord(start);
                scheme.encryptMsg(c1, plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            enc = min(enc, temp);
                scheme.encryptMsg(c2, plain_m2);
                scheme.encryptMsg(c3, plain_m3);
            
            scheme.decrypt_display(sk, c1, "c1");
            scheme.decrypt_display(sk, c2, "c2");
            
            // scheme.mulByiAndEqual(c2);
            // scheme.addAndEqual(c1, c2);

            scheme.decrypt_display(sk, c3, "c3");

            // encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, c1);
            // pcmm_scheme.realData_coeffShift(c1, 0, 1);

            // encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, c3);
            // pcmm_scheme.realData_coeffShift(c3, 0, 1);


            vector<Ciphertext*> rlwe_cipher;
            rlwe_cipher.push_back(&c1);
            rlwe_cipher.push_back(&c2);
            rlwe_cipher.push_back(&c3);
            vector<Ciphertext*> res;
            res.push_back(new Ciphertext(N, L, L, slots, NTL::RR(context.precision)));
            res.push_back(new Ciphertext(N, L, L, slots, NTL::RR(context.precision)));
            res.push_back(new Ciphertext(N, L, L, slots, NTL::RR(context.precision)));
            
            cudaEventRecord(start);
                // scheme.decrypt_display(sk, *rlwe_cipher[0], "before infer");
                bert.infer(rlwe_cipher);
                // scheme.decrypt_display(sk, *rlwe_cipher[0], "before infer");
                // scheme.rescaleAndEqual(c1);
                // scheme.rescaleAndEqual(c2);
                // bert.PCMM_Boot_768_768(plain_mat_device, rlwe_cipher, res, c1.L, 1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            resc = min(resc, temp);
            // scheme.addLeftRotKey_23(sk, 2);
            // scheme.decrypt_display(sk, c1, "before rot");
            // scheme.leftRotateAddSelf_23(c1, 2);
            // scheme.decrypt_display(sk, c1, "after rot");

            // scheme.decrypt_display(sk, *res[0], "*res[0]");
            // scheme.decrypt_display(sk, *res[1], "*res[1]");
            // scheme.decrypt_display(sk, *res[2], "*res[2]");


            // verify pcmm in coffe
            // scheme.decryptMsg(m1_dec, sk, *res[0]);
            // context.decode_coeffs(m1_dec, decode_output);
            // print_device_array(decode_output, 1<<16, "res1");
            // double* decode_output_host = (double*)malloc(sizeof(double)*128*512);
            // cudaMemcpy(decode_output_host, decode_output, sizeof(double)*128*512, cudaMemcpyDeviceToHost);
            // float max_error = 0;
            // for (int i = 0; i < 128; i++){
            //     for (int j = 0; j < 512; j++){
            //         max_error = max(abs(decode_output_host[i*512+j] - mul_res[i*768+j]), max_error);
            //     }
            // }
            // cout << "max_error = " << max_error << endl;
            
            
            // scheme.decryptMsg(m1_dec, sk, *res[2]);
            // context.decode_coeffs(m1_dec, decode_output);
            // print_device_array(decode_output, 1<<16, "res2");
            // cudaMemcpy(decode_output_host, decode_output, sizeof(double)*128*512, cudaMemcpyDeviceToHost);
            // max_error = 0;
            // for (int i = 0; i < 128; i++){
            //     for (int j = 0; j < 256; j++){
            //         max_error = max(abs(decode_output_host[i*512+j*2] - mul_res[i*768+j+512]), max_error);
            //     }
            // }
            // cout << "max_error = " << max_error << endl;


            cudaEventRecord(start);
                // scheme.decryptMsg(m1_dec, sk, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&dec, start, end);
            dec = min(dec, temp);

            
        }

        printf("Time: encode: %f ms decode: %f ms\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f ms dec: %f ms\n", enc*1000, dec*1000);
        printf("Time: hadd: %f ms cadd: %f ms\n", hadd*1000, cadd*1000);
        printf("Time: hmult: %f ms cmult: %f ms\n", hmult*1000, cmult*1000);
        printf("Time: rotate: %f ms conjugate: %f ms\n", rotate*1000, conjugate*1000);
        printf("Time: rescale: %f ms\n", resc);
        printf("Time: bootstrapping: %f ms\n", bootstrapping);
        printf("Time: ccmm: %f ms\n", ccmm_time);
        printf("gen swk: %f ms\n", gen_swk);
        printf("Time: ntt: %f ms intt: %f ms\n\n", ntt*1000, intt*1000);
        
        

        getchar();
    }


    return 0;
}

void read_matrix_from_file(const std::string& filename, cuDoubleComplex* mes, int size) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    std::istringstream iss(content);
    std::string token;
    int count = 0;

    while (std::getline(iss, token, ' ') && count < size) {
        mes[count].x = std::stod(token);
        mes[count].y = 0.0;  // 虚部为 0
        count++;
    }

    file.close();
}

void load_bert_input(const std::string& filename, vector<cuDoubleComplex*>& mes, int token_len, int column_num, int cipher_num) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
    }
    for (int i = 0; i < cipher_num; i++){
        cuDoubleComplex* data = (cuDoubleComplex*)malloc(32768 * sizeof(cuDoubleComplex));
        mes.push_back(data);
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> floatData(token_len * column_num);
    file.read(reinterpret_cast<char*>(floatData.data()), fileSize);
    file.close();

    for (int sed = 0; sed < cipher_num; sed++){
        for (int i = 0; i < token_len; i++) {
            for (int j = 0; j < 256; j++) {
                mes[sed][i*256+j].x = static_cast<double>(floatData[i * column_num + j + 256 * sed]);
                mes[sed][i*256+j].y = 0;
            }
        }
    }
}

