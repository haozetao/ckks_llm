#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "src/ckks/include/Context_23.cuh"
#include "src/ckks/include/TimeUtils.cuh"
#include "src/ckks/include/pcmm/PCMM_Context.cuh"
#include "src/ckks/include/pcmm/PCMM_Scheme.cuh"
#include "src/ckks/include/precision.cuh"

int main(int argc, char* argv[])
{
    if(argc != 3) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;
    int PCMM_N1 = atoi(argv[2]);

    Context_23 context(logN, logslots, 192);
    Scheme_23 scheme(context);
    cudaDeviceSynchronize();
    cout<<"Generate Context OK"<<endl;
    printf("logN: %d Pnum: %d Qnum: %d Tnum: %d dnum: %d gamma: %d\n", logN, context.p_num, context.q_num, context.t_num, context.dnum, context.gamma);

    int N = context.N;
    int slots = context.slots;
    int L = context.L;
    int K = context.K;

    int mlwe_rank = N / PCMM_N1;
    // ring packing always works on level0 ???
    vector<uint64_tt> p_ringpack = {context.pVec[context.p_num - 1]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();

    SecretKey sk(context);

    SchemeAlgo scheme_algo(context, scheme, sk);
    int s2c_level_cost = 3, c2s_level_cost = 3;
    EncodingMatrix encodingMatrix(sk, scheme, 4, 4, 1);
    Bootstrapper bootstrapper(context, scheme, scheme_algo, sk, encodingMatrix, 1);
        bootstrapper.addBootstrappingKey(sk);

    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme, bootstrapper);

    int pq_ringpack_count = pcmm_context.pq_ringpack.size();

    MLWESecretKey mlwe_sk(PCMM_N1, pq_ringpack_count, mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);
    scheme.addEncKey(sk);
    pcmm_scheme.addRepakcingKey(mlwe_sk, sk);
    
    cuDoubleComplex* message_host = new cuDoubleComplex[slots];
    randomComplexArray(message_host, slots, -1./10, 1./10);
    for(int i = 0; i < 8; i++){
        printf("%lf, ", message_host[i]);
    }
    cout<<endl;

    cuDoubleComplex* message_device, *dec_message;
    cudaMalloc(&message_device, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&dec_message, sizeof(cuDoubleComplex) * slots);
	cudaMemcpy(message_device, message_host, sizeof(double) * slots, cudaMemcpyHostToDevice);
    print_device_array(message_device, slots, "message");
    
    CUDATimer cuTimer;
        
    float gen_swk = 1000;
    float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
    float conj_time = 1000;
    float rlwe2mlwe = 1000, ppmm_time = 1000, mlwe2rlwe = 1000;
    float temp = 0;
    int target_level = 1 + s2c_level_cost;

    int mat_M = 256, mat_N = 128;
    float* plain_mat_host = new float[mat_M * mat_N];
    randomFloatArray(plain_mat_host, mat_M * mat_N, 1.0);
    // for(int i = 0; i < mat_M * mat_N; i++){
    //     plain_mat_host[i] = (float)(i % 256) * pow(-1, i) / 10000;
    // }
    float* plain_mat_device;
    cudaMalloc(&plain_mat_device, sizeof(float) * mat_M * mat_N);
    cudaMemcpy(plain_mat_device, plain_mat_host, sizeof(float) * mat_M * mat_N, cudaMemcpyHostToDevice);

    // for(target_level; target_level >= 0; target_level--)
    {
        int decomp_num = mlwe_rank / 2;

        vector<MLWECiphertext*> mlwe_cipher_decomposed;
        for(int i = 0; i < decomp_num; i++){
            MLWECiphertext* mlwe_cipher = new MLWECiphertext(PCMM_N1, q_ringpack_count - 1, q_ringpack_count - 1, mlwe_rank, NTL::RR(context.precision));
            mlwe_cipher_decomposed.push_back(mlwe_cipher);
        }

        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m2(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext m1_dec(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext m2_dec(N, L, target_level, slots, NTL::RR(context.precision));
        MLWEPlaintext mlwe_dec(PCMM_N1, q_ringpack_count - 1, q_ringpack_count - 1, NTL::RR(context.precision));
        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));

        double* real_msg_dec;
        cudaMalloc(&real_msg_dec, sizeof(double) * N);

        double* mlwe_msg_dec;
        cudaMalloc(&mlwe_msg_dec, sizeof(double) * PCMM_N1);

        cuTimer.start();
            // context.encode_coeffs(real_msg1, plain_m1_coeffs_encode);
            context.encode(message_device, plain_m1);
        temp = cuTimer.stop();
        ecd = min(ecd, temp);

        for(int i = 0; i < 1; i++)
        {
            cuTimer.start();
                scheme.encryptMsg(c1, plain_m1);
            temp = cuTimer.stop();
            enc = min(ecd, temp);


            cuTimer.start();
                c2 = c1;
                scheme.mulByiAndEqual(c2);
                scheme.addConstAndEqual(c1, 0.1);
                scheme.addAndEqual(c2, c1);
            temp = cuTimer.stop();
            conj_time = min(conj_time, temp);
            
            scheme.decrypt_display(sk, c2, "before s2c");

            pcmm_scheme.PCMM_Boot(plain_mat_device, c2, mlwe_cipher_decomposed, mat_M, mat_N, PCMM_N1);


            scheme.decrypt_display(sk, *bootstrapper.ctReal, "dec real");

            scheme.decrypt_display(sk, *bootstrapper.ctImag, "dec imag");

            // scheme.decryptMsg(m2_dec, sk, c2);
            // // // context.decode(m2_dec, dec_message);
            // context.decode_coeffs(m2_dec, real_msg_dec);
            // print_device_array(real_msg_dec, slots, "repacking decrypt");
        }


        printf("Time: encode: %f us decode: %f us\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f us dec: %f us\n", enc*1000, dec*1000);       
        printf("gen swk: %f us\n", gen_swk);
        printf("Time: rlwe2mlwe: %f us, ppmm: %f us, mlwe2rlwe: %f us\n", rlwe2mlwe*1000, ppmm_time*1000, mlwe2rlwe*1000);

        printf("Time: conj: %f us\n\n", conj_time*1000);


        getchar();
    }



    return 0;
}