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
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;
    int PCMM_N1 = 128;

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
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme, bootstrapper, sk);

    int pq_ringpack_count = pcmm_context.pq_ringpack.size();

    MLWESecretKey mlwe_sk(PCMM_N1, pq_ringpack_count, mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);
    scheme.addEncKey(sk);
    pcmm_scheme.addRepakcingKey(mlwe_sk, sk);
    
    cuDoubleComplex* message_host = new cuDoubleComplex[slots];
    double* message_host_real = new double[N];
    randomComplexArray(message_host, slots, -1./10, 1./10);
    for(int i = 0; i < slots; i++){
        message_host_real[i] = message_host[i].x;
        message_host_real[i + slots] = message_host[i].y;
    }
    for(int i = 0; i < 8; i++){
        printf("%lf, ", message_host[i]);
    }
    cout<<endl;

    cuDoubleComplex* message_device, *dec_message;
    cudaMalloc(&message_device, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&dec_message, sizeof(cuDoubleComplex) * slots);
	cudaMemcpy(message_device, message_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    print_device_array(message_device, slots, "message");
    
    CUDATimer cuTimer;
        
    float gen_swk = 1000;
    float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
    float conj_time = 1000;
    float rlwe2mlwe = 1000, ppmm_time = 1000, mlwe2rlwe = 1000;
    float temp = 0;
    int target_level = 1 + s2c_level_cost;

    // int mat_M = mlwe_rank, mat_N = mlwe_rank;
    int mat_M = 256, mat_N = 256;
    float* plain_mat_host = new float[mat_M * mat_N];
    randomFloatArray(plain_mat_host, mat_M * mat_N, 0.1);
    // for(int i = 0; i < 512; i++){
    //     for(int j = 0; j < 512; j++){
    //         if (i >= 256 || j >= 256){
    //             plain_mat_host[i * mat_N + j] = 0;
    //         }
    //     }
    // }

    double* plain_gemm_host = new double[N];
    memset(plain_gemm_host, 0, sizeof(double) * N);
    
    printf("mat_M: %d, mat_N: %d, mat_K: %d\n", mat_M, mat_N, PCMM_N1);
    for(int idx_M = 0; idx_M < mat_M; idx_M++){
        for(int idx_K = 0; idx_K < PCMM_N1; idx_K++){
            for(int idx_N = 0; idx_N < mat_N; idx_N++){
                plain_gemm_host[idx_M + idx_K*mat_M] += plain_mat_host[idx_N + idx_M*mat_N] * message_host_real[idx_N + idx_K*mat_N];
            }
        }
    }

    float* plain_mat_device;
    cudaMalloc(&plain_mat_device, sizeof(float) * mat_M * mat_N);
    cudaMemcpy(plain_mat_device, plain_mat_host, sizeof(float) * mat_M * mat_N, cudaMemcpyHostToDevice);
    scheme.addLeftRotKey_23(sk, int(slots/2));
    // for(target_level; target_level >= 0; target_level--)
    {
        int decomp_num = mlwe_rank;

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
                // scheme.leftRotateAndEqual_23(c1, int(slots/2));
                // scheme.mulByiAndEqual(c1);
                // scheme.addAndEqual(c2, c1);
                // scheme.mulByiAndEqual(c2);
                // scheme.addConstAndEqual(c1, 0.1);
                // scheme.addAndEqual(c2, c1);
            temp = cuTimer.stop();
            conj_time = min(conj_time, temp);
            scheme.decrypt_display(sk, c2, "enc c2");
            
            cout << "before ppmm boot scale: " << c2.scale << endl;
            pcmm_scheme.test_PCMM_Boot(plain_mat_device, c2, mlwe_cipher_decomposed, mat_M, mat_N, PCMM_N1);
            cout << "after ppmm boot scale: " << c2.scale << endl;
            // encodingMatrix.EvalSlotsToCoeffs(encodingMatrix.m_U0PreFFT, c2);

            // // scheme.decrypt_display(sk, c2, "dec c2");

            // scheme.decrypt_display(sk, *bootstrapper.ctReal, "dec real");

            // scheme.decrypt_display(sk, *bootstrapper.ctImag, "dec imag");


            // scheme.conjugateAndEqual_23(*bootstrapper.ctImag);
            scheme.mulByiAndEqual(*bootstrapper.ctImag);
            c1 = *bootstrapper.ctReal;
            scheme.addAndEqual(c1, *bootstrapper.ctImag);
            
            // c1 = c2;

            scheme.decryptMsg(m1_dec, sk, c1);
            context.decode(m1_dec, dec_message);
            print_device_array(dec_message, slots, "dec_message");
            // context.decode_coeffs(m1_dec, real_msg_dec, true);
            // print_device_array(real_msg_dec        , slots, "repacking decrypt1");
            // print_device_array(real_msg_dec + slots, slots, "repacking decrypt2");
        }
        printf("=========target ppmm output==========\n");
        for(int i = 0; i < 8; i++){
            // printf("%lf, ", plain_gemm_host[i]);
            printf("%lf, ", message_host_real[i]);
        }
        cout<<endl;
        for(int i = 0; i < 8; i++){
            // printf("%lf, ", plain_gemm_host[i + slots]);
            printf("%lf, ", message_host_real[i + int(slots/2)]);
        }
        printf("\n=====================================\n");

        vector<cuDoubleComplex> real_values_computed(slots);
        cudaMemcpy(real_values_computed.data(), dec_message, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        vector<cuDoubleComplex> real_values_want(slots);
        // memcpy(real_values_want.data(), plain_gemm_host, sizeof(double) * N);
        for(int i = 0; i < slots; i++){
            real_values_want[i].x = plain_gemm_host[i];
            real_values_want[i].y = plain_gemm_host[i + slots];
            // real_values_want[i].x = message_host_real[i];
            // real_values_want[i].y = message_host_real[i + slots];
        }

        printf("=========computed ppmm output==========\n");
        for(int i = 0; i < 8; i++){
            // printf("%lf, ", plain_gemm_host[i]);
            printf("%lf, ", real_values_computed[i].x);
        }
        cout<<endl;
        for(int i = 0; i < 8; i++){
            // printf("%lf, ", plain_gemm_host[i + slots]);
            printf("%lf, ", real_values_computed[i].y);
        }
        cout<<endl;
        printf("\n=====================================\n");
        for(int i = 0; i < slots; i++){
            // if (abs(real_values_want[i].x-real_values_computed[i].x) > 3e-2 || abs(real_values_want[i].y-real_values_computed[i].y) > 3e-2){
            if (abs(real_values_want[i].x-real_values_computed[i].x) > 1e-3 || abs(real_values_want[i].y-real_values_computed[i].y) > 1e-3){
                printf("diff at %d: ", i);
                printf("want: %lf + %lfi, ", real_values_want[i].x, real_values_want[i].y);
                printf("got : %lf + %lfi\n", real_values_computed[i].x, real_values_computed[i].y);
            }
            else{
                // printf("right at %d: ", i);
                // printf("want: %lf + %lfi, ", real_values_want[i].x, real_values_want[i].y);
                // printf("got : %lf + %lfi\n", real_values_computed[i].x, real_values_computed[i].y);
            }
        }

        cout<<endl;
        auto status_real = GetPrecisionStats(real_values_computed, real_values_want);
        cout<<status_real.String();


        printf("Time: encode: %f us decode: %f us\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f us dec: %f us\n", enc*1000, dec*1000);       
        printf("gen swk: %f us\n", gen_swk);
        printf("Time: rlwe2mlwe: %f us, ppmm: %f us, mlwe2rlwe: %f us\n", rlwe2mlwe*1000, ppmm_time*1000, mlwe2rlwe*1000);

        printf("Time: conj: %f us\n\n", conj_time*1000);


        getchar();
    }



    return 0;
}