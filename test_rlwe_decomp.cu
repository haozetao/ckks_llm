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
    int PCMM_N1 = atoi(argv[2]);
    int logslots = logN - 1;

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
    EncodingMatrix encodingMatrix(sk, scheme, 4, 4, 1);
    Bootstrapper boostrapper(context, scheme, scheme_algo, sk, encodingMatrix, 1);
        // boostrapper.addBootstrappingKey(sk);


    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme, boostrapper);

    int pq_ringpack_count = pcmm_context.pq_ringpack.size();

    MLWESecretKey mlwe_sk(PCMM_N1, pq_ringpack_count, mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);
    scheme.addEncKey(sk);
    pcmm_scheme.addRepakcingKey(mlwe_sk, sk);
    
    double* real_mes_host = new double[N];
    randomDoubleArray(real_mes_host, N, 1./10);
    for(int i = 0; i < 8; i++){
        printf("%lf, ", real_mes_host[i]);
    }
    cout<<endl;

    double* real_msg1;
    cudaMalloc(&real_msg1, sizeof(double) * N);
	cudaMemcpy(real_msg1, real_mes_host, sizeof(double) * N, cudaMemcpyHostToDevice);
    
    CUDATimer cuTimer;
    
    print_device_array(real_msg1, N, "message2");

    
    float gen_swk = 1000;
    float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
    float rlwe2mlwe = 1000, ppmm_time = 1000, mlwe2rlwe = 1000;
    float temp = 0;
    int target_level = 1;

    int mat_M = mlwe_rank, mat_N = mlwe_rank;
    float* plain_mat_host = new float[mat_M * mat_N];
    for(int i = 0; i < mat_M * mat_N; i++){
        plain_mat_host[i] = (float)(i % PCMM_N1) / 10000;
    }
    float* plain_mat_device;
    cudaMalloc(&plain_mat_device, sizeof(float) * mat_M * mat_N);
    cudaMemcpy(plain_mat_device, plain_mat_host, sizeof(float) * mat_M * mat_N, cudaMemcpyHostToDevice);

    // for(target_level; target_level >= 0; target_level--)
    {
        int decomp_num = N / PCMM_N1;
        cout<<"decomp_num: "<<decomp_num<<endl;
        vector<MLWECiphertext*> mlwe_cipher_decomposed;
        for(int i = 0; i < decomp_num; i++){
            MLWECiphertext* mlwe_cipher = new MLWECiphertext(PCMM_N1, q_ringpack_count - 1, q_ringpack_count - 1, mlwe_rank, NTL::RR(context.precision));
            mlwe_cipher_decomposed.push_back(mlwe_cipher);
        }

        Plaintext plain_m1_coeffs_encode(N, L, target_level, slots, NTL::RR(context.precision));
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
            context.encode_coeffs(real_msg1, plain_m1_coeffs_encode);
        temp = cuTimer.stop();
        ecd = min(ecd, temp);

        for(int i = 0; i < 10; i++)
        {
            cuTimer.start();
                scheme.encryptMsg(c1, plain_m1_coeffs_encode);
            temp = cuTimer.stop();
            enc = min(ecd, temp);
        
            cuTimer.start();
                pcmm_scheme.rlweCipherDecompose(c1, mlwe_cipher_decomposed, decomp_num, 0);
            temp = cuTimer.stop();
            rlwe2mlwe = min(rlwe2mlwe, temp);

            // for(int i = 0; i < 4; i++){
            //     cuTimer.start();
            //         pcmm_scheme.mlweDecrypt(*mlwe_cipher_decomposed[i], mlwe_sk, mlwe_dec);
            //     temp = cuTimer.stop();
            //     rlwe2mlwe = min(rlwe2mlwe, temp);

            //     pcmm_context.decodeCoeffs(mlwe_dec, mlwe_msg_dec);
            //     print_device_array(mlwe_msg_dec, PCMM_N1, "MLWE message_dec");
            // }

            cuTimer.start();
                scheme.decryptMsg(m1_dec, sk, c1);
            temp = cuTimer.stop();
            dec = min(ecd, temp);

            cuTimer.start();
                context.decode_coeffs(m1_dec, real_msg_dec);
            temp = cuTimer.stop();
            dcd = min(dcd, temp);
            
            print_device_array(real_msg_dec, N, "coeff message_dec2");


            cuTimer.start();
                pcmm_scheme.PPMM(plain_mat_device, mlwe_cipher_decomposed, mat_M, mat_N, PCMM_N1);
            temp = cuTimer.stop();
            ppmm_time = min(ppmm_time, temp);

            cuTimer.start();
                pcmm_scheme.mlweCipherPacking(c2, mlwe_cipher_decomposed, decomp_num, 0);
            temp = cuTimer.stop();
            mlwe2rlwe = min(mlwe2rlwe, temp);


            scheme.decryptMsg(m2_dec, sk, c2);
            context.decode_coeffs(m2_dec, real_msg_dec);
            print_device_array(real_msg_dec, N, "repacking decrypt");
        }

        
        // vector<double> real_values_computed(N);
        // cudaMemcpy(real_values_computed.data(), real_msg_dec, sizeof(double) * N, cudaMemcpyDeviceToHost);

        // vector<double> real_values_want(N);
        // cudaMemcpy(real_values_want.data(), real_msg1, sizeof(double) * N, cudaMemcpyDeviceToHost);

        // auto status_real = GetPrecisionStats(real_values_computed, real_values_want);
        // cout<<status_real.String();

        
        // vector<double> mlwe_values_computed(PCMM_N1);
        // vector<double> mlwe_values_want(PCMM_N1);
        // cudaMemcpy(mlwe_values_computed.data(), mlwe_msg_dec, sizeof(double) * PCMM_N1, cudaMemcpyDeviceToHost);

        // vector<double> temp(N);
        // cudaMemcpy(temp.data(), real_msg_dec, sizeof(double) * N, cudaMemcpyDeviceToHost);
        // for(int i = 0; i < PCMM_N1; i++) {
        //     mlwe_values_want[i] = temp[i * mlwe_rank];
        //     cout<<"("<<mlwe_values_want[i]<<", ";
        //     cout<<mlwe_values_computed[i]<<")  ";
        // }

        // auto status_real = GetPrecisionStats(mlwe_values_computed, mlwe_values_want);
        // cout<<status_real.String();


        printf("Time: encode: %f us decode: %f us\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f us dec: %f us\n", enc*1000, dec*1000);       
        printf("gen swk: %f us\n", gen_swk);
        printf("Time: rlwe2mlwe: %f us, ppmm: %f us, mlwe2rlwe: %f us\n", rlwe2mlwe*1000, ppmm_time*1000, mlwe2rlwe*1000);

        printf("Time: ntt: %f us intt: %f us\n\n", ntt*1000, intt*1000);


        getchar();
    }



    return 0;
}