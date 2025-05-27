#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Ciphertext.cuh"
#include "include/Context_23.cuh"
#include "include/Plaintext.cuh"
#include "include/Scheme_23.cuh"
#include "include/bootstrapping/Bootstrapper.cuh"
#include "include/TimeUtils.cuh"
#include "include/precision.cuh"
#include "include/pcmm/PCMM_Context.cuh"

int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    CUDATimer cuTimer;

    Context_23 context(logN, logslots, 192);
    cout<<"Generate Context OK"<<endl;


    int N = context.N;
    // int K = context.K;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = context.slots;

    int PCMM_N1 = 256;
    int mlwe_rank = N / PCMM_N1;
    vector<uint64_tt> p_ringpack = {context.pVec[0]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();

    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);

    Scheme_23 scheme(context);
    cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    cout<<"Generate sk OK"<<endl;

    scheme.addEncKey(sk);
    cout<<"Generate pk OK"<<endl;

    float gen_swk = 1000;
    float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
    float hadd = 1000, cadd = 1000, hmult = 1000, cmult = 1000;
    float rotate = 1000, conjugate = 1000;
    float bootstrapping = 1000;
    float temp = 0;


    cuDoubleComplex* mes1;
	mes1 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, 1.0);

    double* real_mes_host = new double[N];
    randomDoubleArray(real_mes_host, N, 1./10);
    for(int i = 0; i < 8; i++){
        printf("%lf, ", real_mes_host[i]);
    }
    cout<<endl;

    cuDoubleComplex* complex_msg1;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    
    double* real_msg1;
    cudaMalloc(&real_msg1, sizeof(double) * N);
	cudaMemcpy(real_msg1, real_mes_host, sizeof(double) * N, cudaMemcpyHostToDevice);
    


    print_device_array(complex_msg1, 8, "message1");
    print_device_array(real_msg1, N, "message2");

    int target_level = L;
    // for(target_level; target_level >= 0; target_level--)
    {
        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m1_coeffs_encode(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));

        Plaintext m1_dec(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m2_dec(N, L, L, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        double* real_msg_dec;
        cudaMalloc(&real_msg_dec, sizeof(double) * N);

        cudaEventRecord(start);
            context.encode(complex_msg1, plain_m1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        ecd = min(ecd, temp);
            context.encode_coeffs(real_msg1, plain_m1_coeffs_encode);

        for(int i = 0; i < 10; i++)
        {
            cudaEventRecord(start);
                scheme.encryptMsg(c1, plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            enc = min(enc, temp);
                scheme.encryptMsg(c2, plain_m1_coeffs_encode);



            cudaEventRecord(start);
                scheme.decryptMsg(m1_dec, sk, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&dec, start, end);
            dec = min(dec, temp);
                scheme.decryptMsg(m2_dec, sk, c2);
        }
        cudaEventRecord(start);
            context.decode(m1_dec, complex_msg_dec);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        dcd = min(dcd, temp);
        print_device_array(complex_msg_dec, 8, "message_dec1");

        vector<cuDoubleComplex> complex_values_computed(slots);
        cudaMemcpy(complex_values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        vector<cuDoubleComplex> complex_values_want(slots);
        cudaMemcpy(complex_values_want.data(), complex_msg1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        auto status = GetPrecisionStats(complex_values_computed, complex_values_want);
        cout<<status.String();


            context.decode_coeffs(m2_dec, real_msg_dec);
        print_device_array(real_msg_dec, N, "coeff message_dec2");

        
        vector<double> real_values_computed(N);
        cudaMemcpy(real_values_computed.data(), real_msg_dec, sizeof(double) * N, cudaMemcpyDeviceToHost);

        vector<double> real_values_want(N);
        cudaMemcpy(real_values_want.data(), real_msg1, sizeof(double) * N, cudaMemcpyDeviceToHost);

        auto status_real = GetPrecisionStats(real_values_computed, real_values_want);
        cout<<status_real.String();


        printf("level: %d\n", target_level);

        printf("Time: encode: %f ms decode: %f ms\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f ms dec: %f ms\n", enc*1000, dec*1000);
        printf("Time: hadd: %f ms cadd: %f ms\n", hadd*1000, cadd*1000);
        printf("Time: hmult: %f ms cmult: %f ms\n", hmult*1000, cmult*1000);
        printf("Time: rotate: %f ms conjugate: %f ms\n", rotate*1000, conjugate*1000);
        printf("Time: rescale: %f ms\n", resc*1000);
        printf("Time: bootstrapping: %f ms\n", bootstrapping);
        printf("gen swk: %f ms\n", gen_swk);
        printf("Time: ntt: %f ms intt: %f ms\n\n", ntt*1000, intt*1000);


        getchar();
    }


    return 0;
}