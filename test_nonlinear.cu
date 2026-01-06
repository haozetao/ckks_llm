#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "src/ckks/include/Ciphertext.cuh"
#include "src/ckks/include/Context_23.cuh"
#include "src/ckks/include/Plaintext.cuh"
#include "src/ckks/include/Scheme_23.cuh"
#include "src/ckks/include/TimeUtils.cuh"
#include "src/ckks/include/precision.cuh"
#include "src/ckks/attention/Attention.cuh"

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
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = context.slots;

    Scheme_23 scheme(context);
    cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    cout<<"Generate sk OK"<<endl;

    scheme.mallocMemory();
    scheme.addEncKey(sk);
    cout<<"Generate pk OK"<<endl;

float gen_swk = 1000;
float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
float hadd = 1000, cadd = 1000, hmult = 1000, cmult = 1000;
float rotate = 1000, conjugate = 1000;
float nonlinear = 1000;
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
        scheme_algo.malloc_bsgs_buffer(40);
    Attention attention_scheme(context, scheme, scheme_algo, 128, 12, 64, sk);
        attention_scheme.addKey(sk);


    cuDoubleComplex* mes1;
	mes1 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, -8, 8);

    cuDoubleComplex* complex_msg1;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    
    print_device_array(complex_msg1, 8, "message1");

    int target_level = L;
    // for(target_level; target_level >= 0; target_level--)
    {
        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m2(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m3(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c3(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m1_dec(N, L, L, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        vector<Ciphertext*> cipher_P = {&c1, &c2};

        for(int i = 0; i < 1; i++)
        {
            cudaEventRecord(start);
                context.encode(complex_msg1, plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            ecd = min(ecd, temp);

            cudaEventRecord(start);
                scheme.encryptMsg(c1, plain_m1);
                scheme.encryptMsg(c2, plain_m2);
                scheme.encryptMsg(c3, plain_m3);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            enc = min(enc, temp);


            cout<<"c1.level before nonlinear: "<<c1.l <<"  scale: "<<c1.scale<<endl;
            cuTimer.start();

                // attention_scheme.evalGeLU(c1);
                attention_scheme.evalSiLU(c1);

            temp = cuTimer.stop();
            cout<<"c1.level after nonlinear: "<<c1.l <<"  scale: "<<c1.scale<<endl;

            nonlinear = min(nonlinear, temp);

            cudaEventRecord(start);
                scheme.decryptMsg(m1_dec, sk, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&dec, start, end);
            dec = min(dec, temp);
        }
        cudaEventRecord(start);
            context.decode(m1_dec, complex_msg_dec);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        dcd = min(dcd, temp);
        scheme.decrypt_display(sk, c1, "approx softmax");

        vector<cuDoubleComplex> values_computed(slots);
        scheme.decryptMsg(m1_dec, sk, c1);
        context.decode(m1_dec, complex_msg_dec);
        cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        vector<cuDoubleComplex> values_want(slots);
        cudaMemcpy(values_want.data(), complex_msg1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        cout<<"target: ";
        // verify nonlinear
        for(int i = 0; i < slots; i++){
            double x = values_want[i].x;
            // values_want[i].x = x * normcdf(x);
            values_want[i].x = x * (1 / (1+exp(-x)));
            if(i < 8) printf("%.8lf, ", values_want[i].x);
        }

        auto status = GetPrecisionStats(values_computed, values_want);
        cout<<status.String();
        
        printf("level: %d\n", target_level);

        printf("Time: encode: %f ms decode: %f ms\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f ms dec: %f ms\n", enc*1000, dec*1000);
        printf("Time: hadd: %f ms cadd: %f ms\n", hadd*1000, cadd*1000);
        printf("Time: hmult: %f ms cmult: %f ms\n", hmult*1000, cmult*1000);
        printf("Time: rotate: %f ms conjugate: %f ms\n", rotate*1000, conjugate*1000);
        printf("Time: rescale: %f ms\n", resc*1000);
        printf("Time: nonlinear: %f ms\n", nonlinear);
        printf("gen swk: %f ms\n", gen_swk);
        printf("Time: ntt: %f ms intt: %f ms\n\n", ntt*1000, intt*1000);


        getchar();
    }


    return 0;
}