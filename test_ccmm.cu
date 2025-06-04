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
#include "src/ckks/include/bootstrapping/Bootstrapper.cuh"
#include "src/ckks/include/TimeUtils.cuh"
#include "src/ckks/include/precision.cuh"
#include "src/ckks/ccmm/CCMM_Scheme.cuh"

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
    EncodingMatrix encodingMatrix(sk, scheme, 3, 3, is_STC_first);
    Bootstrapper bootHelper(context, scheme, scheme_algo, sk, encodingMatrix, is_STC_first);
    //     bootHelper.addBootstrappingKey(sk);
    CCMM_Scheme ccmm_scheme(context, scheme, scheme_algo, 128, 32, 256);
        ccmm_scheme.addKey(sk);

    cuDoubleComplex* mes1, *mes2;
	mes1 = new cuDoubleComplex[slots];
    mes2 = new cuDoubleComplex[slots];
	cuDoubleComplex* mes3 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, 1.0);
    randomComplexArray(mes2, slots, 1.0);
    // randomComplexArray(mes3, slots, 1.0);

    for(int i = 0; i < slots; i++)
    {
        mes3[i] = mes1[i + 2048];
    }

    cuDoubleComplex* complex_msg1, *complex_msg2, *complex_msg3;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg2, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg3, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg2, mes2, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg3, mes3, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

    
    print_device_array(complex_msg1, 8, "message1");
    // print_device_array(complex_msg2, slots, "message2");
    // print_device_array(complex_msg3, slots, "message3");

    int target_level = is_STC_first ? bootHelper.encodingMatrix.levelBudgetDec : 0;
    target_level = L - 3 - 9;
    // for(target_level; target_level >= 0; target_level--)
    {
        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m2(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c3(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m1_dec(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m2_dec(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m3_dec(N, L, L, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        for(int i = 0; i < 10; i++)
        {
            cudaEventRecord(start);
                context.encode(complex_msg1, plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            ecd = min(ecd, temp);
                context.encode(complex_msg2, plain_m2);

            cudaEventRecord(start);
                scheme.encryptMsg(c1, plain_m1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            enc = min(enc, temp);
                scheme.encryptMsg(c2, plain_m2);

                // scheme.leftRotate_23(c2, c1, 2);
                // scheme.leftRotateAndEqual_23(c1, 2);
            cudaEventRecord(start);
                ccmm_scheme.CCMM_QK(c1, c2, c3);
                // scheme.leftRotateAddSelf_23(c1, 2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            ccmm_time = min(ccmm_time, temp);

            cudaEventRecord(start);
                // scheme.rescaleAndEqual(c1);
                // scheme.rescaleAndEqual(c2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            resc = min(resc, temp);



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

        // scheme.decrypt_display(sk, c1, "c1");
        // scheme.decrypt_display(sk, c2, "c2");
        scheme.decrypt_display(sk, c3, "c3", 128);

        // vector<cuDoubleComplex> values_computed(slots);
        // cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        // vector<cuDoubleComplex> values_want(slots);
        // cudaMemcpy(values_want.data(), complex_msg1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        // auto status = GetPrecisionStats(values_computed, values_want);
        // cout<<status.String();
            // context.decode(m2_dec, complex_msg_dec);
        // print_device_array(complex_msg_dec, slots, "message_dec2");
            
            // context.decode(m3_dec, complex_msg_dec);
        // print_device_array(complex_msg_dec, slots, "message_dec3");

        printf("level: %d\n", target_level);

        printf("Time: encode: %f ms decode: %f ms\n", ecd*1000, dcd*1000);
        printf("Time: enc: %f ms dec: %f ms\n", enc*1000, dec*1000);
        printf("Time: hadd: %f ms cadd: %f ms\n", hadd*1000, cadd*1000);
        printf("Time: hmult: %f ms cmult: %f ms\n", hmult*1000, cmult*1000);
        printf("Time: rotate: %f ms conjugate: %f ms\n", rotate*1000, conjugate*1000);
        printf("Time: rescale: %f ms\n", resc*1000);
        printf("Time: bootstrapping: %f ms\n", bootstrapping);
        printf("Time: ccmm: %f ms\n", ccmm_time);
        printf("gen swk: %f ms\n", gen_swk);
        printf("Time: ntt: %f ms intt: %f ms\n\n", ntt*1000, intt*1000);


        getchar();
    }


    return 0;
}