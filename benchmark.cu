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

void print_time(char* operation, double time)
{
    printf(
"\t\"%s\":\{\n\
\t\"time\": %lf\n\
\t\},\n", operation, time
    );
}

int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    Context_23 context(logN, logslots);
    // cout<<"Generate Context OK"<<endl;
    // printf("logN: %d Pnum: %d Qnum: %d Tnum: %d dnum: %d gamma: %d\n", logN, context.p_num, context.q_num, context.t_num, context.dnum, context.gamma);

    int N = context.N;
    int L = context.L;
    int K = context.K;
    int slots = context.slots;

    Scheme_23 scheme(context);
    // cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    // cout<<"Generate sk OK"<<endl;

    scheme.mallocMemory();
    scheme.addEncKey(sk);
    // cout<<"Generate pk OK"<<endl;

float gen_swk = 1000;
float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
float hadd = 1000, cadd = 1000, hmult = 1000, cmult = 1000;
float rotate = 1000, conjugate = 1000;
float bootstrapping = 1000;
float temp = 0;

        cudaEventRecord(start);
            scheme.addMultKey_23(sk);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        gen_swk = min(gen_swk, temp);
            scheme.addLeftRotKey_23(sk, 1);
            scheme.addConjKey_23(sk);

    
    SchemeAlgo scheme_algo(context, scheme, sk);
    // Bootstrapper bootHelper(context, scheme, scheme_algo, sk);
        // bootHelper.addBootstrappingKey(sk);

    cuDoubleComplex* mes1, *mes2;
	mes1 = new cuDoubleComplex[slots];
    mes2 = new cuDoubleComplex[slots];
	cuDoubleComplex* mes3 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, 1.0);
    randomComplexArray(mes2, slots, 1.0);

    for(int i = 0; i < slots; i++)
    {
        mes3[i] = cuCmul(mes1[i], mes2[i]);
    }

    cuDoubleComplex* complex_msg1, *complex_msg2, *complex_msg3;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg2, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg3, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg2, mes2, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg3, mes3, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

    // print_device_array(complex_msg1, slots, "message1");
    // print_device_array(complex_msg2, slots, "message2");

    int target_level = L;
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
                scheme.encryptMsg(c3, plain_m2);

            cudaEventRecord(start);
                scheme.addAndEqual(c1, c2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            hadd = min(hadd, temp);

            cudaEventRecord(start);
                scheme.addConstAndEqual(c1, 0.2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            cadd = min(cadd, temp);


            cudaEventRecord(start);
                scheme.multAndEqual_23(c1, c2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            hmult = min(hmult, temp);

            cudaEventRecord(start);
                scheme.multConstAndEqual(c3, 0.8);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            cmult = min(cmult, temp);


            cudaEventRecord(start);
                scheme.leftRotateAndEqual_23(c2, 1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            rotate = min(rotate, temp);

            cudaEventRecord(start);
                scheme.conjugateAndEqual_23(c2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            conjugate = min(conjugate, temp);

            cudaEventRecord(start);
                scheme.rescaleAndEqual(c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            resc = min(resc, temp);
                scheme.rescaleAndEqual(c3);

            cudaEventRecord(start);
                // c3.l = 0;
                // bootHelper.bootstrapping(c3);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            bootstrapping = min(bootstrapping, temp);

            
            cudaEventRecord(start);
                scheme.decryptMsg(m1_dec, sk, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&dec, start, end);
            dec = min(dec, temp);
                scheme.decryptMsg(m2_dec, sk, c2);
                scheme.decryptMsg(m3_dec, sk, c3);
        }
        cudaEventRecord(start);
            context.decode(m1_dec, complex_msg_dec);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        dcd = min(dcd, temp);
        // print_device_array(complex_msg_dec, slots, "message_dec1");
            context.decode(m2_dec, complex_msg_dec);
        // print_device_array(complex_msg_dec, slots, "message_dec2");
        //     context.decode(m3_dec, complex_msg_dec);
        // print_device_array(complex_msg_dec, slots, "message_dec3");

        cout<<"{"<<endl;
        print_time("encode", ecd*1000);
        print_time("decode", dcd*1000);

        print_time("encrypt", enc*1000);
        print_time("decrypt", dec*1000);

        print_time("homomorphic_add", hadd*1000);
        print_time("const_add", cadd*1000);

        print_time("homomorphic_mult", hmult*1000);
        print_time("const_mult", cmult*1000);

        print_time("homomorphic_rotate", rotate*1000);
        print_time("homomorphic_conjugate", conjugate*1000);

        print_time("rescale", resc*1000);

        print_time("bootstrapping", bootstrapping*1000);

        cout<<"}"<<endl;

        // printf("Time: encode: %f ms decode: %f ms\n", ecd*1000, dcd*1000);
        // printf("Time: enc: %f ms dec: %f ms\n", enc*1000, dec*1000);
        // printf("Time: hadd: %f ms cadd: %f ms\n", hadd*1000, cadd*1000);
        // printf("Time: hmult: %f ms cmult: %f ms\n", hmult*1000, cmult*1000);
        // printf("Time: rotate: %f ms conjugate: %f ms\n", rotate*1000, conjugate*1000);
        // printf("Time: rescale: %f ms\n", resc*1000);
        // printf("Time: bootstrapping: %f ms\n", bootstrapping);
    }
    return 0;
}