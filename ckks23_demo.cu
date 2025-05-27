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

int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    Context_23 context(logN, logslots);
    cout<<"Generate Context OK"<<endl;

    // for(auto v : context.pqtVec)
    // {
    //     printf("%llu, ", v);
    // }

    int N = context.N;
    // int K = context.K;
    int L = context.L;
    int K = context.K;
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
    
    SchemeAlgo scheme_algo(context, scheme, sk);
    EncodingMatrix encodingMatrix(sk, scheme);
    Bootstrapper bootHelper(context, scheme, scheme_algo, sk, encodingMatrix);
        // bootHelper.addBootstrappingKey(sk);

    cuDoubleComplex* mes1, *mes2;
	mes1 = new cuDoubleComplex[slots];
    mes2 = new cuDoubleComplex[slots];
	cuDoubleComplex* mes3 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, 1.0);
    randomComplexArray(mes2, slots, 1.0);
    // randomComplexArray(mes3, slots, 1.0);

    for(int i = 0; i < slots; i++)
    {
        mes3[i] = cuCmul(mes1[i], mes2[i]);
        // mes3[i] = make_cuDoubleComplex(mes1[i].x * 0.5, mes1[i].y * 0.5);
        // mes3[i] = mes1[(i+1 + slots) % slots];
        // mes3[i] = make_cuDoubleComplex(mes1[i].x + 0.2, mes1[i].y + 0.1);
        // cuDoubleComplex m1m2 = cuCmul(mes1[i], mes2[i]);
        // mes3[i] = make_cuDoubleComplex(1/(2*M_PI) * sin(20*M_PI * m1m2.x), 1/(2*M_PI) * sin(20*M_PI * m1m2.y));
    }

    cuDoubleComplex* complex_msg1, *complex_msg2, *complex_msg3;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg2, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg3, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg2, mes2, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg3, mes3, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

    print_device_array(complex_msg1, 8, "message1");
    print_device_array(complex_msg2, 8, "message2");
    // print_device_array(complex_msg3, slots, "message3");

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

            cudaEventRecord(start);
                // Ciphertext c3 = scheme.mult_23(c1, c2);
                scheme.multAndEqual_23(c1, c2);
                // scheme.multConstAndEqual(c1, 0.5);
                // scheme.addConstAndEqual(c1, plain_m2);
                // scheme.squareAndEqual(c1);
                // scheme.negateAndEqual(c1);
                // scheme.multConstAndAddCipherEqual(c1, c2, 0.1, NTL::RR(context.qVec[c1.l]));
                // scheme.multAndEqual_23(c1, c2);
                // scheme.multAndEqual_23(c1, c2);

                // scheme.squareAndEqual(c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            hmult = min(hmult, temp);


            cudaEventRecord(start);
                scheme.leftRotateAndEqual_23(c2, 1);
                // scheme.addAndEqual(c1, c2);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            hadd = min(hadd, temp);

            // cudaEventRecord(start);
                scheme.conjugateAndEqual_23(c2);
            //     // scheme.conjugate_23(c2, c3);
            // cudaEventRecord(end);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&temp, start, end);
            // conjugate = min(conjugate, temp);

            cudaEventRecord(start);
                // scheme.addAndEqual(c1, c1);
                // scheme.addConstAndEqual(c1, -0.5);
                // scheme.addConstAndEqual(c1, make_cuDoubleComplex(0.2, 0.1));
                scheme.rescaleAndEqual(c1);
                // scheme.rescaleAndEqual(c3);
                // scheme.rescaleToAndEqual(c1, target_level - 3);
                // scheme.multConstAndEqual(c2, 0.2);
                // scheme.rescaleAndEqual(c2);
                // scheme.addAndEqual(c2, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            resc = min(resc, temp);
            c3 = c1;

            // scheme.decrypt_display(sk, c3);

            cudaEventRecord(start);
                // bootHelper.bootstrapping(c3);
                // bootHelper.EvalModAndEqual(c3);
                // scheme_algo.evalPolynomialChebyshev(c3, NTL::RR(context.precision));
                // scheme.addConstAndEqual(c3, -0.025);
                // scheme.addConstAndEqual(c1, make_cuDoubleComplex(0.2, 0.1));
                // scheme.rescaleAndEqual(c3);
                // scheme.rescaleToAndEqual(c1, target_level - 3);
                // scheme.multConstAndEqual(c2, 0.2);
                // scheme.rescaleAndEqual(c2);
                // scheme.addAndEqual(c2, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            bootstrapping = min(bootstrapping, temp);
                // cout<<"c3.level: "<<c3.l<<endl;

                // scheme.divByiAndEqual(c2);
                // scheme.mulByiAndEqual(c2);

            cudaEventRecord(start);
                scheme.decryptMsg(m1_dec, sk, c1);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&dec, start, end);
            dec = min(dec, temp);
                scheme.decryptMsg(m2_dec, sk, c2);
                scheme.decryptMsg(m3_dec, sk, c3);


            // cudaEventRecord(start);
            //     context.ToNTTInplace(c1.cipher_device, 0, 0, 1, L+1, L+1);
            //     // context.inverseNTT_batch(c1.ax_device, 0, K, 1, L+1);
            // cudaEventRecord(end);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&temp, start, end);
            // ntt = min(ntt, temp);

            // cudaEventRecord(start);
            //     context.FromNTTInplace(c1.cipher_device, 0, 0, 1, L+1, L+1);
            //     // context.forwardNTT_batch(c1.ax_device, 0, K, 1, L+1);
            // cudaEventRecord(end);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&temp, start, end);
            // intt = min(intt, temp);


            // cudaEventRecord(start);
            //     context.decode(m1_dec, complex_msg_dec);
            // cudaEventRecord(end);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&temp, start, end);
            // dcd = min(dcd, temp);

            // int error_num = compare_device_array(complex_msg_dec, complex_msg3, slots, "mult");
            // if(error_num != 0)
            // {
            //     print_device_array(complex_msg_dec, slots, "message_dec1");
            //     cout<<"error_idx: "<< i <<" error_num: "<<error_num<<"  level: "<< c1.l <<endl;
            // }
                // print_device_array(context.decode_buffer, N, L+1, "decode_buffer");
        }
        cudaEventRecord(start);
            context.decode(m1_dec, complex_msg_dec);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        dcd = min(dcd, temp);
        print_device_array(complex_msg_dec, 8, "message_dec1");

            context.decode(m2_dec, complex_msg_dec);
        print_device_array(complex_msg_dec, 8, "message_dec2");
            context.decode(m3_dec, complex_msg_dec);
        print_device_array(complex_msg_dec, 8, "message_dec3");


        // c3 = c2;
        // for(int i = 0; i < 10; i++)
        // {
        //     scheme.leftRotateAndEqual_23(c3, 1);

        //     scheme.decryptMsg(m3_dec, sk, c3);
        //     cout<< "c3.l: " << c3.l << "   c3.scale: " << c3.scale <<endl;
        //     context.decode(m3_dec, complex_msg_dec);
        //     print_device_array(complex_msg_dec, slots, "message_dec3");

        //     // c1 = c3;
        // }

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

    }


    return 0;
}