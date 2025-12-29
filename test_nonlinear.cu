#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

#include "src/ckks/include/Ciphertext.cuh"
#include "src/ckks/include/Context_23.cuh"
#include "src/ckks/include/Plaintext.cuh"
#include "src/ckks/include/Scheme_23.cuh"
#include "src/ckks/include/TimeUtils.cuh"
#include "src/ckks/include/precision.cuh"
#include "src/ckks/attention/Attention.cuh"
void load_softmax_input(const std::string& filename, vector<cuDoubleComplex*>& mes, int token_len, int d);

int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    CUDATimer cuTimer;

    Context_23 context(logN, logslots, 64);
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
        scheme_algo.malloc_bsgs_buffer(context.eval_sine_chebyshev_coeff.size());
    Attention attention_scheme(context, scheme, scheme_algo, 128, 12, 64,sk);
        attention_scheme.addKey(sk);


    cuDoubleComplex* mes1, *mes2, *mes3;
	mes1 = new cuDoubleComplex[slots];
	mes2 = new cuDoubleComplex[slots];
    mes3 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, -10, 10);
    randomComplexArray(mes2, slots, -0.008, 0.009);
    randomComplexArray(mes3, slots, -0.008, 0.009);

    cuDoubleComplex* complex_msg1, *complex_msg2, *complex_msg3;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg2, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg3, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg2, mes2, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    cudaMemcpy(complex_msg3, mes3, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

    
    print_device_array(complex_msg1, 8, "message1");
    print_device_array(complex_msg2, 8, "message2");
    // print_device_array(complex_msg3, 8, "message3");
    vector<cuDoubleComplex*> message_host;
    vector<cuDoubleComplex*> message_device;
    load_softmax_input("softmaxInput_04_103847_281171.bin", message_host, 128, 768);
    for (int i = 0; i < 2; i++){
        cuDoubleComplex* mes_device;
        cudaMalloc(&mes_device, sizeof(cuDoubleComplex) * slots);
        cudaMemcpy(mes_device, message_host[i], sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        message_device.push_back(mes_device);
    }

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
                context.encode(complex_msg2, plain_m2);
                context.encode(complex_msg3, plain_m3);
                context.encode(message_device[0], plain_m1);
                context.encode(message_device[1], plain_m2);

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
                // attention_scheme.evalExp(c1);
                // attention_scheme.evalExp_iter(c1, 1);
                // attention_scheme.evalInv(c1, 10);
                // attention_scheme.evalSqrtInv(c1, sk, 10000);
                // attention_scheme.evalSoftMax(cipher_P);
                // attention_scheme.evalSoftMax_phase1(cipher_P);
                attention_scheme.evalSoftMax_phase1_iter(cipher_P,1);
                // cout<<"c1.level after evalSoftMax_phase1: "<<c1.l <<"  scale: "<<c1.scale<<endl;
                attention_scheme.evalSoftMax_phase2(cipher_P, sk);
                // attention_scheme.FASHE_evalSoftMax(cipher_P, sk);

                // attention_scheme.evalGeLU(c1);
                // attention_scheme.evalSiLU(c1);

                // vector<Ciphertext*> rlwe_cipher;
                // rlwe_cipher.push_back(&c1);
                // rlwe_cipher.push_back(&c2);
                // rlwe_cipher.push_back(&c3);
                // attention_scheme.LayerNorm(rlwe_cipher, sk);
            temp = cuTimer.stop();
            cout<<"c1.level after nonlinear: "<<c1.l <<"  scale: "<<c1.scale<<endl;

            nonlinear = min(nonlinear, temp);


            // cudaMemPrefetchAsync(c1.cipher_device, sizeof(uint64_tt) * N * 2 * (L+1), -1);        
            // cuTimer.start();
            //     cudaMemPrefetchAsync(c1.cipher_device, sizeof(uint64_tt) * N * 2 * (L+1), 0);
            // cout<<"perfetch time: "<<cuTimer.stop()<<endl;

            // scheme.decrypt_display(sk, c3);

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
        // print_device_array(complex_msg_dec, 8, "message_dec1");
        scheme.decrypt_display(sk, c1, "approx softmax");
        scheme.decrypt_display(sk, c2, "approx softmax");

        // vector<cuDoubleComplex> values_computed(slots);
        vector<cuDoubleComplex> values_computed(slots*2);
        // vector<cuDoubleComplex> values_computed(slots*3);

        scheme.decryptMsg(m1_dec, sk, c1);
        context.decode(m1_dec, complex_msg_dec);
        cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        scheme.decryptMsg(m1_dec, sk, c2);
        context.decode(m1_dec, complex_msg_dec);
        cudaMemcpy(values_computed.data() + slots, complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        // scheme.decryptMsg(m1_dec, sk, c3);
        // context.decode(m1_dec, complex_msg_dec);
        // cudaMemcpy(values_computed.data() + 2*slots, complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        // vector<cuDoubleComplex> values_want(slots);
        vector<cuDoubleComplex> values_want(slots*2);
        // vector<cuDoubleComplex> values_want(slots*3);
        // cudaMemcpy(values_want.data(), complex_msg1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        // cudaMemcpy(values_want.data() + slots, complex_msg2, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        // cudaMemcpy(values_want.data() + 2*slots, complex_msg3, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        cudaMemcpy(values_want.data(), message_device[0], sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        cudaMemcpy(values_want.data() + slots, message_device[1], sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
        cout<<"target: ";
        // verify nonlinear
        for(int i = 0; i < 2*slots; i++){
            double x = values_want[i].x;
            // values_want[i].x = 1/x;
            // values_want[i].x = 1/pow(x, 0.5);
            // values_want[i].x = exp(((x-10)));
            // values_want[i].x = x * normcdf(x);
            // values_want[i].x = x * (1 / (1+exp(-x)));
            // if(i < 8) printf("%.8lf, ", values_want[i].x);
        }

        // verify softmax
        int token_len = attention_scheme.token_len;
        int d = attention_scheme.d;
        double max_val = -1e20;
        // 256 columns
        for(int row_id = 0; row_id < token_len; row_id++){
            int row_len = slots / token_len;
            for(int block_id = 0; block_id < row_len / d; block_id++){
                double sum = 0;
                for(int i = 0; i < token_len; i++){
                    int offset = (i/d)*slots + (i%d);
                    double exp_x = exp(values_want[row_id * row_len + block_id * d + offset].x - attention_scheme.softmax_x_max);
                    values_want[row_id * row_len + block_id * d + offset].x = exp_x;
                    sum += exp_x;
                }

                for(int i = 0; i < token_len; i++){
                    int offset = + (i%d) + (i/d)*slots;
                    values_want[row_id * row_len + block_id * d + offset].x /= sum;
                    // max_val = max();
                }

                if(row_id == 0 && block_id == 0)
                {
                    for(int i = 0; i < 8; i++)
                        printf("%.8lf, ", values_want[row_id * row_len + block_id * d + i].x);
                }
            }
        }

        // // verify LayerNorm
        // int token_len = attention_scheme.token_len;
        // int d = attention_scheme.d;
        // int column_num = slots / token_len;
        // for (int i = 0; i < token_len; i++){
        //     double mu = 0;
        //     double sigma = 0;
        //     for (int j = 0; j<column_num;j++){
        //         mu += values_want[slots*0+column_num*i+j].x;
        //         mu += values_want[slots*1+column_num*i+j].x;
        //         mu += values_want[slots*2+column_num*i+j].x;
        //     }
        //     mu /= 768;
        //     for (int j = 0; j<column_num;j++){
        //         sigma += (values_want[slots*0+column_num*i+j].x-mu)*(values_want[slots*0+column_num*i+j].x-mu);
        //         sigma += (values_want[slots*1+column_num*i+j].x-mu)*(values_want[slots*1+column_num*i+j].x-mu);
        //         sigma += (values_want[slots*2+column_num*i+j].x-mu)*(values_want[slots*2+column_num*i+j].x-mu);
        //     }
        //     sigma = sqrt(sigma);
        //     for (int j = 0; j<column_num;j++){
        //         values_want[slots*0+column_num*i+j].x = (values_want[slots*0+column_num*i+j].x-mu)/sigma;
        //         values_want[slots*1+column_num*i+j].x = (values_want[slots*1+column_num*i+j].x-mu)/sigma;
        //         values_want[slots*2+column_num*i+j].x = (values_want[slots*2+column_num*i+j].x-mu)/sigma;
        //     }
        // }

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
void load_softmax_input(const std::string& filename, vector<cuDoubleComplex*>& mes, int token_len, int d) {
    int cipher_num = 2;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
    }
    for (int i = 0; i < cipher_num; i++){
        cuDoubleComplex* data = (cuDoubleComplex*)malloc(32768 * sizeof(cuDoubleComplex));
        mes.push_back(data);
    }


    std::vector<float> floatData(32768 * cipher_num);
    file.read(reinterpret_cast<char*>(floatData.data()), 32768 * cipher_num);
    file.close();
    double max_data = 0;
    for (int i = 0; i < token_len; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 4; k++){
                mes[0][i*256+k*64+j].x = static_cast<double>(floatData[i*128 + j + k*128*128]);
                mes[0][i*256+k*64+j].y = 0;
                mes[1][i*256+k*64+j].x = static_cast<double>(floatData[i*128 + j + k*128*128 + 64]);
                mes[1][i*256+k*64+j].y = 0;
                max_data = max(abs(mes[0][i*256+k*64+j].x), max_data);
                max_data = max(abs(mes[1][i*256+k*64+j].x), max_data);
            }
        }
    }
    cout << "max_data = " << max_data << endl;
}