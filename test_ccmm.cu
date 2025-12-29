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


void read_matrix_from_file(const std::string& filename, cuDoubleComplex* mes, int size);
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
    // Attention attention_scheme(context, scheme, scheme_algo, 128, 12, 64);
    Attention attention_scheme(context, scheme, scheme_algo, 128, 12, 64, sk);
    attention_scheme.addKey(sk);

    cuDoubleComplex* mes1, *mes2;
	mes1 = new cuDoubleComplex[slots];
    mes2 = new cuDoubleComplex[slots];
	cuDoubleComplex* mes3 = new cuDoubleComplex[slots];

    randomComplexArray(mes1, slots, -0.1, 0.1);
    randomComplexArray(mes2, slots, -0.1, 0.1);
    randomComplexArray(mes3, slots, 0.1);

    // splited ccmm param
    // vector<Ciphertext *>cQ(4);
    // vector<Ciphertext *>cK(4);
    // vector<Ciphertext *>cO(8);

    // vector<cuDoubleComplex*> msg_q(4);
    // vector<cuDoubleComplex*> msg_k(4);
    // vector<cuDoubleComplex*> complex_msgq(4);
    // vector<cuDoubleComplex*> complex_msgk(4);
    // vector<Plaintext> plain_q(4);
    // vector<Plaintext> plain_k(4);
    // for (int i=0; i<4; i++){
    //     msg_q[i] = new cuDoubleComplex[slots];
    //     msg_k[i] = new cuDoubleComplex[slots];
    //     randomComplexArray(msg_q[i], slots, -1, 1);
    //     randomComplexArray(msg_k[i], slots, -1, 1);
    //     cudaMalloc(&complex_msgq[i], sizeof(cuDoubleComplex) * slots);
    //     cudaMalloc(&complex_msgk[i], sizeof(cuDoubleComplex) * slots);

    //     cudaMemcpy(complex_msgq[i], msg_q[i], sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    //     cudaMemcpy(complex_msgk[i], msg_k[i], sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    //     plain_q[i] = Plaintext(N, L, L - 3 - 9, slots, NTL::RR(context.precision));
    //     plain_k[i] = Plaintext(N, L, L - 3 - 9, slots, NTL::RR(context.precision));
        
    //     context.encode(complex_msgq[i], plain_q[i]);
    //     context.encode(complex_msgk[i], plain_k[i]);

    //     cQ[i] = new Ciphertext(N, L, L, slots, NTL::RR(context.precision));
    //     cK[i] = new Ciphertext(N, L, L, slots, NTL::RR(context.precision));
    //     scheme.encryptMsg(*cQ[i], plain_q[i]);
    //     scheme.encryptMsg(*cK[i], plain_k[i]);

    //     cO[i*2] = new Ciphertext(N, L, L, slots, NTL::RR(context.precision));
    //     cO[i*2+1] = new Ciphertext(N, L, L, slots, NTL::RR(context.precision));
    // }
    
    
    /*************************************************************************** */

    for(int i = 0; i < slots; i++)
    {
        mes3[i] = mes1[i + 2048];
    }

    /****************Verify ccmm *V ******************************/
    // read_matrix_from_file("python/data/V.txt", mes1, slots);
    // read_matrix_from_file("python/data/sigma_A_1.txt", mes2, slots);
    // read_matrix_from_file("python/data/sigma_A_2.txt", mes3, slots);
    /****************end Verify ccmm *V ******************************/

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
        Plaintext plain_m3(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c3(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c4(N, L, L, slots, NTL::RR(context.precision));
        Plaintext m1_dec(N, L, L, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        for(int i = 0; i < 1; i++)
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
            // *V
            context.encode(complex_msg3, plain_m3);
            scheme.encryptMsg(c3, plain_m3);
            cudaEventRecord(start);
                cout << "before level " << c1.l << endl;
                attention_scheme.CCMM_QK(c1, c2, c3, c4);
                cout << "res level " << c3.l << endl;
                cout << "res level " << c4.l << endl;
                // cout << "res scale " << c3.scale << endl;
                // cout << "res scale " << c4.scale << endl;
                // attention_scheme.CCMM_QK_splited_heads(cQ,cK,cO,16);
                // scheme.leftRotateAddSelf_23(c1, 2);
                
                // attention_scheme.TauAndEqual(c1);
                // cout << "KS_SV = " << attention_scheme.KS_SV << endl;
                // attention_scheme.CCMM_V(c2, c3, c1, c4);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&temp, start, end);
            ccmm_time = min(ccmm_time, temp);
            cout << "KS_SV = " << attention_scheme.KS_SV << endl;

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

            /******************** Verify Q*K^T *********************************/
            cuDoubleComplex *target_res1, *target_res2;
            cuDoubleComplex *res1;
            Plaintext dec_m1(N, L, target_level, slots, NTL::RR(context.precision));
            target_res1 = new cuDoubleComplex[slots];
            target_res2 = new cuDoubleComplex[slots];
            res1 = new cuDoubleComplex[slots];
            memset(target_res1, 0, sizeof(cuDoubleComplex) * slots);
            memset(target_res2, 0, sizeof(cuDoubleComplex) * slots);


            for (int j=0; j<64; j++){
                for (int i = 0; i<128; i++){
                    for (int t = 0; t<4; t++){
                        for (int k = 0; k<64; k++){
                            target_res1[i*256 + t*64 +j] = cuCadd(cuCmul(mes1[((i+j)%128)*256+ t*64 + k] , mes2[((i+j)%128)*256+ t*64 + k]), target_res1[i*256 + t*64 +j]);
                            target_res2[i*256 + t*64 +j] = cuCadd(cuCmul(mes1[((i+j+64)%128)*256+ t*64 + k] , mes2[((i+j+64)%128)*256+ t*64 + k]), target_res2[i*256 + t*64 +j]);
                        }
                    }
                }
            }
            

            scheme.decryptMsg(dec_m1, sk, c3);
            context.decode(dec_m1, complex_msg_dec);
            cudaMemcpy(res1, complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);            
            vector<cuDoubleComplex> values_computed(slots);
            cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
            vector<cuDoubleComplex> values_want(slots);
            for (int i=0; i<slots; i++){
                values_want[i].x = 0;
                values_want[i].y = 0;
            }
            for (int j=0; j<64; j++){
                for (int i = 0; i<128; i++){
                    for (int t = 0; t<4; t++){
                        for (int k = 0; k<64; k++){
                            values_want[i*256 + t*64 +j].x += mes1[i*256+ t*64 + k].x * mes2[((i+j)%128)*256+ t*64 + k].x;
                        }
                    }
                }
            }

            auto status = GetPrecisionStats(values_computed, values_want);
            cout<<status.String();

            scheme.decryptMsg(dec_m1, sk, c4);
            context.decode(dec_m1, complex_msg_dec);
            cudaMemcpy(res1, complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);            
            cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
            for (int i=0; i<slots; i++){
                values_want[i].x = 0;
                values_want[i].y = 0;
            }
            for (int j=0; j<64; j++){
                for (int i = 0; i<128; i++){
                    for (int t = 0; t<4; t++){
                        for (int k = 0; k<64; k++){
                            values_want[i*256 + t*64 +j].x += mes1[i*256+ t*64 + k].x * mes2[((i+j+64)%128)*256+ t*64 + k].x;
                        }
                    }
                }
            }
            status = GetPrecisionStats(values_computed, values_want);
            cout<<status.String();
            /******************** end Verify Q*K^T *********************************/

            /******************** Verify *V *********************************/
            // verify tau
            // cuDoubleComplex *target_res1;
            // target_res1 = new cuDoubleComplex[slots];
            // Plaintext dec_m1(N, L, target_level, slots, NTL::RR(context.precision));
            // read_matrix_from_file("python/data/tau_V.txt", target_res1, slots);
            // scheme.decryptMsg(dec_m1, sk, c1);
            // context.decode(dec_m1, complex_msg_dec);
            // vector<cuDoubleComplex> values_computed(slots);
            // cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
            // vector<cuDoubleComplex> values_want(slots);
            // cudaMemcpy(values_want.data(), target_res1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToHost);
            // auto status = GetPrecisionStats(values_computed, values_want);
            // cout<<status.String();

            // // verify *V
            // // c2,c3 is sigma(A)
            // cuDoubleComplex *target_res_ccmmV;
            // target_res_ccmmV = new cuDoubleComplex[slots];
            // Plaintext dec_ccmmV(N, L, target_level, slots, NTL::RR(context.precision));
            // memset(target_res_ccmmV, 0, sizeof(cuDoubleComplex) * slots);
            // read_matrix_from_file("python/data/mul_V_res.txt", target_res_ccmmV, slots);

            // scheme.decryptMsg(dec_ccmmV, sk, c4);
            // context.decode(dec_ccmmV, complex_msg_dec);
            // // vector<cuDoubleComplex> values_computed(slots);
            // cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
            // // vector<cuDoubleComplex> values_want(slots);
            // cudaMemcpy(values_want.data(), target_res_ccmmV, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToHost);

            // // auto status = GetPrecisionStats(values_computed, values_want);
            // status = GetPrecisionStats(values_computed, values_want);
            // // for (int i = 0; i < slots; i++){
            // //     if (abs(values_computed[i].x-values_want[i].x)>0.01)
            // //     cout << i << ' ';
            // // }
            // cout << endl;
            // cout<<status.String();

            /******************** end Verify *V *********************************/
        }

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