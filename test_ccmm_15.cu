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

void read_matrix_from_file(const std::string& filename, cuDoubleComplex* mes, int rows, int cols);
int main(int argc, char* argv[]){
    int logN = 15;
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

    Scheme_23 scheme(context);
    cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    cout<<"Generate sk OK"<<endl;

    scheme.addEncKey(sk);
    cout<<"Generate pk OK"<<endl;

    scheme.addMultKey_23(sk);
    scheme.addLeftRotKey_23(sk, 1);
    scheme.addConjKey_23(sk);

    cout<<"Generate rlk OK"<<endl;

    int token_len = 128;
    int column_num = slots / token_len;
    int head_blocks = column_num / 64;

    SchemeAlgo scheme_algo(context, scheme, sk);
    Attention attention_scheme(context, scheme, scheme_algo, token_len, 12, 64, sk);
    attention_scheme.addKey(sk);

    cuDoubleComplex* mes1, *mes2;
	mes1 = new cuDoubleComplex[slots];
    mes2 = new cuDoubleComplex[slots];
	cuDoubleComplex* mes3 = new cuDoubleComplex[slots];

    read_matrix_from_file("python/data/V.txt", mes1, token_len, column_num);
    read_matrix_from_file("python/data/sigma_A_1.txt", mes2, token_len, column_num);
    read_matrix_from_file("python/data/sigma_A_2.txt", mes3, token_len, column_num);

    cuDoubleComplex* complex_msg1, *complex_msg2, *complex_msg3;
    cudaMalloc(&complex_msg1, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg2, sizeof(cuDoubleComplex) * slots);
    cudaMalloc(&complex_msg3, sizeof(cuDoubleComplex) * slots);

	cudaMemcpy(complex_msg1, mes1, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg2, mes2, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
	cudaMemcpy(complex_msg3, mes3, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

    int target_level = L-3;
    {
        Plaintext plain_m1(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m2(N, L, target_level, slots, NTL::RR(context.precision));
        Plaintext plain_m3(N, L, target_level, slots, NTL::RR(context.precision));

        Ciphertext c1(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c2(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c3(N, L, L, slots, NTL::RR(context.precision));
        Ciphertext c4(N, L, L, slots, NTL::RR(context.precision));

        cuDoubleComplex* complex_msg_dec;
        cudaMalloc(&complex_msg_dec, sizeof(cuDoubleComplex) * slots);

        context.encode(complex_msg1, plain_m1);
        context.encode(complex_msg2, plain_m2);
        context.encode(complex_msg3, plain_m3);

        scheme.encryptMsg(c1, plain_m1);
        scheme.encryptMsg(c2, plain_m2);
        scheme.encryptMsg(c3, plain_m3);

        attention_scheme.CCMM_QK(c1, c2, c3, c4);

        Plaintext dec_m1(N, L, target_level, slots, NTL::RR(context.precision));
        vector<cuDoubleComplex> values_computed(slots);
        vector<cuDoubleComplex> values_want(slots);

        for (int i = 0; i < slots; i++){
            values_want[i].x = 0;
            values_want[i].y = 0;
        }

        scheme.decryptMsg(dec_m1, sk, c3);
        context.decode(dec_m1, complex_msg_dec);
        cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        for (int j=0; j<64; j++){
            for (int i = 0; i<token_len; i++){
                for (int t = 0; t<head_blocks; t++){
                    for (int k = 0; k<64; k++){
                        values_want[i*column_num + t*64 + j].x += mes1[i*column_num + t*64 + k].x * mes2[((i + j) % token_len) * column_num + t*64 + k].x;
                    }
                }
            }
        }

        auto status = GetPrecisionStats(values_computed, values_want);
        cout<<status.String();

        for (int i = 0; i < slots; i++){
            values_want[i].x = 0;
            values_want[i].y = 0;
        }

        scheme.decryptMsg(dec_m1, sk, c4);
        context.decode(dec_m1, complex_msg_dec);
        cudaMemcpy(values_computed.data(), complex_msg_dec, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);

        for (int j=0; j<64; j++){
            for (int i = 0; i<token_len; i++){
                for (int t = 0; t<head_blocks; t++){
                    for (int k = 0; k<64; k++){
                        values_want[i*column_num + t*64 + j].x += mes1[i*column_num + t*64 + k].x * mes2[((i + j + token_len/2) % token_len) * column_num + t*64 + k].x;
                    }
                }
            }
        }

        status = GetPrecisionStats(values_computed, values_want);
        cout<<status.String();
    }

    return 0;
}

void read_matrix_from_file(const std::string& filename, cuDoubleComplex* mes, int rows, int cols) {
    const int want = rows * cols;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        for (int i = 0; i < want; i++) {
            mes[i].x = 0.0;
            mes[i].y = 0.0;
        }
        return;
    }

    std::vector<double> values;
    values.reserve(want);

    std::string token;
    while (file >> token) {
        try {
            values.push_back(std::stod(token));
        } catch (...) {
        }
    }
    file.close();

    const int file_cols = (int(values.size()) >= rows * cols * 2) ? (cols * 2) : cols;

    for (int i = 0; i < want; i++) {
        mes[i].x = 0.0;
        mes[i].y = 0.0;
    }

    if (int(values.size()) < rows * file_cols) {
        const int copy_n = std::min<int>(int(values.size()), want);
        for (int i = 0; i < copy_n; i++) {
            mes[i].x = values[i];
        }
        return;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            mes[r * cols + c].x = values[r * file_cols + c];
        }
    }
}
