#include <iostream>
#include <cblas.h>
#include <random>
#include <chrono>
#include <omp.h>
using namespace std;

int M = 65536, N = 4096, K = 4096;

// 定义矩阵A、B和结果矩阵C

int main() {
    // 定义矩阵A和B的维度
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    std::random_device rd;  //如果可用的话，从一个随机数发生器上获得一个真正的随机数
    std::mt19937 gen(rd()); //gen是一个使用rd()作种子初始化的标准梅森旋转算法的随机数发生器
    std::uniform_real_distribution<> distribution(0, 1);
    
    #pragma omp parallel for num_threads(16)
    for(int i = 0; i < M*K; i++){
        A[i] = distribution(gen);
    }
    for(int i = 0; i < 4; i++) cout<<A[i]<<", ";
    cout<<endl;

    #pragma omp parallel for num_threads(16)
    for(int i = 0; i < K*N; i++){
        B[i] = distribution(gen);
    }
    for(int i = 0; i < 4; i++) cout<<B[i]<<", ";
    cout<<endl;
    // 执行矩阵乘法 C = alpha * A * B + beta * C
    // 这里 alpha = 1.0, beta = 0.0

    auto start = std::chrono::high_resolution_clock::now();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << duration.count() << " us to execute." << std::endl;


    // 打印结果矩阵C
    printf("矩阵乘法结果:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}