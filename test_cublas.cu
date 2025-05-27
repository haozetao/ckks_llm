#include <cublas_v2.h>
#include <iostream>
// #include <cblas.h>
#include <random>
#include <chrono>
#include <omp.h>
using namespace std;
int M = 4096, N = 4096, K = 65536;

int main() {
    // 初始化 cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // 定义矩阵A和B的维度
    double *h_A = new double[M * K];
    double *h_B = new double[K * N];
    double *h_C = new double[M * N];


    std::random_device rd;  //如果可用的话，从一个随机数发生器上获得一个真正的随机数
    std::mt19937 gen(rd()); //gen是一个使用rd()作种子初始化的标准梅森旋转算法的随机数发生器
    std::uniform_real_distribution<> distribution(0, 1);
    

    // #pragma omp parallel for num_threads(16)
    // for(int i = 0; i < M*K; i++){
    //     h_A[i] = distribution(gen);
    // }
    // for(int i = 0; i < 4; i++) cout<<h_A[i]<<", ";
    // cout<<endl;

    // #pragma omp parallel for num_threads(16)
    // for(int i = 0; i < K*N; i++){
    //     h_B[i] = distribution(gen);
    // }
    // for(int i = 0; i < 4; i++) cout<<h_B[i]<<", ";
    // cout<<endl;

    // 分配设备内存
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(double));
    cudaMalloc((void**)&d_B, K * N * sizeof(double));
    cudaMalloc((void**)&d_C, M * N * sizeof(double));

    // 将数据从主机拷贝到设备
    cublasSetMatrix(M, K, sizeof(double), h_A, M, d_A, M);
    cublasSetMatrix(K, N, sizeof(double), h_B, K, d_B, K);

    // 执行矩阵乘法 C = A * B
    const double alpha = 1.0f, beta = 0.0f;

    
    auto start = std::chrono::high_resolution_clock::now();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << duration.count() << " us to execute." << std::endl;


    // 将结果从设备拷贝回主机
    cublasGetMatrix(M, N, sizeof(double), d_C, M, h_C, M);

    // 打印结果
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << h_C[i + j * M] << " ";
        }
        std::cout << std::endl;
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);

    return 0;
}