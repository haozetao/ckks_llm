#pragma once

#include "uint128.cuh"
#include "cuComplex.h"

#include <algorithm>
#include <stdlib.h>
#include <random>
#include <set>
#include <iostream>

using namespace std;

uint64_tt modpow128(uint64_tt a, uint64_tt b, uint64_tt mod)
{
    uint64_tt res = 1;
    while(b)
    {
        if(b & 1)
        {
            __uint128_t r128 = res;
            r128 *= a;
            res = uint64_tt(r128 % mod);
        }
        __uint128_t t128 = a;
        t128 *= a;
        a = uint64_tt(t128 % mod);
        b >>= 1;
    }
    return res;
}

uint64_tt mulMod128(uint64_tt a, uint64_tt b, uint64_tt q)
{
    __uint128_t c = a;
    c *= b;
    return uint64_tt(c % q);
}

uint64_tt x_Shoup(uint64_tt x, uint64_tt q)
{
    __uint128_t temp = x;
    temp <<= 64;
    return uint64_tt(temp / q);
}

//return a^-1 mod q
uint64_tt modinv128(uint64_tt a, uint64_tt q)
{
    a = a % q;
    uint64_tt ainv = modpow128(a, q - 2, q);
    return ainv;
}

uint64_tt bit_length(uint64_tt a)
{
    uint64_tt len = 0;        // 初始长度为0
    for(; a > 0; ++len)    // 判断num是否大于0，否则长度+1
        a >>= 1;	         // 使用除法进行运算，直到num小于1
    return len;
}

void findPrimeFactors(set<uint64_tt> &s, uint64_tt number) {
	while (number % 2 == 0) {
		s.insert(2);
		number /= 2;
	}
	for (uint64_t i = 3; i < sqrt(number); i++) {
		while (number % i == 0) {
			s.insert(i);
			number /= i;
		}
	}
	if (number > 2) {
		s.insert(number);
	}
}

uint64_tt findPrimitiveRoot(uint64_tt modulus) {
	set<uint64_tt> s;
	uint64_tt phi = modulus - 1;
	findPrimeFactors(s, phi);
	for (uint64_tt r = 2; r <= phi; r++) {
		bool flag = false;
		for (auto it = s.begin(); it != s.end(); it++) {
			if (modpow128(r, phi / (*it), modulus) == 1) {
				flag = true;
				break;
			}
		}
		if (flag == false) {
			return r;
		}
	}
	return -1;
}

uint64_tt findMthRootOfUnity(uint64_tt M, uint64_tt mod) {
    uint64_tt res;
    res = findPrimitiveRoot(mod);
    if((mod - 1) % M == 0) {
        uint64_tt factor = (mod - 1) / M;
        res = modpow128(res, factor, mod);
        return res;
    }
    else {
        return -1;
    }
}

// __host__ __device__ uint64_tt bitReverse(uint64_tt a, int bit_length)
// {
//     uint64_tt res = 0;

//     for (int i = 0; i < bit_length; i++)
//     {
//         res <<= 1;
//         res = (a & 1) | res;
//         a >>= 1;
//     }

//     return res;
// }

__forceinline__ __host__ __device__ uint32_t bitReverse(uint32_t operand, int bit_count) noexcept {
        operand = (((operand & uint32_t(0xaaaaaaaa)) >> 1) | ((operand & uint32_t(0x55555555)) << 1));
        operand = (((operand & uint32_t(0xcccccccc)) >> 2) | ((operand & uint32_t(0x33333333)) << 2));
        operand = (((operand & uint32_t(0xf0f0f0f0)) >> 4) | ((operand & uint32_t(0x0f0f0f0f)) << 4));
        operand = (((operand & uint32_t(0xff00ff00)) >> 8) | ((operand & uint32_t(0x00ff00ff)) << 8));
        operand = (operand >> 16) | (operand << 16);
        return operand >> (32 - bit_count);
}

std::random_device dev;
std::mt19937_64 rng(dev());

void randomComplexArray(cuDoubleComplex* ComplexArray, long slots, double bound = 1.0)
{
    std::uniform_int_distribution<int> randnum(0, RAND_MAX);

	for (long i = 0; i < slots; ++i) {
		// ComplexArray[i].x = ((double) rand()/(RAND_MAX) - 0.5) * 2 * bound;
        // ComplexArray[i].y = ((double) rand()/(RAND_MAX) - 0.5) * 2 * bound;
        // ComplexArray[i].x = (double) randnum(rng)/(RAND_MAX) * bound;
        // ComplexArray[i].y = (double) randnum(rng)/(RAND_MAX) * bound;

		// ComplexArray[i].x = (double) rand()/(RAND_MAX) * bound;
		ComplexArray[i].x = (10 + i % 10) / 100.;
		ComplexArray[i].y = 0;

        // ComplexArray[i].y = (double) rand()/(RAND_MAX) * bound;
	}
}

void randomDoubleArray(double* doubleArray, long num, double bound = 1.0)
{
    std::uniform_real_distribution<double> randnum(0, bound);

	for (long i = 0; i < num; ++i) {
		// doubleArray[i] = randnum(rng);
		doubleArray[i] = (i%256)/10000.;

	}
}

void randomFloatArray(float* doubleArray, long num, double bound = 1.0)
{
    std::uniform_real_distribution<float> randnum(-bound, bound);

	for (long i = 0; i < num; ++i) {
		doubleArray[i] = randnum(rng);
		// doubleArray[i] = (i%256)/10000.;

	}
}

// void fillTablePsi128(uint64_tt psi, uint64_tt q, uint64_tt psiinv, uint64_tt psiTable[], uint64_tt psiinvTable[], uint32_tt n)
// {
//     psiTable[0] = psiinvTable[0] = 1;
//     for (int i = 1; i < n; i++)
//     {
//         int idx_prev = bitReverse(i-1, log2(n));
//         int idx_next = bitReverse(i, log2(n));
//         psiTable[idx_next] = mulMod128(psi, psiTable[idx_prev], q);
//         psiinvTable[idx_next] = mulMod128(psiinv, psiinvTable[idx_prev], q);
//     }
// }

void fillTablePsi128_special(uint64_tt psi, uint64_tt q, uint64_tt psiinv, uint64_tt psiTable[], uint64_tt psiinvTable[], uint32_tt n, uint64_tt inv_degree)
{
    psiTable[0] = psiinvTable[0] = 1;
    for (int i = 1; i < n; i++)
    {
        int idx_prev = bitReverse(i-1, log2(n));
        int idx_next = bitReverse(i, log2(n));
        psiTable[idx_next] = mulMod128(psi, psiTable[idx_prev], q);
        psiinvTable[idx_next] = mulMod128(psiinv, psiinvTable[idx_prev], q);
    }
    psiinvTable[1] = mulMod128(psiinvTable[1], inv_degree, q);
}

void fillTablePsi_shoup128(uint64_tt psiTable[], uint64_tt q, uint64_tt psiinv_Table[], uint64_tt psi_shoup_table[], uint64_tt psiinv_shoup_table[], uint32_tt n)
{
    for (int i = 0; i < n; i++)
    {
        psi_shoup_table[i] = x_Shoup(psiTable[i], q);
        psiinv_shoup_table[i] = x_Shoup(psiinv_Table[i], q);
    }
}

void bitReverseArray(uint64_tt array[], uint32_tt n)
{
    uint64_tt* temp = (uint64_tt*)malloc(sizeof(uint64_tt) * n);
    uint32_tt log2n = log2(n);
    for(int i = 0; i < n; i++)
    {
        temp[i] = array[bitReverse(i, log2n)];
    }
    memcpy(array, temp, sizeof(uint64_tt) * n);
    free(temp);
}

void randomArray128(uint64_tt a[], int n, uint64_tt q)
{
    std::uniform_int_distribution<uint64_tt> randnum(0, q - 1);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void randomArray64(uint32_tt a[], int n, uint32_tt q)
{
    std::uniform_int_distribution<uint32_tt> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void randomArray8(uint8_tt a[], int n, uint8_tt q)
{
    std::uniform_int_distribution<uint8_tt> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

//poly a * poly b on  Zm[x]/(x^n+1)
uint64_tt* refPolyMul128(uint64_tt a[], uint64_tt b[], uint64_tt m, int n)
{
    uint64_tt* c = (uint64_tt*)malloc(sizeof(uint64_tt) * n * 2);
    uint64_tt* d = (uint64_tt*)malloc(sizeof(uint64_tt) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = (__uint128_t(a[i]) * b[j] + c[i + j]) % m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}

uint32_tt* refPolyMul64(uint32_tt a[], uint32_tt b[], uint32_tt m, int n)
{
    uint32_tt* c = (uint32_tt*)malloc(sizeof(uint32_tt) * n * 2);
    uint32_tt* d = (uint32_tt*)malloc(sizeof(uint32_tt) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = ((uint64_tt)a[i] * b[j]) % m + c[i + j] % m;
            c[i + j] %= m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}

void print_device_array(int* data, int N, int mod_num, const char* vname)
{
    int* array_PQ = new int[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(int) * N * mod_num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+256; t++)
        {
            printf("%llu, ", array_PQ[i*N + t]);
            if(t % 16 == 15) printf("\n");
        }
        printf("],\n");
    }
    printf("]\n");
    delete array_PQ;
}

void print_device_array(float* data, int N, int mod_num, const char* vname)
{
    float* array_PQ = new float[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(float) * N * mod_num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+8; t++)
        {
            printf("%f, ", array_PQ[i*N + t]);
        }
        printf("],\n");
    }
    printf("]\n");
    delete array_PQ;
}


void print_device_array(uint64_tt* data, int N, int mod_num, const char* vname)
{
    uint64_tt* array_PQ = new uint64_tt[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * mod_num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+16; t++)
        {
            printf("%llu, ", array_PQ[i*N + t]);
        }
        printf("],\n");
    }
    // printf("]\n");
    delete array_PQ;
}

void check_array(uint64_tt* data, int N, int mod_num, const char* vname, uint64_tt q)
{
    uint64_tt* array_PQ = new uint64_tt[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * mod_num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+N; t++)
        {
            if(array_PQ[i*N + t] >= q)
                printf("error: array[%d] = %llu > q!\n", i*N + t, array_PQ[i*N + t]);
            // printf("%d, ", i*N + t);
        }
        printf("],\n");
    }
    printf("]\n");
    delete array_PQ;
}


void check_array_equal(uint64_tt* data1, uint64_tt* data2, int N, int mod_num, const char* vname)
{
    uint64_tt* array_PQ1 = new uint64_tt[N * mod_num];
    uint64_tt* array_PQ2 = new uint64_tt[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(uint64_tt) * N * mod_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(uint64_tt) * N * mod_num, cudaMemcpyDeviceToHost);

    printf("%s = [", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+N; t++)
        {
            if(array_PQ1[i*N + t] != array_PQ2[i*N + t])
                printf("error: array1[%d] != array2[%d] %llu %llu > q!\n", i*N + t, i*N + t, array_PQ1[i*N + t], array_PQ2[i*N + t]);
        }
        printf("],\n");
    }
    printf("]\n");
    delete array_PQ1;
    delete array_PQ2;
}

void print_device_array(cuDoubleComplex* data, int slots, const char* vname)
{
    cuDoubleComplex* array_PQ = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    cout.precision(8);
    for(int i = start; i < start+8; i++)
    {
        printf("%.8lf + i*%.8lf, ", array_PQ[i].x, array_PQ[i].y);
        // cout << fixed << array_PQ[i].x << " + i*" << array_PQ[i].y << ", ";
    }
    printf("]\n");
    delete array_PQ;
}

void print_device_array(double* data, int num, const char* vname)
{
    double* array_PQ = new double[num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(double) * num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    cout.precision(8);
    double max_value = 0;
    for(int i = start; i < start+num; i++)
    {
        if(array_PQ[i] > max_value) max_value = array_PQ[i];
    }
    for(int i = start; i < start+8; i++)
    {
        printf("%.8lf, ", array_PQ[i]);
    }
    printf("], max: %.8lf\n", max_value);
    delete array_PQ;
}

void print_device_array(double* data, int num, const char* vname, int rank, int offset)
{
    double* array_PQ = new double[num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(double) * num, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    cout.precision(8);
    int count = 8;
    for(int i = start; i < start+num; i++)
    {
        if(i % rank == offset) {
            if(count-- >= 0)
                printf("%.8lf, ", array_PQ[i]);
        }
    }
    printf("]\n");
    delete array_PQ;
}

int compare_device_array(cuDoubleComplex* data1, cuDoubleComplex* data2, int slots, const char* vname)
{
    cuDoubleComplex* array_PQ1 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ2 = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    int cnt = 0;
    for(int i = 0; i < slots; i++)
    {
        if(abs(array_PQ1[i].x - array_PQ2[i].x) < 1e-3 && abs(array_PQ1[i].y - array_PQ2[i].y) < 1e-3) continue;
        // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        // printf("%d, %lf+%lfi vs %lf+%lfi\n", i, array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        cnt++;
    }
    // printf("%s error num: %d\n", vname, cnt);
    delete array_PQ1;
    delete array_PQ2;

    return cnt;
}

void compare_device_array(cuDoubleComplex* data1, cuDoubleComplex* data2, cuDoubleComplex* data3, int slots, const char* vname)
{
    cuDoubleComplex* array_PQ1 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ2 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ3 = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ3, data3, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    int cnt = 0;
    for(int i = 0; i < slots; i++)
    {
        array_PQ3[i] = cuCmul(array_PQ2[i], array_PQ3[i]);
        if(abs(array_PQ1[i].x - array_PQ3[i].x) < 1e-5 && abs(array_PQ1[i].y - array_PQ3[i].y) < 1e-5) continue;
        // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        // printf("%d, %lf+%lfi vs %lf+%lfi\n", i, array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        cnt++;
    }
    printf("%s error num: %d\n", vname, cnt);
    delete array_PQ1;
    delete array_PQ2;
    delete array_PQ3;
}

void compare_device_array(uint64_tt* data1, uint64_tt* data2, int N, int len, const char* vname)
{
    uint64_tt* array_PQ1 = new uint64_tt[len*N];
    uint64_tt* array_PQ2 = new uint64_tt[len*N];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(uint64_tt) * len*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(uint64_tt) * len*N, cudaMemcpyDeviceToHost);
    printf("%s error num: ", vname);
    int cnt = 0;
    for(int idx = 0; idx < len; idx++)
    {
        cnt = 0;
        for(int i = 0; i < N; i++)
        {
            if(array_PQ1[idx*N + i] == array_PQ2[idx*N + i]) continue;
            // else printf("%d %d %llu %llu\n", idx, i, array_PQ1[idx*N + i], array_PQ2[idx*N + i]);
            // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
            // printf("%d, %llu vs %llu\n", i, array_PQ1[i], array_PQ2[i]);
            cnt++;
        }
        printf("(%d,%d), ", idx, cnt);
    }
    cout<<endl;
    delete array_PQ1;
    delete array_PQ2;
}

void print_device_array(uint128_tt* data, int N, int len, const char* vname)
{
    uint128_tt* array_PQ = new uint128_tt[N * len];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint128_tt) * N * len, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    for(int i = 0; i < len; i++)
    {
        printf("[");
        for(int t = 0; t < 4; t++)
        {
            printf("(%llx,%llx), ", array_PQ[i*N + t].high, array_PQ[i*N + t].low);
        }
        printf("],");
    }
    printf("]\n");
    delete array_PQ;
}

void checkHWT(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);
    
    printf("array_PQ = [\n");
    for(int t = 0; t < N; t++)
    {
        if(array_PQ[t] != 0)
        printf("%llu %d\n", array_PQ[t], t);
    }
    printf("]\n");
    delete array_PQ;
}

void count_Zero(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);

    printf("array_PQ non zero count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] != 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");
    delete array_PQ;
}

void count_ZO(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);

    printf("array_PQ 0 count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] == 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");

    printf("array_PQ +-1 count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] != 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");
    delete array_PQ;
}

void count_Gaussian(uint64_tt* data, int N, int K, int L, vector<uint64_tt> q_vec)
{
    uint64_tt* array_PQ = new uint64_tt[N*(L)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N, cudaMemcpyDeviceToHost);

    vector<int> count(50, 0);
    uint64_tt q = q_vec[0];
    uint64_tt half_q = q / 2;

    printf("array gaussian count:\n");
    int cnt = 0;
    for(int t = 0; t < N; t++)
    {
        int64_t now = array_PQ[t];
        if(now >= half_q)
            now -= int64_t(q);
        count[now + 25]++;
    }

    for(int i = 0; i < 50; i++){
        cout<<count[i]<<", ";
    }
    cout<<endl;
    delete array_PQ;
}