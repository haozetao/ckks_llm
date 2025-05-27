#pragma once

#include "uint128.cuh"

// PubKey and AutoKey
class Key {
public:
    uint64_tt *ax_device = nullptr;
    uint64_tt *bx_device = nullptr;
    uint64_tt *cipher_device = nullptr;
    // Ring dim
	int N;
    // num of q
    int L;
    // num of p
    int K;
    // decompose num
    int dnum;
    Key(){cipher_device = nullptr;}
	Key(int N, int L, int K, int dnum) : N(N), L(L), K(K), dnum(dnum)
	{
        cudaMalloc(&cipher_device, sizeof(uint64_tt) * N * (L + 1 + K) * dnum * 2);
        ax_device = cipher_device;
        bx_device = cipher_device + N * (L + 1 + K) * dnum;
	}

	Key(uint64_tt* cipher, int N, int L, int K, int dnum) : N(N), L(L), K(K), dnum(dnum)
	{
        cipher_device = cipher;
        ax_device = cipher_device;
        bx_device = cipher_device + N * (L + 1 + K) * dnum;
	}

    Key(const Key& c) : N(c.N), L(c.L), K(c.K), dnum(c.dnum)
    {
        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&cipher_device, sizeof(uint64_tt) * N * (L + 1 + K) * dnum * 2);
        }
        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N * (L + 1 + K) * dnum * 2, cudaMemcpyDeviceToDevice);
        }
        ax_device = cipher_device;
        bx_device = cipher_device + N * (L + 1 + K) * dnum;
    }

    Key& operator = (const Key& key)
    {
        if(this == &key) return *this;
        
        N = key.N;
        L = key.L;
        K = key.K;

        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&cipher_device, sizeof(uint64_tt) * N * (L + 1 + K) * dnum * 2);
            ax_device = cipher_device;
            bx_device = cipher_device + N * (L + 1 + K) * dnum;
        }
        cudaMemcpy(this->cipher_device, key.cipher_device, sizeof(uint64_tt) * N * (L + 1 + K) * dnum * 2, cudaMemcpyDeviceToDevice);
        
        return *this;
    }

	virtual ~Key()
	{
        if(cipher_device != nullptr)
        {
            cudaDeviceSynchronize();
            cudaFree(cipher_device);
            cudaDeviceSynchronize();
            cipher_device = nullptr;
        }
        ax_device = nullptr;
        bx_device = nullptr;
        cipher_device = nullptr;
	}
};