#pragma once

#include "../uint128.cuh"
#include "../Utils.cuh"

class MLWECiphertext {
public:
    // ax[0], ax[1], ..., ax[k-1], bx mod q0
    // ax[0], ax[1], ..., ax[k-1], bx mod q1
    uint64_tt *cipher_device = nullptr;

    // Ring dim
    int N1;
    // to malloc memory
    int L;
    // mul level
    int l;
    // MLWE rank
    int k;
    // scale
    NTL::RR scale;
    
    MLWECiphertext(){cipher_device = nullptr;}
    MLWECiphertext(int N1, int L, int l, int k, NTL::RR scale) : N1(N1), L(L), l(l), k(k), scale(scale)
    {
        cudaMalloc(&cipher_device, sizeof(uint64_tt) * N1 * (L + 1) * (k + 1));
    }

    MLWECiphertext(uint64_tt *cipher, int N1, int L, int l, int k, NTL::RR scale) : N1(N1), L(L), l(l), k(k), scale(scale)
    {
        cipher_device = cipher;
    }

    MLWECiphertext(const MLWECiphertext& c) : N1(c.N1), L(c.L), l(c.l), k(c.k), scale(c.scale)
    {
        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&(this->cipher_device), sizeof(uint64_tt) * N1 * (L + 1) * (k + 1));
        }
        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N1 * (L + 1) * (k + 1), cudaMemcpyDeviceToDevice);
        }
    }

    MLWECiphertext& operator = (MLWECiphertext& c)
    {
        if(this == &c) return *this;
        
        this->N1 = c.N1;
        this->L = c.L;
        this->l = c.l;
        this->k = c.k;
        this->scale = c.scale;

        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&cipher_device, sizeof(uint64_tt) * N1 * (L + 1) * (k + 1));
        }

        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N1 * (L + 1) * (k + 1), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    virtual ~MLWECiphertext()
    {
        if(cipher_device != nullptr)
        {
            cudaDeviceSynchronize();
            cudaFree(cipher_device);
            cudaDeviceSynchronize();
            cipher_device = nullptr;
        }
    }
};
 