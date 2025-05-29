#pragma once
#include "../uint128.cuh"
#include "../Utils.cuh"
#include "../SecretKey.cuh"

class MLWESecretKey {
public:
    uint64_tt *sx_device = nullptr;

    // Ring dim
    int N1;
    // to malloc memory
    int L;
    // MLWE rank
    int k;
    
    MLWESecretKey(){sx_device = nullptr;}
    MLWESecretKey(int N1, int L, int k) : N1(N1), L(L), k(k)
    {
        cudaMalloc(&sx_device, sizeof(uint64_tt) * N1 * k * (L + 1));
    }

    MLWESecretKey& operator = (MLWESecretKey& mlwe_sk)
    {
        if(this == &mlwe_sk) return *this;
        
        this->N1 = mlwe_sk.N1;
        this->L = mlwe_sk.L;
        this->k = mlwe_sk.k;

        if(this->sx_device == nullptr)
        {
            cudaMalloc(&sx_device, sizeof(uint64_tt) * N1 * k * (L + 1));
        }

        if(mlwe_sk.sx_device != nullptr)
        {
            cudaMemcpy(this->sx_device, mlwe_sk.sx_device, sizeof(uint64_tt) * N1 * k * (L + 1), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    virtual ~MLWESecretKey()
    {
        if(sx_device != nullptr)
        {
            cudaDeviceSynchronize();
            cudaFree(sx_device);
            cudaDeviceSynchronize();
            sx_device = nullptr;
        }
    }
};
 