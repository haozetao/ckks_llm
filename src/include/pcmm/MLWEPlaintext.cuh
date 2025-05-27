#pragma once

#include "../uint128.cuh"
#include "../Utils.cuh"


class MLWEPlaintext {
public:
    MLWEPlaintext(){mx_device = nullptr;}
    MLWEPlaintext(int N1, int L, int l, NTL::RR scale) : N1(N1), L(L), l(l), scale(scale)
    {
        if(l > L) cout<<"new MLWEPlaintext error!"<<endl;
        cudaMalloc(&mx_device, sizeof(uint64_tt) * N1 * (L + 1));
    }

    MLWEPlaintext(const MLWEPlaintext& c) : N1(c.N1), L(c.L), l(c.l), scale(c.scale)
    {
        if(this->mx_device == nullptr)
        {
            cudaMalloc(&(this->mx_device), sizeof(uint64_tt) * N1 * (L + 1));
            this->mx_device = this->mx_device;
        }
        if(c.mx_device != nullptr)
        {
            cudaMemcpy(this->mx_device, c.mx_device, sizeof(uint64_tt) * N1 * (L + 1), cudaMemcpyDeviceToDevice);
        }
    }

    MLWEPlaintext& operator = (const MLWEPlaintext& m)
    {
        if(this == &m) return *this;
        
        N1 = m.N1;
        L = m.L;
        l = m.l;
        scale = m.scale;

        if(this->mx_device == nullptr)
        {
            cudaMalloc(&mx_device, sizeof(uint64_tt) * N1 * (L + 1));
        }
        cudaMemcpy(mx_device, m.mx_device, sizeof(uint64_tt) * N1 * (L + 1), cudaMemcpyDeviceToDevice);
        return *this;
    }

    virtual ~MLWEPlaintext()
    {
        if(mx_device != nullptr)
        {
            cudaDeviceSynchronize();
            cudaFree(mx_device);
            cudaDeviceSynchronize();
            mx_device = nullptr;
        }
    }

    uint64_tt *mx_device = nullptr;
    // Ring dim
    int N1;
    // mul level
    int l;
    // to malloc memory
    int L;
    NTL::RR scale;
};