#pragma once

#include "uint128.cuh"

// PubKey and AutoKey
class Key_decomp {
public:
    uint64_tt *ax_device = nullptr;
    uint64_tt *bx_device = nullptr;
    uint64_tt *cipher_device = nullptr;
    // Ring dim
	int N;
    // decompose num
    int dnum;
    // t_num = T_num
    int t_num;
    int blockNum;
    Key_decomp(){cipher_device = nullptr;}
	Key_decomp(int N, int dnum, int t_num, int blockNum) : N(N), dnum(dnum), t_num(t_num), blockNum(blockNum)
	{
        cudaMalloc(&cipher_device, sizeof(uint64_tt) * N * t_num * blockNum * dnum * 2);
        ax_device = cipher_device;
        bx_device = cipher_device + N * t_num * blockNum * dnum;
	}

    Key_decomp(const Key_decomp& c) : N(N), dnum(dnum), t_num(t_num), blockNum(blockNum)
    {
        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&(this->cipher_device), sizeof(uint64_tt) * N * t_num * blockNum * dnum * 2);
        }
        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N * t_num * blockNum * dnum * 2, cudaMemcpyDeviceToDevice);
        }
        this->ax_device = this->cipher_device;
        this->bx_device = this->cipher_device + N * t_num * blockNum * dnum;
    }

    Key_decomp& operator = (const Key_decomp& key)
    {
        if(this == &key) return *this;
        
        N = key.N;
        dnum = key.dnum;
        t_num = key.t_num;

        if(this->cipher_device == nullptr)
        {
            cudaMalloc(&cipher_device, sizeof(uint64_tt) * N * t_num * blockNum * dnum * 2);
            ax_device = cipher_device;
            bx_device = cipher_device + N * t_num * blockNum * dnum;
        }
        cudaMemcpy(this->cipher_device, key.cipher_device, sizeof(uint64_tt) * N * t_num * blockNum * dnum * 2, cudaMemcpyDeviceToDevice);
        
        return *this;
    }

	virtual ~Key_decomp()
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