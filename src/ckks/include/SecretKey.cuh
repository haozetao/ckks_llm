#pragma once

#include "uint128.cuh"
#include "Context_23.h"
#include "RNG.cuh"
#include "Sampler.cuh"
#include "ntt_60bit.cuh"

// SecretKey
class SecretKey {
public:

	uint64_tt* sx_host;

	uint64_tt* sx_device;
    // Ring dim
	int N;
	int L, K;

	SecretKey(Context_23& context, cudaStream_t stream = 0)
	{
		N = context.N;
		L = context.L;
		K = context.K;
		int logN = context.logN;
		sx_host = new uint64_tt[N * (K+L+1)];
        cudaMalloc(&sx_device, sizeof(uint64_tt) * N * (K+L+1));
		
        Sampler::HWTSampler(context.randomArray_sk_device, sx_device, logN, context.h, 0, 0, K+L+1);
		//context.forwardNTT_batch(sx_device, 0, 0, 1, K+L+1);
		context.ToNTTInplace(sx_device, 0, 0, 1, K+L+1, K+L+1);
	}

    SecretKey operator = (SecretKey Secretkey)
    {
        if(this == &Secretkey) return *this;
        
        N = Secretkey.N;
        
        cudaMemcpy(this->sx_device, Secretkey.sx_device, sizeof(uint64_tt) * N, cudaMemcpyDeviceToDevice);

        return *this;
    }

	virtual ~SecretKey()
	{
		delete sx_host;

		cudaDeviceSynchronize();
		cudaFree(sx_device);
		cudaDeviceSynchronize();
	}

	void copyToHost()
	{
		cudaMemcpy(sx_host, sx_device, sizeof(uint64_tt) * N, cudaMemcpyDeviceToHost);
	}
};