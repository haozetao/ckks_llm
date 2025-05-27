#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "Utils.cuh"
#include "uint128.cuh"
#include "RNG_impl.cuh"


namespace RNG
{
    int threadsPerBlock, blocksPerGrid;
    uint8_tt k[32] = {0x95, 0xb0, 0x36, 0x49, 0xda, 0x89, 0x24, 0xb8, 0x80, 0x10, 0x14, 0xb3, 0x1, 0x21, 0x75, 0x7b, 0x69, 0x93, 0x3, 0x31, 0x79, 0x55, 0x3b, 0xd, 0xda, 0x37, 0x65, 0xd, 0x80, 0xd6, 0xfb, 0x87};
    uint64_t v_nonce;
    uint8_tt h_nonce[XSALSA20_CRYPTO_NONCEBYTES] = {0x26, 0x3a, 0x81, 0x67, 0xbd, 0xda, 0x5e, 0x4d, 0x33, 0x7a, 0x6e, 0xd6, 0x26, 0xda, 0x20, 0xee, 0x74, 0x19, 0xef, 0xa4, 0xcd, 0x85, 0x40, 0x7d,};

    void generateRandom_device(uint8_tt* a, uint64_tt n)
    {
        uint64_tt N, NBLKS = n / 64;
        uint64_tt size = NBLKS * XSALSA20_BLOCKSZ;
        // cout<<"NBLKS: "<<NBLKS<<endl;
        // cout<<"size: "<<size<<endl;

        // memset(k, 1, XSALSA20_CRYPTO_KEYBYTES);
        // memset(h_nonce, 12, XSALSA20_CRYPTO_NONCEBYTES);
        randomArray8(k,XSALSA20_CRYPTO_KEYBYTES,0xff);
        randomArray8(h_nonce,XSALSA20_CRYPTO_NONCEBYTES,0xff);

        // cout<<"k: ";
        // for(int i = 0; i < XSALSA20_CRYPTO_KEYBYTES; i++)
        // {
        //     printf("0x%x, ", k[i]);
        // }
        // cout<<endl<<"h_nonce: ";
        // for(int i = 0; i < XSALSA20_CRYPTO_NONCEBYTES; i++)
        // {
        //     printf("0x%x, ", h_nonce[i]);
        // }
        // cout<<endl;

        cudaMemcpyToSymbolAsync(key, k, XSALSA20_CRYPTO_KEYBYTES, 0, cudaMemcpyHostToDevice); //re add async
        v_nonce = load_littleendian64(h_nonce);
        threadsPerBlock = THREADS_PER_BLOCK;

        cudaMemsetAsync(a, 0, size); //re add async

        N = NBLKS;
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        VecCrypt <<<blocksPerGrid, threadsPerBlock, 0, 0 >>> (a, N, size, v_nonce, 1);
    }
};