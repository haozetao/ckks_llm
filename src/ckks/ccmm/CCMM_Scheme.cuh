#pragma once

#include "CCMM_Scheme.h"

CCMM_Scheme::CCMM_Scheme(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, int d, int head_num, int token_len)
    : context(context), scheme(scheme), scheme_algo(scheme_algo), d(d), head_num(head_num), token_len(token_len) {
    mult_buffer = vector<Ciphertext*>(2);
    for(int i = 0; i < 2; i++){
        mult_buffer[i] = scheme_algo.chebyshev_tree_pool[0];
    }
    rot_buffer = scheme_algo.chebyshev_tree_pool[2];

    int N = context.N;
    int L = context.L;
    int slots = context.slots;

    column_mask_buffer = new cuDoubleComplex[slots];
    cudaMalloc(&column_mask_buffer_device, sizeof(cuDoubleComplex) * slots);
    for(int i = 0; i < slots; i++){
        column_mask_buffer[i].x = 0;
        if(i % d == 0){
            column_mask_buffer[i].x = 1;
        }
        column_mask_buffer[i].y = 0;
    }
    column_mask = new Plaintext(N, L, L, slots, NTL::RR(pow(context.precision, 1)));
    cudaMemcpy(column_mask_buffer_device, column_mask_buffer, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
    context.encode(column_mask_buffer_device, *column_mask);

    // 256 and more
    if(token_len >= 256){
        int row_size = slots / token_len;
        printf("row_size: %d \n", row_size);

        for(int diag_idx = 0; diag_idx < row_size + 1; diag_idx++){
            Plaintext* temp_diag = new Plaintext(N, L, L, slots, NTL::RR(pow(context.precision, 1)));
            diag_mask.push_back(temp_diag);

            memset(column_mask_buffer, 0, sizeof(cuDoubleComplex) * slots);

            for(int i = 0; i < token_len; i++){
                if(i < row_size + 1){
                    for(int j = 0; j < row_size; j++){
                        if(i == j + row_size) {
                            column_mask_buffer[i*row_size + j].x = 1;                    
                            printf("(%d, %d)", i, j);
                        }
                    }
                }
            }
            printf("\n");

            cudaMemcpy(column_mask_buffer_device, column_mask_buffer, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(column_mask_buffer_device, *temp_diag);
        }

        for(int diag_idx = 0; diag_idx < token_len; diag_idx++){
            Plaintext* temp_diag = new Plaintext(N, L, L, slots, NTL::RR(pow(context.precision, 1)));
            diag_mask_2.push_back(temp_diag);


            // for(int i = 0; i < token_len; i++){
            //     int row_size = slots / token_len;
            //     for(int j = 0; j < row_size; j++){
            //         column_mask_buffer[i*row_size + j].x = 0;
            //         if(i > diag_idx + j){
            //             column_mask_buffer[i*row_size + j].x = 1;
            //         }
            //         column_mask_buffer[i*row_size + j].y = 0;
            //     }
            // }

            cudaMemcpy(column_mask_buffer_device, column_mask_buffer, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            context.encode(column_mask_buffer_device, *temp_diag);
        }


    } else {
        // 128 and less
        printf("Warning: do not support token_len < 256.\n");
    }
}

void CCMM_Scheme::addKey(SecretKey& sk)
{
    for(int i = 0; i < log(64) + 1; i++){
        scheme.addLeftRotKey_23(sk, 1 << i);
        printf("%d ", 1<<i);
        scheme.addLeftRotKey_23(sk, 32768 - (1 << i));
        printf("%d ", 32768 - (1 << i));
    }
}

void CCMM_Scheme::CCMM_QK(Ciphertext& Q, Ciphertext& K, Ciphertext& O)
{
    int slots = context.slots;

    if(d * token_len > slots){
        printf("Error QKV matrix must in one cipher\n");
    }

    scheme.mult_23(*mult_buffer[0], Q, K);
    scheme.rescaleAndEqual(*mult_buffer[0]);
    // *mult_buffer = Q;

    // reduce sum
    for(int i = log(d) + 1; i >= 0; i--){
        scheme.leftRotateAddSelf_23(*mult_buffer[0], 1 << i);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    multConstDiagAndEqual(*mult_buffer[0], *column_mask);

    // repeate
    for(int i = 0; i < log(d) + 1; i++){
        scheme.leftRotateAddSelf_23(*mult_buffer[0], 32768 - (1 << i));
    }
    *mult_buffer[1] = *mult_buffer[0];
    multConstDiagAndEqual(*mult_buffer[0], *diag_mask[0]);

    // mask

    O = *mult_buffer[0];
}

void CCMM_Scheme::multConstDiagAndEqual(Ciphertext& cipher, Plaintext& cnst, int rot_num)
{
    int L = context.L;
    int N = context.N;
    int K = context.K;

    barrett_2batch_device(cipher.cipher_device, cnst.mx_device, N, 0, 0, K, cipher.l+1, L+1);

    cipher.scale *= cnst.scale;
    scheme.rescaleAndEqual(cipher);
}