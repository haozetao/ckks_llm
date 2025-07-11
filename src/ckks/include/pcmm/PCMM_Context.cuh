#pragma once

#include "PCMM_Context.h"
#include "smallntt_60bit.cuh"
#include "ringSwitch.cuh"

PCMM_Context::PCMM_Context(int N1, int mlwe_rank, vector<uint64_tt> p_ringpack, vector<uint64_tt> q_ringpack, Context_23& context): N1(N1), mlwe_rank(mlwe_rank), p_ringpack(p_ringpack), q_ringpack(q_ringpack), context(context)
{
    for(auto i : p_ringpack) pq_ringpack.push_back(i);
    for(auto i : q_ringpack) pq_ringpack.push_back(i);

    ringpack_p_count = p_ringpack.size();
    ringpack_q_count = q_ringpack.size();
    ringpack_pq_count = pq_ringpack.size();
    preComputeOnCPU();
    copyMemoryToGPU();
}

void PCMM_Context::preComputeOnCPU()
{
    // copy pq, compute psi, mu
    for(int i = 0; i < ringpack_pq_count; i++)
    {
        NTL::ZZ mu(1);
		mu <<= 128;
        cout<<"pq_ringpack[i]: "<<pq_ringpack[i]<<endl;
		mu /= pq_ringpack[i];
		pq_ringpack_mu.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pq_ringpack_mu_high.push_back(pq_ringpack_mu[i].high);
		pq_ringpack_mu_low.push_back(pq_ringpack_mu[i].low);
		uint64_tt root = findMthRootOfUnity(2*N1, pq_ringpack[i]);
		psi_pq_ringpack.push_back(root);
    }
    // for(auto i : pq_ringpack) printf("%llu, ", i);
    // cout<<endl;
    // for(auto i : psi_pq_ringpack) printf("%llu, ", i);
    // cout<<endl;


    // compute psi^-1, N1^-1
    for (int i = 0; i < ringpack_pq_count; i++){
        psi_inv_pq_ringpack.push_back(modinv128(psi_pq_ringpack[i], pq_ringpack[i])); // pPsiInv
        
        N1_inv_host.push_back(modinv128(N1, pq_ringpack[i]));
        N1_inv_shoup_host.push_back(x_Shoup(N1_inv_host[i], pq_ringpack[i]));
    }


    for(int i = 0; i < ringpack_q_count - 1; i++){
        uint64_tt temp = 1;
        for(int i = 0; i < ringpack_p_count; i++){
            temp = mulMod128(temp, modinv128(p_ringpack[i], q_ringpack[i]), q_ringpack[i]);
        }
        p_inv_mod_qi_host.push_back(temp);
        p_inv_mod_qi_shoup_host.push_back(x_Shoup(temp, q_ringpack[i]));
    }
}

void PCMM_Context::copyMemoryToGPU(){

    // cudaMalloc(&pq_ringpack_device, sizeof(uint64_tt) * ringpack_pq_count);
    // cudaMemcpy(pq_ringpack_device, pq_ringpack.data(), sizeof(uint64_tt) * ringpack_pq_count, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ringpack_pq_cons, pq_ringpack.data(), sizeof(uint64_tt) * ringpack_pq_count, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ringpack_pq_mu_high_cons, pq_ringpack_mu_high.data(), sizeof(uint64_tt) * ringpack_pq_count, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ringpack_pq_mu_low_cons, pq_ringpack_mu_low.data(), sizeof(uint64_tt) * ringpack_pq_count, 0, cudaMemcpyHostToDevice);

    //pqPsiTable and pqPsiInvTable
    uint64_tt** ringPackPsiTable = new uint64_tt*[ringpack_pq_count];
	uint64_tt** ringPackPsiInvTable = new uint64_tt*[ringpack_pq_count];

	/*******************************************N1-NTT******************************************************/
    for (int i = 0; i < ringpack_pq_count; i++)
	{
		ringPackPsiTable[i] = new uint64_tt[N1];
		ringPackPsiInvTable[i] = new uint64_tt[N1];
        fillTablePsi128_special(psi_pq_ringpack[i], pq_ringpack[i], psi_inv_pq_ringpack[i], ringPackPsiTable[i], ringPackPsiInvTable[i], N1, N1_inv_host[i]);
    }

	uint64_tt** psi_shoup_table = new uint64_tt*[ringpack_pq_count];
    uint64_tt** psiinv_shoup_table = new uint64_tt*[ringpack_pq_count];
    for (int i = 0; i < ringpack_pq_count; i++)
	{
		psi_shoup_table[i] = new uint64_tt[N1];
		psiinv_shoup_table[i] = new uint64_tt[N1];
        fillTablePsi_shoup128(ringPackPsiTable[i], pq_ringpack[i], ringPackPsiInvTable[i], psi_shoup_table[i], psiinv_shoup_table[i], N1);
    }

	cudaMalloc(&N1_psi_table_device, sizeof(uint64_tt) * N1 * ringpack_pq_count);
	cudaMalloc(&N1_psiinv_table_device, sizeof(uint64_tt) * N1 * ringpack_pq_count);
    
	cudaMalloc(&N1_psi_shoup_table_device, sizeof(uint64_tt) * N1 * ringpack_pq_count);
	cudaMalloc(&N1_psiinv_shoup_table_device, sizeof(uint64_tt) * N1 * ringpack_pq_count);
    for (int i = 0; i < ringpack_pq_count; i++)
	{
		cudaMemcpy(N1_psi_table_device + i * N1, ringPackPsiTable[i], sizeof(uint64_tt) * N1, cudaMemcpyHostToDevice);
		cudaMemcpy(N1_psiinv_table_device + i * N1, ringPackPsiInvTable[i], sizeof(uint64_tt) * N1, cudaMemcpyHostToDevice);

		cudaMemcpy(N1_psi_shoup_table_device + i * N1, psi_shoup_table[i], sizeof(uint64_tt) * N1, cudaMemcpyHostToDevice);
		cudaMemcpy(N1_psiinv_shoup_table_device + i * N1, psiinv_shoup_table[i], sizeof(uint64_tt) * N1, cudaMemcpyHostToDevice);

		delete ringPackPsiTable[i];
		delete ringPackPsiInvTable[i];

		delete psi_shoup_table[i];
		delete psiinv_shoup_table[i];
	}
	delete ringPackPsiTable;
	delete ringPackPsiInvTable;
    
	delete psi_shoup_table;
	delete psiinv_shoup_table;

    cudaMalloc(&N1_inv_device, sizeof(uint64_tt) * (ringpack_pq_count));
    cudaMalloc(&N1_inv_shoup_device, sizeof(uint64_tt) * (ringpack_pq_count));
    cudaMemcpy(N1_inv_device, N1_inv_host.data(), sizeof(uint64_tt) * ringpack_pq_count, cudaMemcpyHostToDevice);
    cudaMemcpy(N1_inv_shoup_device, N1_inv_shoup_host.data(), sizeof(uint64_tt) * ringpack_pq_count, cudaMemcpyHostToDevice);
	/*******************************************N1-NTT******************************************************/

    /***************************************ModDown pq -> q*************************************************/
    cudaMalloc(&p_inv_mod_qi, sizeof(uint64_tt) * p_inv_mod_qi_host.size());
    cudaMemcpy(p_inv_mod_qi, p_inv_mod_qi_host.data(), sizeof(uint64_tt) * (ringpack_q_count - 1), cudaMemcpyHostToDevice);
    cudaMalloc(&p_inv_mod_qi_shoup, sizeof(uint64_tt) * p_inv_mod_qi_host.size());
    cudaMemcpy(p_inv_mod_qi_shoup, p_inv_mod_qi_shoup_host.data(), sizeof(uint64_tt) * (ringpack_q_count - 1), cudaMemcpyHostToDevice);
}

#define MLWEEncodeCoeffs_block 128

__global__ void MLWEEncodeCoeffs_kernel(uint64_tt* data, double* vals, int N1, int mod_num, long long scaler, int p_num)
{
	uint32_tt idx_mod = blockIdx.y;
	register int local_tid = blockIdx.x * MLWEEncodeCoeffs_block + threadIdx.x; //num:N1

	register long mi = long(vals[local_tid]);

	register uint64_tt mod = pqt_cons[idx_mod + p_num];

	data[local_tid + idx_mod * N1] = mi >= 0 ? (uint64_tt) mi : (uint64_tt) (pqt_cons[p_num + idx_mod] + mi);
}

__host__ void PCMM_Context::encodeCoeffs(MLWEPlaintext& msg, double* vals)
{
    int l = msg.l;
    int p_num = context.p_num;

    dim3 encode_dim(msg.N1 / MLWEEncodeCoeffs_block, l+1);

    long long scaler = to_long(msg.scale);
    MLWEEncodeCoeffs_kernel <<< encode_dim, MLWEEncodeCoeffs_block >>> (msg.mx_device, vals, N1, l+1, scaler, p_num);
    // ToNTTInplace(msg.mx_device, 1, l+1, 0, ringpack_p_count, 1);
}

__global__ void MLWEDecodeCoeffs_kernel(uint64_tt* data, double* vals, int N1, int mod_num, long long scaler, int p_num)
{
	uint32_tt idx_mod = blockIdx.y;
	register int local_tid = blockIdx.x * MLWEEncodeCoeffs_block + threadIdx.x; //num:N1

	register long mi = long(data[local_tid]);

	register uint64_tt mod = pqt_cons[idx_mod + p_num];
    register uint64_tt mod2 = mod >> 1;

    vals[local_tid] = mi <= mod2 ? ((double) (mi) / scaler) : (((double) (mi) - (double) (mod)) / scaler);
}

__host__ void PCMM_Context::decodeCoeffs(MLWEPlaintext& msg, double* vals)
{
    int l = msg.l;
    int p_num = context.p_num;
    int N1 = msg.N1;

    dim3 decode_dim(N1 / MLWEEncodeCoeffs_block, 1);

    long long scaler = to_long(msg.scale);
	
	cudaMemcpy(context.decode_buffer_device, msg.mx_device, sizeof(uint64_tt) * N1 * (msg.L+1), cudaMemcpyDeviceToDevice);
    FromNTTInplace(context.decode_buffer_device, 1, l+1, 0, ringpack_p_count, 1);
    MLWEDecodeCoeffs_kernel <<< decode_dim, MLWEEncodeCoeffs_block >>> (context.decode_buffer_device, vals, N1, l+1, scaler, p_num);

	// cudaMemcpy(context.decode_buffer_host, context.decode_buffer_device, sizeof(uint64_tt) * N1 * (l+1), cudaMemcpyDeviceToHost);

	// vector<ZZ> coeffsBigint(N1);

    // std::vector<NTL::ZZ> crtReconstruction(l + 1);
    // uint64_tt tmp;
    // ZZ ModulusBigint = ZZ(1);
    // for (int i = 0; i < l + 1; i++) {
    //     ModulusBigint *= ZZ(q_ringpack[i]);
    // }

    // for (int i = 0; i < l + 1; i++) {
    //     crtReconstruction[i] = ModulusBigint / q_ringpack[i];
    //     tmp = crtReconstruction[i] % q_ringpack[i];
    //     tmp = modinv128(tmp, q_ringpack[i]);
    //     crtReconstruction[i] *= tmp;
    // }

    // for (int i = 0; i < N1; i++) {
    //     tmp = 0;
    //     for (int k = 0; k < l + 1; k++) {
    //         coeffsBigint[i] += ZZ(context.decode_buffer_host[k * N1 + i]) * crtReconstruction[k];
    //     }
    //     coeffsBigint[i] %= ModulusBigint;
    // }

    // NTL::ZZ Q_modules(1);
    // for (int i = 0; i < l+1; i++)
    // {
    //     Q_modules *= q_ringpack[i];
    // }
    // NTL::ZZ Q_half = Q_modules/2;
    // int sign;
    // ZZ c;
    // ZZ ci;
	// vector<double> v(N1);
    // for (long idx = 0; idx < N1; idx++)
	// {
    //     c = coeffsBigint[idx]<= Q_half ? coeffsBigint[idx] : coeffsBigint[idx] - Q_modules;
    //     v[idx] = to_double(NTL::conv<RR>(c)/msg.scale);
    // }
	// // cout<<endl<<endl;
	// cudaMemcpy(vals, v.data(), sizeof(double) * N1, cudaMemcpyHostToDevice);
}

// __host__ void PCMM_Context::ringUp(uint64_tt* smallRing_polys, uint64_tt* bigRing_polys, int mod_num)
// {
//     int N = context.N;
//     int p_num = context.p_num;

//     dim3 ringDown_dim(N / ringSwitch_block, mod_num);
//     ringUp_kernel <<< ringDown_dim, ringSwitch_block >>> (smallRing_polys, bigRing_polys, N, N1, mlwe_rank, p_num);
// }

// __host__ void PCMM_Context::ringDown(uint64_tt* bigRing_polys, uint64_tt* smallRing_polys, int mod_num)
// {
//     int N = context.N;
//     int p_num = context.p_num;

//     dim3 ringDown_dim(N / ringSwitch_block, mod_num);
//     ringDown_kernel <<< ringDown_dim, ringSwitch_block >>> (bigRing_polys, smallRing_polys, N, N1, mlwe_rank, p_num);
// }

// only for 2^8 ntt
void PCMM_Context::ToNTTInplace(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size)
{
    // 1 block -> 8 ntt
    // 8 warp -> 8 ntt
    // 8*32 thread -> 256 element
    int ntt_per_block = warp_number;
    int sm_count = ceil(1.0 * poly_num / ntt_per_block);
    // cout<<"poly_num: "<<poly_num<<endl;
    // cout<<"sm_count: "<<sm_count<<endl;
    dim3 ntt_dim(sm_count, mod_num, mod_batch_size);
    dim3 thread_dim(32, ntt_per_block);
    // cout<<"start_mod_idx: "<<start_mod_idx<<endl;

    if(N1 != 256) return;
    NTT256_kernel <<< ntt_dim, thread_dim, sizeof(uint64_tt) * ntt_per_block * 256, 0>>> (data, poly_num, mod_num, start_poly_idx, start_mod_idx, mod_batch_size, N1, N1_psi_table_device, N1_psi_shoup_table_device);
}


// only for 2^8 ntt
void PCMM_Context::FromNTTInplace(uint64_tt* data, int poly_num, int mod_num, int start_poly_idx, int start_mod_idx, int mod_batch_size)
{
    int ntt_per_block = warp_number;
    int sm_count = ceil(1.0 * poly_num / ntt_per_block);
    // cout<<"sm_count: "<<sm_count<<endl;
    dim3 ntt_dim(sm_count, mod_num, mod_batch_size);
    dim3 thread_dim(32, ntt_per_block);
    // cout<<"start_mod_idx: "<<start_mod_idx<<endl;

    if(N1 != 256) return;
    INTT256_kernel <<< ntt_dim, thread_dim, sizeof(uint64_tt) * ntt_per_block * 256, 0>>> (data, poly_num, mod_num, start_poly_idx, start_mod_idx, mod_batch_size, N1, N1_psiinv_table_device, N1_psiinv_shoup_table_device, N1_inv_device, N1_inv_shoup_device);
}