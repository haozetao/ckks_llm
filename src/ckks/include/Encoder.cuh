#pragma once

#include <NTL/tools.h>

#include "Context_23.h"
#include "cuda.h"

__global__ void encode_kernel(uint64_tt* out, cuDoubleComplex* vals , long mod_num, long N, int p_num, long gap, int thread)
{
	uint32_tt index = blockIdx.y % mod_num;
	register uint64_tt* mi  = out + index * N;
	register int local_tid = blockIdx.x * thread + threadIdx.x; //num:slots

	register long mir = long(vals[local_tid].x);
	register long mii = long(vals[local_tid].y);

	register uint64_tt mod = pqt_cons[index + p_num];

	mi[local_tid * gap]           = mir >= 0 ? (uint64_tt) mir : (uint64_tt) (pqt_cons[p_num + index] + mir);
	mi[local_tid * gap + (N / 2)] = mii >= 0 ? (uint64_tt) mii : (uint64_tt) (pqt_cons[p_num + index] + mii);
}

__global__ void decode_kernel(uint64_tt* in, cuDoubleComplex* vals, long N, int p_num, long gap, long precision, int thread)
{
	uint64_tt pr = pqt_cons[p_num];
	uint64_tt pr_2 = pqt_cons[p_num] >> 1;
	register int local_tid = blockIdx.x * thread + threadIdx.x; //num:slots

	register double mir = in[local_tid * gap]           <= pr_2 ? 
							((double) (in[local_tid * gap]) / precision) : 
							(((double) (in[local_tid * gap]) - (double) (pr)) / precision);

	register double mii = in[(N / 2) + local_tid * gap] <= pr_2 ? 
							((double) (in[(N / 2) + local_tid * gap]) / precision) : 
							(((double) (in[(N / 2) + local_tid * gap]) - (double) (pr)) / precision);

	vals[local_tid].x = mir;
	vals[local_tid].y = mii;
}

__host__ void Context_23::encode(cuDoubleComplex* vals, Plaintext& msg)
{
	long mes_slots = msg.slots;
	int gap = Nh / mes_slots;
	int level = msg.l;
	
	long scale = to_long((msg.scale));

	cudaMemcpy(encode_buffer, vals, sizeof(cuDoubleComplex)*mes_slots, cudaMemcpyDeviceToDevice);
	fftInv_batch(encode_buffer, ksiPows_device, mes_slots, scale, rotGroups_device, 2*N);
	int block = 1;
	int thread = mes_slots;
	if(mes_slots > 1024){
		block = mes_slots / 1024;
		thread = 1024;
	}

	bitReverse_device(encode_buffer, mes_slots, block, thread);

	dim3 encode_dim(block , level+1);
	encode_kernel <<<encode_dim, thread, 0, 0 >>> (msg.mx_device, encode_buffer, level+1, N, p_num, gap, thread);
	
	// print_device_array(msg.mx_device, N, L+1, "encode");
	ToNTTInplace(msg.mx_device,0, K, 1, level+1, L+1);
}

#define encode_coeffs_dim 1024
__global__ void encode_coeffs_kernel(uint64_tt* out, double* vals, int N, int p_num, long precision)
{
	uint32_tt index = blockIdx.y;
	register uint64_tt* mi = out + index * N;
	register int thread_idx = blockIdx.x * encode_coeffs_dim + threadIdx.x;

	register long data = long(vals[thread_idx] * precision);

	register uint64_tt mod = pqt_cons[index + p_num];
	mi[thread_idx] = data >= 0 ? (uint64_tt) data : (uint64_tt) (mod + data);
}

__host__ void Context_23::encode_coeffs(double* vals, Plaintext& msg)
{
	long mes_slots = msg.slots;
	int gap = Nh / mes_slots;
	int level = msg.l;
	long scale = to_long(msg.scale);

	cudaMemcpy(encode_coeffs_buffer, vals, sizeof(cuDoubleComplex)*mes_slots, cudaMemcpyDeviceToDevice);

	dim3 encode_dim(N / encode_coeffs_dim, level+1);
	encode_coeffs_kernel <<<encode_dim, encode_coeffs_dim >>> (msg.mx_device, encode_coeffs_buffer, N, p_num, scale);

	print_device_array(msg.mx_device, N, level+1, "encode coeff mx");

	ToNTTInplace(msg.mx_device,0, K, 1, level+1, L+1);
}

__global__ void decode_coeffs_kernel(uint64_tt* in, double* vals, int p_num, long scale)
{
	uint64_tt mod = pqt_cons[p_num];
	uint64_tt mod_2 = pqt_cons[p_num] >> 1;
	register int thread_idx = blockIdx.x * encode_coeffs_dim + threadIdx.x;

	register double data = in[thread_idx] <= mod_2 ? 
							((double) (in[thread_idx]) / scale) : 
							(((double) (in[thread_idx]) - (double) (mod)) / scale);
	vals[thread_idx] = data;
}


using ZZ = NTL::ZZ;
using RR = NTL::RR;
__host__ void Context_23::decode_coeffs(Plaintext& msg, double* vals, bool is_bitRev)
{
	long slots = msg.slots;
	int gap = Nh / slots;
	int level = msg.l;
	int L = msg.L;
	long scale = to_long(msg.scale);

	cudaMemcpy(decode_buffer_device, msg.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);
	FromNTTInplace(decode_buffer_device, 0, K, 1, level+1, L+1);

	// cudaMemcpy(decode_buffer_device, msg.mx_device, sizeof(uint64_tt) * N * (level+1), cudaMemcpyDeviceToDevice);
	// dim3 decode_dim(N / encode_coeffs_dim);
	// decode_coeffs_kernel <<< decode_dim, encode_coeffs_dim >>> (decode_buffer_device, vals, p_num, scale);

	cudaMemcpy(decode_buffer_host, decode_buffer_device, sizeof(uint64_tt) * N * (level+1), cudaMemcpyDeviceToHost);

	vector<ZZ> coeffsBigint(N);

    std::vector<NTL::ZZ> crtReconstruction(level + 1);
    uint64_tt tmp;
    ZZ ModulusBigint = ZZ(1);
    for (int i = 0; i < level + 1; i++) {
        ModulusBigint *= ZZ(qVec[i]);
    }

    for (int i = 0; i < level + 1; i++) {
        crtReconstruction[i] = ModulusBigint / qVec[i];
        tmp = crtReconstruction[i] % qVec[i];
        tmp = modinv128(tmp, qVec[i]);
        crtReconstruction[i] *= tmp;
    }

    for (int i = 0; i < N; i++) {
        tmp = 0;
        for (int k = 0; k < level + 1; k++) {
            coeffsBigint[i] += ZZ(decode_buffer_host[k * N + i]) * crtReconstruction[k];
        }
        coeffsBigint[i] %= ModulusBigint;
    }

    NTL::ZZ Q_modules(1);
    for (int i = 0; i < level+1; i++)
    {
        Q_modules *= qVec[i];
    }
    NTL::ZZ Q_half = Q_modules/2;
    int sign;
    ZZ c;
    ZZ ci;
	vector<double> v(N);
    // for (long idx = 0; idx < N; idx++)
	// {
    //     c = coeffsBigint[idx]<= Q_half ? coeffsBigint[idx] : coeffsBigint[idx] - Q_modules;
    //     v[idx] = to_double(NTL::conv<RR>(c)/msg.scale);
    // }
	for (long j = 0, jdx = Nh, idx = 0; j < slots; j++, jdx += gap, idx += gap)
	{
        c = coeffsBigint[idx]<= Q_half ? coeffsBigint[idx] : coeffsBigint[idx] -  Q_modules;
        ci = coeffsBigint[jdx]<= Q_half ? coeffsBigint[jdx] : coeffsBigint[jdx] -  Q_modules;

		if(is_bitRev){
			v[bitReverse(idx, log2(slots))] = to_double(NTL::conv<RR>(c)/msg.scale);
			v[bitReverse(idx, log2(slots)) + Nh] = to_double(NTL::conv<RR>(ci)/msg.scale);
		} else 
		{
			v[idx] = to_double(NTL::conv<RR>(c)/msg.scale);
			v[jdx] = to_double(NTL::conv<RR>(ci)/msg.scale);
		}
    }
	if(is_bitRev){
		cout<<"decode coeffs with bitrev"<<endl;
	}
	cudaMemcpy(vals, v.data(), sizeof(double) * N, cudaMemcpyHostToDevice);
}

void Context_23::PolyToBigintLvl(int l, uint64_tt* p1, int gap, std::vector<ZZ>& coeffsBigint) {

    std::vector<NTL::ZZ> crtReconstruction(l + 1);
    uint64_tt tmp;
    ZZ ModulusBigint = ZZ(1);
    for (int i = 0; i < l + 1; i++) {
        ModulusBigint *= ZZ(qVec[i]);
    }

    for (int i = 0; i < l + 1; i++) {
        crtReconstruction[i] = ModulusBigint / qVec[i];
        tmp = crtReconstruction[i] % qVec[i];
        tmp = modinv128(tmp, qVec[i]);
        crtReconstruction[i] *= tmp;
    }

    for (int i = 0, j = 0; j < N; i++, j += gap) {
        tmp = 0;

        for (int k = 0; k < l + 1; k++) {
            coeffsBigint[i] += ZZ(p1[k * N + j]) * crtReconstruction[k];
        }
        coeffsBigint[i] %= ModulusBigint;
    }
}

__host__ void Context_23::decode(Plaintext& msg, cuDoubleComplex* vals)
{
	long slots = msg.slots;
	int gap = Nh / slots;
	int l = msg.l;
	int L = msg.L;

	cudaMemcpy(decode_buffer_device, msg.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);
	FromNTTInplace(decode_buffer_device, 0, K, 1, l+1, L+1);
	// print_device_array(decode_buffer_device, N, L+1, "decode_buffer");	
	int block = 1;
	int thread = slots;
	if(slots > 1024){
		block = slots / 1024;
		thread = 1024;
	}
	
	// dim3 decode_dim(block);
	// decode_kernel <<<decode_dim, thread, 0, 0 >>> (decode_buffer_device, vals, N, p_num, gap, to_double(msg.scale), thread);

	cudaMemcpy(decode_buffer_host, decode_buffer_device, sizeof(uint64_tt) * N * (l+1), cudaMemcpyDeviceToHost);

	vector<ZZ> coeffsBigint(N);
    PolyToBigintLvl(l, decode_buffer_host, gap, coeffsBigint);//CRT

    NTL::ZZ Q_modules(1);
    for (int i = 0; i < l+1; i++)
    {
        Q_modules *= qVec[i];
    }
    NTL::ZZ Q_half = Q_modules/2;
    int sign;
    ZZ c;
    ZZ ci;
	vector<cuDoubleComplex> v(slots);
	// cout<<endl<<endl;
    for (long j = 0, jdx = Nh, idx = 0; j < slots; j++, jdx += gap, idx += gap)
	{
        c = coeffsBigint[idx]<= Q_half ? coeffsBigint[idx] : coeffsBigint[idx] -  Q_modules;
        ci = coeffsBigint[jdx]<= Q_half ? coeffsBigint[jdx] : coeffsBigint[jdx] -  Q_modules;

		// cout<<c<<", "<<ci<<", ";

        v[j].x = to_double(NTL::conv<RR>(c)/msg.scale);
        v[j].y = to_double(NTL::conv<RR>(ci)/msg.scale);

		// if(j < 4) cout<<c<<", "<<ci<<endl;
    }
	// cout<<endl<<endl;
	cudaMemcpy(vals, v.data(), sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);

	bitReverse_device(vals , slots, block, thread);
	fft_batch(vals, ksiPows_device, slots, rotGroups_device, 2*N);
}

__global__ void encode_kernel_T(uint64_tt* out, cuDoubleComplex* vals , long N, int p_num, int q_num, long gap, int thread)
{
	uint32_tt idx_in_mod = blockIdx.y;
	register uint64_tt *mi  = out + idx_in_mod * N;
	register int idx_in_slots = blockIdx.x * thread + threadIdx.x; //num:slots

	register long mir = long(vals[idx_in_slots].x);
	register long mii = long(vals[idx_in_slots].y);

	register uint64_tt mod = pqt_cons[p_num+q_num + idx_in_mod];
	mi[idx_in_slots * gap]           = mir >= 0 ? (uint64_tt) mir : (uint64_tt) (mod + mir);
	mi[idx_in_slots * gap + (N / 2)] = mii >= 0 ? (uint64_tt) mii : (uint64_tt) (mod + mii);

	// mi[idx_in_slots * gap]           = (mi[idx_in_slots * gap]           + halfTmodpqti_cons[p_num + q_num + idx_in_mod]) % mod;
	// mi[idx_in_slots * gap + (N / 2)] = (mi[idx_in_slots * gap + (N / 2)] + halfTmodpqti_cons[p_num + q_num + idx_in_mod]) % mod;
}

__host__ void Context_23::encode_T(cuDoubleComplex* vals, PlaintextT& msg_T, NTL::RR scale)
{
	int gap = Nh / slots;
	
	cudaMemcpy(encode_buffer, vals, sizeof(cuDoubleComplex)*slots, cudaMemcpyDeviceToDevice);
	
	fftInv_batch(encode_buffer, ksiPows_device, slots, to_double(scale), rotGroups_device, 2*N);
	//print_device_array(encode_buffer, slots, "fft");

	int block = 1;
	int thread = slots;
	if(slots > 1024){
		block = slots / 1024;
		thread = 1024;
	}

	bitReverse_device(encode_buffer, slots, block, thread);

	dim3 encode_dim(block, t_num);
	encode_kernel_T <<<encode_dim, thread, 0, 0 >>> (msg_T.mx_device, encode_buffer, N, p_num, q_num, gap, thread);
	
	//print_device_array(msg_PQl, N, K+L+1, "encode");
	ToNTTInplace(msg_T.mx_device, 0, K+L+1, 1, t_num, t_num);
}