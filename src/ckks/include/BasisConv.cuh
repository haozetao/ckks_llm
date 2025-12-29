#pragma once

#include "Context_23.h"
#include "TimeUtils.cuh"

#define modUpRitoT_block 128
__global__
__launch_bounds__(
    modUpRitoT_block, 
    POLY_MIN_BLOCKS) 
void modUpPQtoT_23_kernel(uint64_tt* output, uint64_tt* input, uint32_tt n, int PQ_num, int t_num, int p_num, int q_num, int l, int gamma, int Ri_blockNum,
		uint64_tt* RiHatInvVecModRi_23_device, uint64_tt* RiHatInvVecModRi_23_shoup_device, uint64_tt* RiHatVecModT_23_device)
{
	register int idx_in_block = blockIdx.y;	
	register int idx_in_poly = blockIdx.z * modUpRitoT_block + threadIdx.x;
    // register int idx = idx_in_poly + blockIdx.y * t_num * n;

	register uint64_tt rb[max_gamma_num] = {0};
	register double vv = 0;

#pragma unroll
	for(int i = 0; i < gamma && i + idx_in_block*gamma < p_num + l + 1; i++)
	{
		register uint64_tt ri = pqt_cons[i + idx_in_block*gamma];
		register uint64_tt ra = input[(i + idx_in_block*gamma +  blockIdx.x * (p_num + q_num))*n + idx_in_poly];

		register uint64_tt RiHatInvVecModRi = RiHatInvVecModRi_23_device[i + idx_in_block*gamma];
		register uint64_tt RiHatInvVecModRi_shoup = RiHatInvVecModRi_23_shoup_device[i + idx_in_block*gamma];
		rb[i] = mulMod_shoup(ra, RiHatInvVecModRi, RiHatInvVecModRi_shoup, ri);

		vv += double(rb[i]) / ri;
	}

	register int vv_floor = (int)vv;

#pragma unroll
	for(int idx_in_T = 0; idx_in_T < t_num; idx_in_T++)
	{
		register uint64_tt t = pqt_cons[p_num + q_num + idx_in_T];
		register uint128_tt mu_t(pqt_mu_cons_high[p_num + q_num + idx_in_T], pqt_mu_cons_low[p_num + q_num + idx_in_T]);

		register uint128_tt acc = 0;

	#pragma unroll
		for(int j = 0; j < gamma && j + idx_in_block*gamma < p_num + l + 1; j++)
		{
			register uint64_tt RiHatVecModT = RiHatVecModT_23_device[idx_in_block*t_num*gamma + idx_in_T*gamma + j];

			madc_uint64_uint64_uint128(rb[j], RiHatVecModT, acc);
		}

		register int idx_in_output = blockIdx.x * Ri_blockNum * t_num + idx_in_block*t_num + idx_in_T;

		sub_uint128_uint64(acc, vv_floor * Rimodti_cons[idx_in_T * Ri_blockNum + idx_in_block], acc);

		singleBarrett_new(acc, t, mu_t);
		output[idx_in_output*n + idx_in_poly] = acc.low;
	}
}

void Context_23::modUpPQtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size)
{
	// // print_device_array(input, N, K+L+1, "ax_Ri");
	// int blockNum = ceil(double(p_num + l+1) / gamma);
	// dim3 modUp_dim(N / modUpRitoT_block, blockNum, t_num);
	// modUpPQtoT_23_kernel <<< modUp_dim, modUpRitoT_block >>>
	// 	(output, input, N, p_num+q_num, t_num, p_num, l, gamma, Ri_blockNum, 
	// 	RiHatInvVecModRi_23_device, RiHatInvVecModRi_23_shoup_device, RiHatVecModT_23_device, batch_size);
	// // print_device_array(output, N, t_num*Ri_blockNum, "ax_T");

	// print_device_array(input, N, K+L+1, "ax_Ri");
	int blockNum = ceil(double(p_num + l+1) / gamma);
	// dim3 modUp_dim(N / modUpRitoT_block, blockNum, batch_size);
	dim3 modUp_dim(batch_size, blockNum, N / modUpRitoT_block);
	modUpPQtoT_23_kernel <<< modUp_dim, modUpRitoT_block >>>
		(output, input, N, p_num+q_num, t_num, p_num, q_num, l, gamma, Ri_blockNum, 
		RiHatInvVecModRi_23_device, RiHatInvVecModRi_23_shoup_device, RiHatVecModT_23_device);
	// print_device_array(output, N, t_num*Ri_blockNum, "ax_T");
}


#define ebconv_padding_size 1
#define zeroPadding_modUpQjtoT_block 64
__global__
__launch_bounds__(
    zeroPadding_modUpQjtoT_block, 
    POLY_MIN_BLOCKS) 
void modUpQjtoT_23_kernel(uint64_tt* output, uint64_tt* input, uint32_tt n, int p_num, int q_num, int t_num, int l, int gamma, int Qj_blockNum,
			uint64_tt* QjHatInvVecModQj_23_device, uint64_tt* QjHatInvVecModQj_23_shoup_device, uint64_tt* QjHatVecModT_23_device, uint64_tt* Qjmodti_device)
{
	register int idx_in_poly = blockIdx.z * zeroPadding_modUpQjtoT_block + threadIdx.x;
	register int idx_in_block = blockIdx.y;

	register uint64_tt rb[ebconv_padding_size][max_p_num] = {0}; // sizeof(rb) must > gamma
	register double vv[ebconv_padding_size] = {0};

#pragma unroll
	for(int j = 0; j < p_num && j + idx_in_block*p_num < l+1; j++)
	{
		register int idx_in_pq = j + idx_in_block*p_num;
		register uint64_tt qi = pqt_cons[p_num + idx_in_pq];
		// register uint64_tt ra = input[blockIdx.x*q_num*n + idx_in_pq*n + idx_in_poly];

		register uint64_tt QjHatInvVecModQj = QjHatInvVecModQj_23_device[idx_in_pq];
		register uint64_tt QjHatInvVecModQj_shoup = QjHatInvVecModQj_23_shoup_device[idx_in_pq];
		// rb[j] = mulMod_shoup(ra, QjHatInvVecModQj, QjHatInvVecModQj_shoup, qi);

		#pragma unroll
		for(int iter = 0; iter < ebconv_padding_size; iter++){
			register uint64_tt ra = input[blockIdx.x*q_num*n + idx_in_pq*n + idx_in_poly + iter*gridDim.z*blockDim.x];
			rb[iter][j] = mulMod_shoup(ra, QjHatInvVecModQj, QjHatInvVecModQj_shoup, qi);
			
			double rr = (double)(rb[iter][j]);
			vv[iter] += rr / (double)(qi);
		}
		// double rr = (double)(rb[j]);
		// vv += rr / (double)(qi);
	}

	// register int vv_floor = (int)vv;

#pragma unroll
	for(int i = 0; i < t_num; i++)
	{
		register uint64_tt t = pqt_cons[p_num + q_num + i];
		register uint128_tt mu_t(pqt_mu_cons_high[p_num + q_num + i], pqt_mu_cons_low[p_num + q_num + i]);

		register uint128_tt acc[ebconv_padding_size] = 0;

	#pragma unroll
		for(int j = 0; j < p_num && j + idx_in_block*p_num < l+1; j++)
		{
			register uint64_tt QjHatVecModT = QjHatVecModT_23_device[idx_in_block*t_num*p_num + i*p_num + j];

			// madc_uint64_uint64_uint128(rb[j], QjHatVecModT, acc);
			#pragma unroll
			for(int iter = 0; iter < ebconv_padding_size; iter++){
				madc_uint64_uint64_uint128(rb[iter][j], QjHatVecModT, acc[iter]);
			}
		}

		register int idx_in_output = blockIdx.x*Qj_blockNum*t_num + idx_in_block*t_num + i;

		// sub_uint128_uint64(acc, vv_floor * Qjmodti_device[idx_in_output], acc);


		// singleBarrett_new(acc, t, mu_t);
		// output[idx_in_output*n + idx_in_poly] = acc.low;

		#pragma unroll
		for(int iter = 0; iter < ebconv_padding_size; iter++){
			register int vv_floor = (int)vv[iter];
			sub_uint128_uint64(acc[iter], vv_floor * Qjmodti_device[idx_in_output], acc[iter]);
			
			singleBarrett_new(acc[iter], t, mu_t);
			output[idx_in_output*n + idx_in_poly + iter*gridDim.z*blockDim.x] = acc[iter].low;
		}
	}
}

__host__ void Context_23::modUpQjtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size)
{
	int blockNum = ceil(double(l+1) / p_num);

	// print_device_array(input, N, l+1, "ax_Qj");
	// dim3 zeroPadding_modUpQjtoT_dim(N / zeroPadding_modUpQjtoT_block , blockNum, batch_size);
	// CUDATimer timer;
	dim3 zeroPadding_modUpQjtoT_dim(batch_size, blockNum, N / zeroPadding_modUpQjtoT_block / ebconv_padding_size);
	// timer.start();
	modUpQjtoT_23_kernel <<< zeroPadding_modUpQjtoT_dim, zeroPadding_modUpQjtoT_block, (t_num*Qj_blockNum + t_num*p_num)*sizeof(uint64_tt), 0 >>>
		(output, input, N, p_num, q_num, t_num, l, gamma, Qj_blockNum, 
		QjHatInvVecModQj_23_device + l*Qj_blockNum*p_num,
		QjHatInvVecModQj_23_shoup_device + l*Qj_blockNum*p_num,
		QjHatVecModT_23_device + l*Qj_blockNum*t_num*p_num,
		Qjmodti_device + l*t_num*Qj_blockNum);
		
	// cout<<"modUpQjtoT blockNum: "<<N / zeroPadding_modUpQjtoT_block / ebconv_padding_size<<" time: "<<timer.stop()*1000<<" us"<<endl;
	// print_device_array(output, N, blockNum*t_num, "ax_T");
}


#define modUpTtoQj_block 128
__global__
__launch_bounds__(
    modUpTtoQj_block, 
    POLY_MIN_BLOCKS) 
void modUpTtoPQl_23_kernel(uint64_tt* modUp_TtoQj_buffer, uint64_tt* exProduct_T_temp, int n, int p_num, int q_num, int t_num, int l,
 	int gamma, int Ri_blockNum, uint64_tt* THatInvVecModti_23_device, uint64_tt* THatInvVecModti_23_shoup_device, uint64_tt* THatVecModRi_23_device)
{
	register int idx_in_poly = blockIdx.z * modUpTtoQj_block + threadIdx.x;
	register int idx_in_block = blockIdx.y;
	register int idx_in_cipher = blockIdx.x;

	register uint64_tt rb[max_tnum]; // sizeof(rb) must > t_num
	register double vv = 0;

#pragma unroll
	for(int i = 0; i < t_num; i++)
	{
		register uint64_tt t = pqt_cons[p_num + q_num + i];
		register uint64_tt ra = exProduct_T_temp[idx_in_cipher*Ri_blockNum*t_num*n + (idx_in_block*t_num + i)*n + idx_in_poly];

		ra = ra + halfTmodpqti_cons[p_num + q_num + i];

		register uint64_tt THatInvVecModti = THatInvVecModti_23_device[i];
		register uint64_tt THatInvVecModti_shoup = THatInvVecModti_23_shoup_device[i];
		rb[i] = mulMod_shoup(ra, THatInvVecModti, THatInvVecModti_shoup, t);

		double rr = (double)(rb[i]);
		vv += rr / (double)(t);
	}

	register int vv_floor = (int)vv;

#pragma unroll
	for(int j = 0; j < gamma && j + idx_in_block*gamma < p_num+l+1; j++)
	{
		register int idx_in_pq = j + idx_in_block*gamma;
		register uint64_tt ri = pqt_cons[idx_in_pq];
		register uint128_tt mu_ri(pqt_mu_cons_high[idx_in_pq], pqt_mu_cons_low[idx_in_pq]);

		register uint128_tt acc = 0;
	
	#pragma unroll
		for(int i = 0; i < t_num; i++)
		{
			register uint64_tt THatVecModRi = THatVecModRi_23_device[idx_in_pq*t_num + i];

			madc_uint64_uint64_uint128(rb[i], THatVecModRi, acc);
		}

		sub_uint128_uint64(acc, vv_floor*Tmodpqi_cons[idx_in_pq], acc);

		singleBarrett_new(acc, ri, mu_ri);

		acc.low = acc.low + ri - halfTmodpqti_cons[idx_in_pq];
		csub_q(acc.low, ri);

		modUp_TtoQj_buffer[idx_in_cipher*Ri_blockNum*t_num*n + idx_in_pq * n + idx_in_poly] = acc.low;
	}
}

__host__ void Context_23::modUpTtoPQl_23(uint64_tt* modUp_TtoQj_buffer, uint64_tt* exProduct_T_temp, int l, int batch_size)
{
	// print_device_array(exProduct_T_temp, N, 2*t_num*Ri_blockNum, "ext_T");
	int blockNum = ceil(double(K+l+1) / gamma);
	// int iter_num = 2;
	// dim3 modUpTtoRi_23_dim(N / modUpTtoQj_block, blockNum, batch_size);
	dim3 modUpTtoRi_23_dim(batch_size, blockNum, N / modUpTtoQj_block);
	modUpTtoPQl_23_kernel <<< modUpTtoRi_23_dim, modUpTtoQj_block, (t_num*gamma +t_num)*sizeof(uint64_tt), 0 >>>
		(modUp_TtoQj_buffer, exProduct_T_temp, N, p_num, q_num, t_num, l, gamma, Ri_blockNum, THatInvVecModti_23_device, THatInvVecModti_23_shoup_device, THatVecModRi_23_device);

	// print_device_array(modUp_TtoQj_buffer, N, 2*t_num*Ri_blockNum, "ext_Ri");
}


__global__
__launch_bounds__(
    modUpTtoQj_block, 
    POLY_MIN_BLOCKS) 
void modDownPQltoQl_23_alpha1_multi_kernel(uint64_tt* output, uint64_tt* input, uint32_tt n, int p_num, int q_num, int t_num, int l, int Ri_blockNum,
	uint64_tt* pHatVecModq_23_device)
{
	register int idx_in_poly = blockIdx.z * modUpTtoQj_block + threadIdx.x;
	register int idx_in_pq = blockIdx.y;
	register int idx_in_cipher = blockIdx.x;

	register uint64_tt qi = pqt_cons[p_num + idx_in_pq];
	// register uint64_tt mu_qi_hi = pqt_mu_cons_high[p_num + idx_in_pq];
	register uint64_tt pi = pqt_cons[0];
	register uint64_tt ra = input[idx_in_cipher*Ri_blockNum*t_num*n + idx_in_poly];

	// barrett_reduce_uint64_uint64(ra, qi, mu_qi_hi);
	ra %= qi;

	register uint64_tt Pinvmodqi = Pinvmodqi_cons[idx_in_pq];
	register uint64_tt Pinvmodqi_shoup = Pinvmodqi_shoup_cons[idx_in_pq];
	ra = qi - ra + input[idx_in_cipher*Ri_blockNum*t_num*n + (p_num+idx_in_pq) * n + idx_in_poly];
	output[idx_in_cipher*q_num*n + idx_in_pq*n + idx_in_poly] = mulMod_shoup(ra, Pinvmodqi, Pinvmodqi_shoup, qi);
}

__global__
__launch_bounds__(
    modUpTtoQj_block, 
    POLY_MIN_BLOCKS)
__global__ void modDownPQltoQl_23_multi_kernel(uint64_tt* output, uint64_tt* input, uint32_tt n, int p_num, int q_num, int t_num, int l, int Ri_blockNum,
	uint64_tt* pHatVecModq_23_device)
{
	register int idx_in_poly = blockIdx.z * modUpTtoQj_block + threadIdx.x;
	register int idx_in_pq = blockIdx.y;
	register int idx_in_cipher = blockIdx.x;

	register uint64_tt qi = pqt_cons[p_num + idx_in_pq];
	register uint128_tt mu_qi(pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]);

	register uint128_tt acc;

	// register double vv = 0;

#pragma unroll
	for(int i = 0; i < p_num; i++)
	{
		register uint64_tt pi = pqt_cons[i];
		register uint64_tt ra = input[idx_in_cipher*Ri_blockNum*t_num*n + i * n + idx_in_poly];

		register uint64_tt pHatInvVecModp = pHatInvVecModp_cons[i];
		register uint64_tt pHatInvVecModp_shoup = pHatInvVecModp_shoup_cons[i];

		register uint64_tt rb = mulMod_shoup(ra, pHatInvVecModp, pHatInvVecModp_shoup, pi);

		// vv += double(rb) / pi;

		register uint64_tt pHatVecModq = pHatVecModq_23_device[idx_in_pq*p_num + i];

		madc_uint64_uint64_uint128(rb, pHatVecModq, acc);
	}
	// register int vv_floor = int(vv);
	// sub_uint128_uint64(acc, vv_floor * Pmodqt_cons[idx_in_pq], acc);

	singleBarrett_new(acc, qi, mu_qi);

	register uint64_tt Pinvmodqi = Pinvmodqi_cons[idx_in_pq];
	register uint64_tt Pinvmodqi_shoup = Pinvmodqi_shoup_cons[idx_in_pq];
	acc.low = qi - acc.low + input[idx_in_cipher*Ri_blockNum*t_num*n + (p_num+idx_in_pq) * n + idx_in_poly];
	output[idx_in_cipher*q_num*n + idx_in_pq*n + idx_in_poly] = mulMod_shoup(acc.low, Pinvmodqi, Pinvmodqi_shoup, qi);
}

__host__ void Context_23::modDownPQltoQl_23(uint64_tt* output, uint64_tt* modUp_TtoQj_buffer, int l, int batch_size)
{
	// print_device_array(modUp_TtoQj_buffer, N, 2*t_num*Ri_blockNum, "ext_Ri");

	// dim3 modDownRitoQj_23_dim(N / modUpTtoQj_block, l+1, batch_size);
	dim3 modDownRitoQj_23_dim(batch_size, l+1, N / modUpTtoQj_block);
	if(p_num == 1)
	{
		modDownPQltoQl_23_alpha1_multi_kernel <<< modDownRitoQj_23_dim, modUpTtoQj_block, p_num*sizeof(uint64_tt)>>>
			(output, modUp_TtoQj_buffer, N, p_num, q_num, t_num, l, Ri_blockNum,
			pHatVecModq_23_device);
	}
	else
	{
		modDownPQltoQl_23_multi_kernel <<< modDownRitoQj_23_dim, modUpTtoQj_block, p_num*sizeof(uint64_tt)>>>
			(output, modUp_TtoQj_buffer, N, p_num, q_num, t_num, l, Ri_blockNum,
			pHatVecModq_23_device);
	}
	// print_device_array(output, N, 2*(L+1), "ext_l");
}