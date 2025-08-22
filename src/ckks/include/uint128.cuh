#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <cinttypes>
#include <string>
#include <math.h>
#include <NTL/RR.h>
#include <NTL/ZZ.h>

typedef unsigned char uint8_tt;
typedef unsigned int uint32_tt;
typedef unsigned long long uint64_tt;
typedef long long int64_tt;


#define check_mod_operation 1

class uint128_tt
{
public:
	
	uint64_tt low;
	uint64_tt high;

	__host__ __device__ __forceinline__ uint128_tt(uint64_tt high, uint64_tt low) : high(high), low(low)
	{}

	__host__ __device__ __forceinline__ uint128_tt()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_tt(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_tt& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}
	
	__host__ __device__ __forceinline__ uint128_tt operator<<(const int& shift)
	{
		uint128_tt z;
		if (shift < 64) {
			z.high = (high << shift) | (low >> (64 - shift));
			z.low = low << shift;
			return z;
		} else if (shift == 64){
			z.high = low;
			z.low = 0;
			return z;
		} else {
			z.high = low << (shift - 64);
			z.low = 0;
			return z;
		}
	}

	// __host__ __device__ __forceinline__ uint128_tt operator>>(const int& shift)
	// {
	// 	uint128_tt z;

	// 	z.low = low >> shift;
	// 	z.low = (high << (64 - shift)) | z.low;
	// 	z.high = high >> shift;

	// 	return z;
	// }
};

// __host__ __device__ __forceinline__ uint128_tt operator+(const uint128_tt& x, const uint128_tt& y)
// {
// 	uint128_tt z;

// 	z.low = x.low + y.low;
// 	z.high = x.high + y.high + (z.low < x.low);

// 	return z;
// }

// __host__ __device__ __forceinline__ uint128_tt operator+(const uint128_tt& x, const uint64_t& y)
// {
// 	uint128_tt z;

// 	z.low = x.low + y;
// 	z.high = x.high + (z.low < x.low);

// 	return z;
// }

// __host__ __device__ __forceinline__ uint128_tt operator-(const uint128_tt& x, const uint128_tt& y)
// {
// 	uint128_tt z;

// 	z.low = x.low - y.low;
// 	z.high = x.high - y.high - (x.low < y.low);

// 	return z;
	
// }

// __host__ __device__ __forceinline__ void operator-=(uint128_tt& x, const uint128_tt& y)
// {
// 	x.high = x.high - y.high - (x.low < y.low);
// 	x.low = x.low - y.low;
// }

// __host__ __device__ __forceinline__ uint128_tt operator-(const uint128_tt& x, const uint64_t& y)
// {
// 	uint128_tt z;

// 	z.low = x.low - y;
// 	z.high = x.high - (x.low < y);

// 	return z;
// }

// c = a * b
__device__ __forceinline__ void mul64(const uint64_tt& a, const uint64_tt& b, uint128_tt& c)
{
	asm("{\n\t"
		"mad.lo.cc.u64		%1, %2, %3, 0;		\n\t"
		"madc.hi.u64		%0, %2, %3, 0;		\n\t"
		"}"
		: "=l"(c.high), "=l"(c.low)
		: "l"(a), "l"(b));
}

__forceinline__ __device__ void csub_q(uint64_tt& a, uint64_tt q) 
{
	register uint64_tt tmp = a - q;
	a = tmp + (tmp >> 63) * q;
}

// a \in [0, 2q)
// output a \in [0, q)
// __forceinline__ __device__ void csub_q(uint64_tt& a, uint64_tt q) 
// {
// 	register uint64_tt tmp1 = 0, tmp2 = 0;

// 	asm("{\n\t"
// 	"sub.cc.u64			%0, %2, %3;		\n\t"
// 	"addc.cc.u64		%1,	0,	0;		\n\t"
// 	"}"
// 	: "=l"(tmp1), "=l"(tmp2)
// 	: "l"(a), "l"(q));
// 	a = tmp1 + (tmp2^1)*q;

// #if check_mod_operation
// 	if(a > q) printf("error csub_q\n");
// #endif
// }

// c = a * b \in [0, q)
__device__ __forceinline__ uint64_tt mulMod_shoup(const uint64_tt& a, const uint64_tt& b, const uint64_tt& b_shoup, uint64_tt& mod)
{
	uint64_tt hi = __umul64hi(a, b_shoup);
	uint64_tt ra = a * b - hi * mod;
	csub_q(ra, mod);

#if check_mod_operation
	if(ra > mod) printf("error mulMod_shoup\n");
#endif
	return ra;
}

// from phantom-fhe
__device__ __forceinline__ void singleBarrett_new(uint128_tt& a, uint64_tt& q, uint128_tt& mu)
{
	uint64_tt result;
	asm volatile(
		"{\n\t"
		" .reg .u64 	tmp;\n\t"
		// Multiply input and const_ratio
		// Round 1
		" mul.hi.u64 	tmp, %1, %3;\n\t"
		" mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
		" madc.hi.u64 	%0, %1, %4, 0;\n\t"
		// Round 2
		" mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
		" madc.hi.u64 	%0, %2, %3, %0;\n\t"
		// This is all we care about
		" mad.lo.u64 	%0, %2, %4, %0;\n\t"
		// Barrett subtraction
		" mul.lo.u64 	%0, %0, %5;\n\t"
		" sub.u64 		%0, %1, %0;\n\t"
		"}"
		: "=l"(result)
		: "l"(a.low), "l"(a.high), "l"(mu.low), "l"(mu.high), "l"(q));

	csub_q(result, q);
	a.high = 0;
	a.low = result;

#if check_mod_operation
	if(a.low > q) printf("error singleBarrett_new\n");
#endif
}

__forceinline__ __device__ void barrett_reduce_uint64_uint64(uint64_tt& a,
															 uint64_tt& q,
															 const uint64_tt& mu_hi)
{
	uint64_tt s = __umul64hi(mu_hi, a);
	a = a - s * q;
	csub_q(a, q);

#if check_mod_operation
	if(a > q) printf("error barrett_reduce_uint64_uint64\n");
#endif
}

// #define singleBarrett_qq 0
// __device__ __forceinline__ void singleBarrett_new(uint128_tt &a, uint64_tt q, uint128_tt mu)
// {
//     uint64_tt res;
//     asm("{\n\t"
//         "mul.hi.u64       %0, %2, %3;  		    \n\t"
//         "mad.hi.u64       %0, %1, %4, %0;		\n\t"
//         "mad.lo.u64       %0, %1, %3, %0;   	\n\t"
//         "mul.lo.u64       %0, %0, %5;      		\n\t"
//         "sub.u64          %0, %2, %0;      		\n\t"
//         "}"
//         : "=l"(res)
//         : "l"(a.high), "l"(a.low), "l"(mu.high), "l"(mu.low), "l"(q));
// 	csub_q(res, q<<1);
// 	csub_q(res, q);
//     a.high = 0;
//     a.low = res;
// }

__device__ __forceinline__ void sub_uint128_uint128(uint128_tt& a, const uint128_tt& b)
{
	asm("{\n\t"
		"sub.cc.u64      %1, %3, %5;    \n\t"
		"subc.u64        %0, %2, %4;    \n\t"
		"}"
		: "=l"(a.high), "=l"(a.low)
		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
}

// __device__ __forceinline__ void add_uint128_uint128(uint128_tt& a, const uint128_tt& b)
// {
// 	asm("{\n\t"
// 		"add.cc.u64      %1, %3, %5;    \n\t"
// 		"addc.u64        %0, %2, %4;    \n\t"
// 		"}"
// 		: "=l"(a.high), "=l"(a.low)
// 		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
// }
__device__ __forceinline__ void add_uint128_uint128(uint128_tt& a, const uint128_tt& b)
{
	uint128_tt z;
	z.low = a.low + b.low;
	z.high = a.high + b.high + (z.low < a.low);
	a = z;
}

__forceinline__ __device__ void sub_uint128_uint64(const uint128_tt& operand1,
												   const uint64_tt& operand2,
												   uint128_tt& result)
{
	asm("{\n\t"
		"sub.cc.u64     %1, %3, %4;\n\t"
		"subc.u64    	%0, %2, 0;\n\t"
		"}"
		: "=l"(result.high), "=l"(result.low)
		: "l"(operand1.high), "l"(operand1.low), "l"(operand2));
}

__forceinline__ __device__ void add_uint128_uint64(const uint128_tt& operand1,
												   const uint64_tt& operand2,
												   uint128_tt& result)
{
	asm("{\n\t"
		"add.cc.u64     %1, %3, %4;\n\t"
		"addc.u64    	%0, %2, 0;\n\t"
		"}"
		: "=l"(result.high), "=l"(result.low)
		: "l"(operand1.high), "l"(operand1.low), "l"(operand2));
}

__forceinline__ __device__ void madc_uint64_uint64_uint128(const uint64_tt& operand1,
														   const uint64_tt& operand2,
														   uint128_tt& result)
{
	asm("{\n\t"
		"mad.lo.cc.u64		%0, %4, %5, %2;\n\t"
		"madc.hi.u64    	%1, %4, %5, %3;\n\t"
		"}"
		: "=l"(result.low), "=l"(result.high)
		: "l"(result.low), "l"(result.high), "l"(operand1), "l"(operand2));
}

__forceinline__ __device__ uint64_tt negate_modq(const uint64_tt& operand1, const uint64_tt& q)
{
	if(operand1 == 0) return 0;
	return q - operand1;
}


#define max_tnum 14
#define max_Riblock_num 16
#define max_Qjblock_num 48
#define max_pqt_num 80
#define max_p_num 10
#define max_gamma_num 10

// pqt_i in constant memory
__constant__ uint64_tt pqt_cons[max_pqt_num];
__constant__ uint64_tt pqt2_cons[max_pqt_num];
// pq_mu_i in constant memory
__constant__ uint64_tt pqt_mu_cons_high[max_pqt_num];
__constant__ uint64_tt pqt_mu_cons_low[max_pqt_num];
// T//2 mod pqt in constant memory
__constant__ uint64_tt halfTmodpqti_cons[max_pqt_num];

__constant__ uint64_tt Pmodqt_cons[max_pqt_num];
__constant__ uint64_tt Pmodqt_shoup_cons[max_pqt_num];
__constant__ uint64_tt Pinvmodqi_cons[max_pqt_num];
__constant__ uint64_tt Pinvmodqi_shoup_cons[max_pqt_num];
__constant__ uint64_tt pHatInvVecModp_cons[max_p_num];
__constant__ uint64_tt pHatInvVecModp_shoup_cons[max_p_num];
__constant__ uint64_tt Rimodti_cons[max_tnum * max_Riblock_num];
__constant__ uint64_tt Tmodpqi_cons[max_pqt_num];