#pragma once

#include "Context_23.h"

using namespace std;

void Context_23::getPrimeCKKS(int hamming_weight)
{
	switch (logN)
	{
	case 13:
		//  9q 3p 5t 4gamma (0.131)(best)
		// 10q 2p 4t 3gamma (0.155)
		// 11q 1p 3t 3gamma (0.173)
		precision = 0x1000000000; // 1 << 36

        qVec = {
			0x4001b00001, 
			0xfff9c0001, 0xfff8e0001, 0xfff840001, 0xfff700001,
        };
		pVec = { // 36 x 6
            0x10a19000001, //0x10a3b000001,
		};
		tVec = { // 60 x 8
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, //0xfffffffff5a0001,
		};
        gamma = 3;
		// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
		break;

	case 14:
		//  9q 3p 5t 4gamma (0.131)(best)
		// 10q 2p 4t 3gamma (0.155)
		// 11q 1p 3t 3gamma (0.173)
		precision = 0x1000000000; // 1 << 36

        qVec = {
			0x4001b00001, 
			0xfff9c0001, 0xfff8e0001, 0xfff840001,	0xfff700001,
			0xfff640001, 
			0x1000a20001, 
			0x1000b40001, 0x1000f60001,	0x10011a0001,// 0x1001220001,
        };
		pVec = { // 36 x 6
            0x10a19000001, 0x10a3b000001,// 0x10a83000001,
		};
		tVec = { // 60 x 8
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, 0xfffffffff5a0001,
			// 0xfffffffff2a0001,// 0xfffffffff240001,
		};
        gamma = 3;
		// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
		break;

	case 15:
        precision = 1L<<49; // 1 << 36

        qVec = {
			0x20000018e0001, 0x2000001c00001, 0x2000001ca0001, 0x2000001d20001, 
			0x1fffffee80001, 0x1fffffede0001, 0x1fffffec40001, 0x1fffffe780001, 
			0x2000000ce0001, 0x20000013a0001, 0x20000013c0001, 0x20000015a0001, 
			
			// 0x1ffffffea0001, 0x1ffffffd40001, 0x1ffffffba0001, 0x1ffffffb40001, 
			// 0x20000001a0001, 0x20000005e0001, 0x2000000860001, 0x2000000b00001, 
			// 0x1ffffffb00001, 0x1ffffffa20001, 0x1ffffff780001, 0x1ffffff5c0001, 
			// 0x1fffffe5a0001, 0x1fffffe480001, 0x1fffffde80001, 0x1fffffde20001,
        };
		pVec = { // 36 x 6
			0x2000002600001, 0x20000027e0001, 0x2000002800001, 0x20000029e0001, 

		};
		tVec = { // 60 x 8
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, 0xfffffffff5a0001,
			0xfffffffff2a0001, 0xfffffffff240001,
			0xffffffffefe0001, // 0xffffffffeca0001,
		};
        gamma = 4;
		// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
		break;

	case 16:
        precision = 1L<<49; // 1 << 36
		// 44q 1p 3gamma 3t
		// 44q 4p 7gamma 7t
		// 42q 6p 
		qVec = { 
			//  without bootstrapping
			// 36bit x ??
			// 0x10004a0001, 0x1000500001, 0x1000960001, 0x1000a20001,
			// 0x1000b40001, 0x1000f60001, 0x10011a0001, 0x1001220001, 
			// 0xfff280001, 0xfff100001, 0xffefe0001, 0xffee80001, 

			// 0xffff00001, 0xfff9c0001, 0xfff8e0001, 0xfff840001, 
			// 0xfff700001, 0xfff640001, 0xfff4c0001, 0xfff3c0001, 
			// 0x10014c0001, 0x1001680001, 0x10017c0001, 0x1001880001, 
			
			// 0xffee20001, 0xffeda0001, 0xffeca0001, 0xffea40001, 
			// 0xffe940001, 0xffe920001, 0xffe760001, 0xffe040001, 
			// 0x1001940001, 0x1001a40001, 0x1001d00001, // 0x1001fa0001, 

			// 0xffdf80001, 0xffdf00001, 0xffdd20001, 0xffdbc0001, 
			// 0x1002180001, 0x10021c0001, 0x10021e0001, 0x1002300001, 
			// 0x1002340001, 0x1002480001, 0x1002540001, 0x10025a0001, 

			// for bootstrapping
			// 48bit
			// 0x10000001a0001, 0x10000001e0001, 0x1000000320001, 0x1000000380001, 			
			// 0xfffffffa0001, 0xfffffff00001, 0xffffffde0001, 0xffffff6a0001, 
			// 0x1000000500001, 0x1000000720001, 0x1000000ba0001, 0x1000000c00001, 
			
			// 0xfffffe8e0001, 0xfffffe5e0001, 0xfffffe5c0001, 0xfffffe580001, 
			// 0x1000000e00001, 0x1000001140001, 0x10000011a0001, 0x1000001380001, 
			// 0xffffff280001, 0xffffff060001, 0xfffffed60001, 0xfffffebc0001, 

			0x20000018e0001, 0x2000001c00001, 0x2000001ca0001, 0x2000001d20001, 
			0x1fffffee80001, 0x1fffffede0001, 0x1fffffec40001, 0x1fffffe780001, 
			0x2000000ce0001, 0x20000013a0001, 0x20000013c0001, 0x20000015a0001, 
			
			0x1ffffffea0001, 0x1ffffffd40001, 0x1ffffffba0001, 0x1ffffffb40001, 
			0x20000001a0001, 0x20000005e0001, 0x2000000860001, 0x2000000b00001, 
			0x1ffffffb00001, 0x1ffffffa20001, 0x1ffffff780001, 0x1ffffff5c0001, 
			0x1fffffe5a0001, 0x1fffffe480001, 0x1fffffde80001, 0x1fffffde20001,
		};
		pVec = { // 36 x 8
			// 0x20000e0001, 0x2000140001, 0x20004a0001, 0x2000580001, 
			// 0x2000760001, 0x2000be0001, 
			// 0x2000c40001, // 0x2000ce0001, 
			// // 0x2000da0001, 0x2000ee0001, 0x2001580001, 0x20016c0001,

			// 0x7ffffd20001, 0x7ffffaa0001, 0x7ffffa80001, 0x7ffff8c0001, 
			// 0x7ffff620001, 0x7ffff380001, 
			// 0x7ffff360001, 0x7ffff260001, 
			// 0x7fffe7c0001, 0x7fffe660001, 0x7fffe600001, 0x7fffe460001, 

			
			// 0x1000001440001, 0x10000015e0001, 0x1000001760001, 0x10000017c0001, 
			// 0x1000001880001, 0x1000001ce0001, // 0x1000001ec0001, 0x1000001fa0001,
			
			//0x2000001d60001, 0x2000001e80001, 0x2000002020001, 0x2000002260001, 
			0x2000002600001, 0x20000027e0001, 0x2000002800001, 0x20000029e0001, 
		};
		tVec = { // 60 x 10
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, 0xfffffffff5a0001,
			0xfffffffff2a0001, 0xfffffffff240001,
			0xffffffffefe0001, 0xffffffffeca0001,

			//0xffffffffd2a0001, 0xffffffffbf20001,
			//0xffffffff1fe0001, // 0xffffffff0c60001,
			// 0xfffffffef8e0001, // 0xfffffffed1e0001,
			// 0xfffffffe69e0001, 0xfffffffe4960001,

			// // 0x7ffffffffcc0001, 0x7ffffffffba0001,
			// // 0x7ffffffffb00001, 0x7ffffffff320001,
			// 0x7ffffffff2c0001, // 0x7ffffffff240001,
			// 0x7fffffffefa0001, 0x7fffffffede0001,
			// 0x1fffffffd80001, // 0x1fffffffb60001, 
			// // 0x1fffffff920001, 0x1fffffff900001, 
			// 0x1fffffff8c0001, 0x1fffffff740001, 0x1fffffff5c0001, 0x1fffffff4a0001,
		
			// 0x7fffffffe900001, 0x7fffffffe3c0001,
		};
        gamma = 4;
		// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r

		// (1/2pi)^1/4 * cos((x - 0.025) * 5 * pi)

		break;
	default:
		break;
	}
	if(hamming_weight == 64){
		// deg = 32 for the case of sk's hamming weight = 64
		// after modup the coeffs are in [-12,12]
		// (1/2pi)^0.25 cos(5pi x) on [-1,1]
		// eval_sine_chebyshev_coeff = {
		// 	0.08151901123667502, -2.423417716864714e-16, 0.1801151036341428, 2.0476928804320693e-16, 
		// 	0.21918003942151904, 3.3940360564680214e-16, 0.23493811667739298, 7.952933770473556e-17, 
		// 	0.14591318563283584, -1.8472930065553757e-16, -0.0868201018180367, 1.1428400971018604e-16, 
		// 	-0.26375631567246055, 3.419886769053114e-16, -0.009644738752433574, -2.7255192760361437e-16, 
			
		// 	0.3063629671619839, -8.234216392531856e-17, -0.2736243112492961, -1.5639103860729592e-16, 
		// 	0.13166482834922644, 5.730215052519396e-17, -0.04293705816443825, -1.6691031770572663e-17, 
		// 	0.010424130737126499, 1.742438262030931e-16, -0.001985999699585261, -1.2899370891073593e-16, 
		// 	0.00030762663662056906, -3.2906830602999636e-16, -3.99092135655925e-05, -9.066232632611505e-17
		// };
		// double_angle_cost_level = 2;
		eval_sine_chebyshev_coeff = {
			-0.14401680084669205, -3.079204648066709e-16, -0.3476430167314101, -5.61900299121954e-17, 
			-0.4325416565000668, 2.991865796383431e-16, -0.184478210109887, -6.510515187128937e-17, 
			0.5117195606552281, 1.4166828155139625e-16, -0.25230602824436693, 1.3104500739415751e-16, 
			0.06367780908917112, 4.2771786538594163e-17, -0.0102192375444178, -4.581604932549691e-17, 
			
			0.0011546348522567492, -5.738125216112128e-16, -9.763524111681942e-05, 4.874371146044649e-16, 
			6.441324532849805e-06, -7.82345106198884e-18, -3.415606147977149e-07, 4.3113505964431566e-17
		};
		double_angle_cost_level = 3;
		eval_sine_K = 12;
	} else if(hamming_weight == 192){
		// deg = 64 for the case of sk's hamming weight = 192
		// after modup the coeffs are in [-25,25]
		// (1/2pi)^0.25 cos(5pi x) on [-1,1]
		eval_sine_chebyshev_coeff = {
			-0.20848398838931007, 3.079204648066709e-17, -0.44915138066126764, -9.775237185829787e-18, 
			-0.4338597805643831, -5.620786054503757e-18, -0.04826012312644306, 6.305042788451188e-17, 
			0.5756991397398961, 1.5127211619518719e-16, -0.34230449716826267, -1.0154048598534574e-16, 
			0.09835465361190608, 1.722889254900243e-16, -0.017632115607294278, -8.699990702570468e-17, 
			
			0.002205183040244195, -2.3827202322089167e-16, -0.0002053253964503218, 6.214617156678451e-16, 
			1.486644873035847e-05, -2.382720232208916e-16, -8.63267831706962e-07, -3.49464976119931e-17
		};
		double_angle_cost_level = 4;
		eval_sine_K = 25;
	} else {
        throw invalid_argument("hamming weight error!");
	}
}