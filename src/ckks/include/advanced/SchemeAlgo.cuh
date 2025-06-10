#pragma once

#include "SchemeAlgo.h"

SchemeAlgo::SchemeAlgo(Context_23& context, Scheme_23& scheme, SecretKey& secretkey):context(context), scheme(scheme), secretkey(secretkey)
{
    N = context.N;
	slots = context.slots;
	logN = context.logN;
	logslots = context.logslots;
    q_num = context.q_num;
    p_num = context.p_num;
    maxLevel = context.L;
    precision = context.precision;
}

void SchemeAlgo::malloc_bsgs_buffer(int cheby_degree)
{
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int Qj_blockNum = context.Qj_blockNum;

    int cheby_tree_level = ceil(log2(cheby_degree));

    int tree_node_num = 1<<int(ceil(log2(cheby_degree)));
    cudaMalloc(&chebyshev_tree_cipher_pool, sizeof(uint64_tt) * N * q_num * 2 * tree_node_num);
    for(int i = 0; i < tree_node_num; i++)
    {
        Ciphertext* this_cipher = new Ciphertext(chebyshev_tree_cipher_pool + i * (N * q_num * 2), N, maxLevel, maxLevel - 4, slots, NTL::RR(precision));
        chebyshev_tree_pool.push_back(this_cipher);
    }

    for(int i = 0; i < ceil(log2(cheby_degree)); i++)
    {
        cheby_basis_pool.push_back(new Ciphertext(N, maxLevel, maxLevel-4, slots, NTL::RR(precision)));
    }
}

void splitCoeffsPolyVector(int split, vector<Chebyshev_Polynomial*>& chebyshev_poly_coeff_tree_pool, int tree_idx)
{
    Chebyshev_Polynomial *coeffs = chebyshev_poly_coeff_tree_pool[tree_idx];
    Chebyshev_Polynomial *coeffsq = chebyshev_poly_coeff_tree_pool[tree_idx*2];
    Chebyshev_Polynomial *coeffsr = chebyshev_poly_coeff_tree_pool[tree_idx*2 + 1];

    coeffsr->coeffs = vector<double>(split);
    if(coeffs->maxDegree == coeffs->degree())
    {
        coeffsr->maxDegree = split - 1;
    }
    else
    {
        coeffsr->maxDegree = coeffs->maxDegree - (coeffs->degree() - split + 1);
    }

    for(int i = 0; i < split; i++) coeffsr->coeffs[i] = coeffs->coeffs[i];

    coeffsq->coeffs = vector<double>(coeffs->degree() - split + 1);
    coeffsq->maxDegree = coeffs->maxDegree;

    coeffsq->coeffs[0] = coeffs->coeffs[split];

    for(int i = split+1, j = 1; i < coeffs->degree() + 1; i++, j++)
    {
        coeffsq->coeffs[i - split] = coeffs->coeffs[i] * 2;
        coeffsr->coeffs[split - j] = coeffsr->coeffs[split - j] - coeffs->coeffs[i];
    }
}

bool isNotNegligible(double a)
{
    // return abs(a) > 1e-14;
    // if(abs(a) > 1e-14) cout<<"Neg"<<endl;
    return true;
}

void SchemeAlgo::call_prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
{
    prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx, cheby_poly_pool_vector);
}

void SchemeAlgo::prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
{
    Chebyshev_Polynomial* poly = cheby_poly_pool_vector[tree_idx];
    if(poly->degree() < (1 << logSplit))
    {
        if(logSplit > 1 && poly->maxDegree%(1<<(logSplit+1)) > (1<<(logSplit-1)))
        {
            logDegree = ceil(log2(poly->degree()));
            logSplit = logDegree >> 1;
            
            cout<<"prepareChebyshevCoeffsTree1: "<<tree_idx<<endl;
            prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx, cheby_poly_pool_vector);
            return;
        }
        // cout<<"return here"<<endl;
        return;
    }

    int nextPower = 1 << logSplit;
    for(nextPower; nextPower < (poly->degree()>>1) + 1;) 
        nextPower <<= 1;

    splitCoeffsPolyVector(nextPower, cheby_poly_pool_vector, tree_idx);

    prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx*2, cheby_poly_pool_vector);
    prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx*2+1, cheby_poly_pool_vector);
}

void SchemeAlgo::evalRecurse(NTL::RR target_scale, int logSplit, int logDegree, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
{
    Chebyshev_Polynomial* poly = cheby_poly_pool_vector[tree_idx];

    if(poly->degree() < (1 << logSplit))
    {
        if(logSplit > 1 && poly->maxDegree%(1<<(logSplit+1)) > (1<<(logSplit-1)))
        {
            logDegree = ceil(log2(poly->degree()));
            logSplit = logDegree >> 1;

            evalRecurse(target_scale, logSplit, logDegree, tree_idx, cheby_poly_pool_vector);
            return;
        }
        evalFromPowerBasis(target_scale, tree_idx, cheby_poly_pool_vector);
        return;
    }
    int nextPower = 1 << logSplit;
    for(nextPower; nextPower < (poly->degree()>>1) + 1;) 
        nextPower <<= 1;

    // Chebyshev_Polynomial coeffsq, coeffsr;
    Chebyshev_Polynomial *coeffsq = cheby_poly_pool_vector[tree_idx*2];
    Chebyshev_Polynomial *coeffsr = cheby_poly_pool_vector[tree_idx*2 + 1];

    int idx_nextPower = int(log2(nextPower));
    Ciphertext* xpow = cheby_basis_pool[idx_nextPower];

    int level = xpow->l - 1;

    if (poly->maxDegree >= 1<<(logDegree-1)) {
		level++;
	}

    uint64_tt current_qi = context.qVec[level];

    evalRecurse(target_scale*current_qi/xpow->scale, logSplit, logDegree, tree_idx*2, cheby_poly_pool_vector);
    evalRecurse(target_scale, logSplit, logDegree, tree_idx*2+1, cheby_poly_pool_vector);

    scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *xpow);
    scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);

    scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);
}

void SchemeAlgo::evalFromPowerBasis(NTL::RR target_scale, int tree_idx, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
{
    Chebyshev_Polynomial *poly = cheby_poly_pool_vector[tree_idx];
    int mini = 0;
    for(int i = poly->degree(); i > 0; i--)
    {
        if(isNotNegligible(poly->coeffs[i]))
            mini = max(i, mini);
    }

    double c1 = poly->coeffs[0];

    int idx_mini = int(log2(mini));

    uint64_tt currentQi = context.qVec[cheby_basis_pool[idx_mini]->l];
    NTL::RR ctScale = target_scale * currentQi;

    // cudaMemset(chebyshev_tree_pool[tree_idx], 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
    chebyshev_tree_pool[tree_idx]->l = cheby_basis_pool[idx_mini]->l;
    chebyshev_tree_pool[tree_idx]->scale = ctScale;

    double c2 = poly->coeffs[1];

    if(isNotNegligible(c2))
    {
        NTL::RR constScale = target_scale * currentQi / cheby_basis_pool[0]->scale;

        scheme.multConstAndAddCipherEqual(*chebyshev_tree_pool[tree_idx], *(cheby_basis_pool[0]), c2, constScale);
    }
    // cout<<endl;

    if(isNotNegligible(c1))
    {
        scheme.addConstAndEqual(*chebyshev_tree_pool[tree_idx], c1);
    }


    scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);
    return;
}

// void SchemeAlgo::evalIteration(NTL::RR target_scale, int logDegree, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
// {
//     int evalchebyDegree = cheby_poly_pool_vector.size();
//     uint64_tt currentQi = context.qVec[cheby_basis_pool[0]->l];
//     NTL::RR ctScale = target_scale * currentQi;
//     NTL::RR constScale = ctScale / cheby_basis_pool[0]->scale;

//     for(int tree_idx = 31; tree_idx >= 16; tree_idx--)
//     {
//         Chebyshev_Polynomial *poly = cheby_poly_pool_vector[tree_idx];

//         double c1 = poly->coeffs[0];
//         double c2 = poly->coeffs[1];

//         uint64_tt currentQi = context.qVec[cheby_basis_pool[0]->l];

//         // cudaMemsetAsync(chebyshev_tree_pool[tree_idx], 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
//         chebyshev_tree_pool[tree_idx]->l = cheby_basis_pool[0]->l;
//         chebyshev_tree_pool[tree_idx]->scale = ctScale;

//         scheme.multConstAndAddCipherEqual(*chebyshev_tree_pool[tree_idx], *(cheby_basis_pool[0]), c2, constScale);
//         scheme.addConstAndEqual(*chebyshev_tree_pool[tree_idx], c1);

//         scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);
//     }

//     // cout<<(evalchebyDegree>>1) - 1<<" "<<1<<endl;
//     for(int tree_idx = 15; tree_idx >= 1; tree_idx--)
//     {
//         int idx_nextPower = 4 - int(log2(tree_idx));
//         Ciphertext* xpow = cheby_basis_pool[idx_nextPower];

//         Chebyshev_Polynomial *poly = cheby_poly_pool_vector[tree_idx];

//         scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *xpow);
//         scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);

//         scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);
//     }
// }

// #define const_layer_block 256
// __global__
// __launch_bounds__(
//     const_layer_block, 
//     POLY_MIN_BLOCKS)
// void cipher_cipher_mul_const_add_const_batch_kernel(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* add_const_real_buffer, uint32_tt n, int q_num, int idx_mod, int batch_size)
// {
//     register uint32_tt idx_in_pq = blockIdx.y;
//     register int idx_in_poly = blockIdx.x * const_layer_block + threadIdx.x;

//     register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
//     register uint64_tt ra;
//     register uint64_tt rb0 = device_b[(idx_in_pq + 0*q_num) * n + idx_in_poly];
//     register uint64_tt rb1 = device_b[(idx_in_pq + 1*q_num) * n + idx_in_poly];
//     register uint64_tt rc, rc_shoup;

// #pragma unroll
//     for(int i = 16; i < 32; i++)
//     {
//         rc = add_const_real_buffer[idx_in_pq + i*q_num*4];
//         rc_shoup = add_const_real_buffer[idx_in_pq + q_num + i*q_num*4];

//         ra = mulMod_shoup(rb0, rc, rc_shoup, q);
//         device_a[(idx_in_pq + 0*q_num) * n + idx_in_poly + i*n*q_num*2] = ra;

//         ra = mulMod_shoup(rb1, rc, rc_shoup, q) + add_const_real_buffer[idx_in_pq + q_num*2 + i*q_num*4];
//         csub_q(ra, q);
//         device_a[(idx_in_pq + 1*q_num) * n + idx_in_poly + i*n*q_num*2] = ra;
//     }
// }

// void SchemeAlgo::evalIterationBatch(NTL::RR target_scale, int logDegree)
// {
//     int evalchebyDegree = chebyshev_poly_coeff_tree_pool.size();
//     uint64_tt currentQi = context.qVec[cheby_basis_pool[0]->l];
//     NTL::RR ctScale = target_scale * currentQi;
//     NTL::RR constScale = ctScale / cheby_basis_pool[0]->scale;

//     int t_num = context.t_num;
//     int Qj_blockNum = context.Qj_blockNum;
//     int Ri_blockNum = context.Ri_blockNum;

//     int cipher_min_level = maxLevel;

// /********************************************const layer********************************************/
// #pragma unroll
//     for(int tree_idx = evalchebyDegree>>1; tree_idx < evalchebyDegree; tree_idx++)
//     {
//         Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];

//         double c1 = poly->coeffs[0];
//         double c2 = poly->coeffs[1];

//         Ciphertext *cipher1 = chebyshev_tree_pool[tree_idx], *cipher2 = cheby_basis_pool[0];

//         uint64_tt currentQi = context.qVec[cipher2->l];

//         cudaMemsetAsync(chebyshev_tree_pool[tree_idx]->cipher_device, 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
//         cipher1->l = cipher2->l;
//         cipher1->scale = ctScale;

//         cipher1->l = min(cipher1->l, cipher2->l);
//         cipher_min_level = cipher1->l;
//         // cout<<"cipher_min_level: "<<cipher_min_level<<endl;

//         NTL::ZZ scaled_c2 = to_ZZ(round(target_scale * c2));
//         NTL::ZZ scaled_c1 = to_ZZ(round(cipher1->scale * c1));
//         for(int i = 0; i < cipher_min_level+1; i++)
//         {
//             add_const_copy_vec[i + tree_idx * q_num*4] = scaled_c2 % context.qVec[i];
//             add_const_copy_vec[i + tree_idx * q_num*4 + q_num] = x_Shoup(add_const_copy_vec[i + tree_idx * q_num*4], context.qVec[i]);
//             add_const_copy_vec[i + tree_idx * q_num*4 + q_num*2] = scaled_c1 % context.qVec[i];
//         }
//     }
//     cudaMemcpyAsync(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

//     dim3 mul_dim(N / const_layer_block, cipher_min_level+1);
//     cipher_cipher_mul_const_add_const_batch_kernel <<< mul_dim, const_layer_block >>> (chebyshev_tree_cipher_pool, cheby_basis_pool[0]->cipher_device, add_const_buffer, N, q_num, context.K, evalchebyDegree>>1);

// /********************************************const layer rescale********************************************/
// #pragma unroll
//     for(int tree_idx = evalchebyDegree>>1; tree_idx < evalchebyDegree; tree_idx++) scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);

// #pragma unroll
//     for(int tree_level = int(log2(evalchebyDegree)-2); tree_level >= 1; tree_level--)
//     {
//         int idx_start = (1<<tree_level), idx_end = (1<<(tree_level+1));
//         // printf("idx_start: %d   idx_end: %d\n", idx_start, idx_end);
//         for(int tree_idx = idx_start; tree_idx < idx_end; tree_idx++)
//         {
//             Ciphertext* cipher2 = cheby_basis_pool[4 - tree_level];
//             Ciphertext* cipher1 = chebyshev_tree_pool[tree_idx*2];
//             Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];

//             // scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *cipher2);
//             scheme.multAndEqual_beforeIP_23(*cipher1, *cipher2,
//                                             IP_input_temp  + (tree_idx-idx_start)*N*t_num*Qj_blockNum, 
//                                             axbx1_mul + (tree_idx-idx_start)*N*q_num,
//                                             bxbx_mul + (tree_idx-idx_start)*N*q_num);
//         }
//         context.external_product_T_swk_reuse(IP_output_temp, 
//                                             IP_input_temp, 
//                                             scheme.rlk_23->cipher_device, chebyshev_tree_pool[idx_end]->l, idx_start);

//         for(int tree_idx = idx_start; tree_idx < idx_end; tree_idx++)
//         {
//             Ciphertext* cipher2 = cheby_basis_pool[4 - tree_level];
//             Ciphertext* cipher1 = chebyshev_tree_pool[tree_idx*2];

//             scheme.multAndEqual_afterIP_23(*cipher1, *cipher2,
//                                            IP_output_temp + (tree_idx-idx_start)*N*t_num*Ri_blockNum*2,
//                                             axbx1_mul + (tree_idx-idx_start)*N*q_num,
//                                             bxbx_mul + (tree_idx-idx_start)*N*q_num);

//             scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);
//             scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);
//         }
//     }

//     Ciphertext* cipher2 = cheby_basis_pool[4];
//     Ciphertext* cipher1 = chebyshev_tree_pool[2];
//     Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[1];

//     scheme.multAndEqual_23(*chebyshev_tree_pool[2], *cipher2);

//     scheme.rescaleAndEqual(*chebyshev_tree_pool[2], target_scale);
//     scheme.add(*chebyshev_tree_pool[1], *chebyshev_tree_pool[2], *chebyshev_tree_pool[3]);
// }

void SchemeAlgo::evalPolynomialChebyshev(Ciphertext &cipher, NTL::RR target_scale, vector<Chebyshev_Polynomial*>& cheby_poly_pool_vector)
{
    int degree = cheby_poly_pool_vector[1]->degree();
    int logDegree = ceil(log2(degree));
    int logSplit = (logDegree >> 1);

    *(cheby_basis_pool[0]) = cipher;

    for(int idx = 1; idx < logDegree; idx++)
    {
        scheme.square_double_add_const_rescale(*(cheby_basis_pool[idx]), *(cheby_basis_pool[idx-1]), -1);
    }

    evalRecurse(target_scale, logSplit, logDegree, 1, cheby_poly_pool_vector);
    // evalIteration(target_scale, logDegree);
    // evalIterationBatch(target_scale, logDegree);
    // cout<<endl;

    cipher = *chebyshev_tree_pool[1];
}