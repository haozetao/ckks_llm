#pragma once

#include "Attention.h"
#include "../include/advanced/SchemeAlgo.cuh"
#include "../include/TimeUtils.cuh"

Attention::Attention(Context_23& context, Scheme_23& scheme, SchemeAlgo& scheme_algo, int token_len, int head_num, int d, SecretKey& sk)
    : context(context), scheme(scheme), scheme_algo(scheme_algo), token_len(token_len), head_num(head_num), d(d), sk(sk)
{
    if (scheme_algo.chebyshev_tree_pool.empty()) {
        scheme_algo.malloc_bsgs_buffer(32);
    }

    // exp(5(x-1)) 
    // x \in [-1, 1] -> [-10, 0]
    exp_cheby_coeffs = {0.18354081267894184, 0.32794453388908446, 0.23590381180224984, 0.1392214845586662, 
        0.06883803033185001, 0.029080636250469596, 0.01067675783091106, 0.0034564177904279412, 
        0.000998788017712841, 0.0002602965792747787, 6.172033233564635e-05, 1.3415806954468434e-05,
         2.6907827538706158e-06, 5.007266909759075e-07, 8.707430942944683e-08, 1.4433549381509271e-08};
    int exp_tree_node_num = 1<<int(ceil(log2(exp_cheby_coeffs.size())));
    for(int i = 0; i < exp_tree_node_num; i++)
    {
        exp_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    exp_cheby_poly_pool[1]->coeffs = exp_cheby_coeffs;
    exp_cheby_poly_pool[1]->maxDegree = exp_cheby_coeffs.size() - 1;

    {
        int degree = exp_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, exp_cheby_poly_pool);
    }



    // sigmoid_cheby_coeffs = {0.5, 0.5876811235265726, -2.007269726835591e-17, -0.12149623872303088, 
    //     -2.0072697268355913e-17, 0.035317888765259355, -1.0917210757920756e-16, -0.010693712782924763, 
    //     -1.8652562430325655e-16, 0.003260293428351401, -2.2019441004172328e-16, -0.0009931752451993756, 
    //     -9.895698703494615e-18, 0.00029550137814745605, -1.761892554953261e-17, -6.453127146099342e-05};
    // sigmoid_cheby_coeffs = {0.4999999999999999, 0.5876811283760183, -1.2245589764298666e-16, -0.12149626092930495, 
    //     9.11654200268892e-17, 0.03531796339510528, 2.500215614893023e-16, -0.010693957687070982, 
    //     2.6705411550039875e-16, 0.0032610953439276167, 4.965267385241698e-16, -0.000995800506585915, 
    //     5.013755013555445e-16, 0.00030409563293387924, 2.66463491171798e-16, -9.266603859557535e-05, 
    //     5.512919147316323e-16, 2.757290625878708e-05, 3.833288719350705e-16, -6.021416971387329e-06};
    sigmoid_cheby_coeffs = {0.49999999999999983, 0.6190561249910002, -3.779948047650626e-16, -0.16826527291908855, 
        -1.912969045814174e-16, 0.07086825138977516, 2.597786544428372e-16, -0.03212482603897069, 
        3.589248098772761e-17, 0.014825676200313642, -1.1719743484722945e-16, -0.006875368855731378, 
        2.2107731351872553e-16, 0.003192797680713223, -2.0128123752342198e-16, -0.0014832571008630544, 
        8.831562613597343e-16, 0.0006891406846301364, 2.741745067338399e-16, -0.00032018501678578935, 
        8.444549204356228e-16, 0.00014874420322440814, 4.574944982035514e-16, -6.905745663065152e-05, 
        2.2032077969240172e-16, 3.196897811145114e-05, 5.390316225920323e-16, -1.4600593693464791e-05, 
        6.80826518041179e-16, 6.238592363887885e-06, 3.2031350231224433e-16, -1.7248688345201689e-06};
    int sigmoid_tree_node_num = 1<<int(ceil(log2(sigmoid_cheby_coeffs.size())));
    for(int i = 0; i < sigmoid_tree_node_num; i++)
    {
        sigmoid_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    sigmoid_cheby_poly_pool[1]->coeffs = sigmoid_cheby_coeffs;
    sigmoid_cheby_poly_pool[1]->maxDegree = sigmoid_cheby_coeffs.size() - 1;
    {
        int degree = sigmoid_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, sigmoid_cheby_poly_pool);
    }
    sigmoid_x_max = 8;

    sigmoid_x_iter_2x_cheby_coeffs = {0.39135661064466676, 0.4311355190020014, 0.29684494955071405, 0.003438274502918351, 
        -0.10094472423159448, -0.053088638542605315, 0.007633927547045461, 0.022294214131234984, 
        0.008820873471426465, -0.0031790881125600147, -0.004686461144646479, -0.0013095091946053207, 
        0.0009484847435840586, 0.0009406999216865256, 0.0001556118752286015, -0.00024556942168665153, 
        -0.00017996299374255673, -7.596639137797628e-06, 5.8616228847240585e-05, 3.279028127733078e-05, 
        -3.4063866112867723e-06, -1.3616968671623255e-05, -5.935806433449409e-06, 2.521040148190015e-06};
    int sigmoid_x_iter_2x_tree_node_num = 1<<int(ceil(log2(sigmoid_x_iter_2x_cheby_coeffs.size())));
    for(int i = 0; i < sigmoid_x_iter_2x_tree_node_num; i++)
    {
        sigmoid_x_iter_2x_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    sigmoid_x_iter_2x_cheby_poly_pool[1]->coeffs = sigmoid_x_iter_2x_cheby_coeffs;
    sigmoid_x_iter_2x_cheby_poly_pool[1]->maxDegree = sigmoid_x_iter_2x_cheby_coeffs.size() - 1;
    {
        int degree = sigmoid_x_iter_2x_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, sigmoid_x_iter_2x_cheby_poly_pool);
    }

    // CDF_cheby_coeffs = {0.5, 0.6234577610429392, 8.07567900971873e-17, -0.17602452170600366, 
    //     1.1776422425135919e-16, 0.0761747991351066, -1.2929493435476463e-16, -0.03371833834600424, 
    //     -1.6116422569495642e-16, 0.014111526810286363, -2.0330980266310844e-16, -0.005444024438389357, 
    //     -1.224401966956653e-16, 0.0018840923558029613, -4.5685761667957374e-17, -0.0004415815053097376};
    CDF_cheby_coeffs = {0.49999999999999983, 0.6315854460608845, -3.8007362690147523e-16, -0.19758285625101127, -2.6928769802283985e-16, 0.10444703623487543, 3.3013421842775765e-16, -0.06173815279554332, -6.365934059249024e-17, 0.037352705565676415, -1.2762246142446612e-16, -0.022368752712872825, 2.052864515595652e-16, 0.013050442364969566, -3.1643334912463447e-16, -0.0073560603063709045, 8.572315590623611e-16, 0.003987549158219894, 1.901223608737585e-16, -0.0020735719558088446, 7.054569286734498e-16, 0.0010330783868063083, 5.506038234887787e-16, -0.0004928276902164144, 2.1704922439447746e-16, 0.00022494605544299398, 4.0420673305027137e-16, -9.775200192125993e-05, 5.534540149915478e-16, 3.899079251192208e-05, 3.769570264744171e-16, -1.0220905643195679e-05};
    int CDF_tree_node_num = 1<<int(ceil(log2(CDF_cheby_coeffs.size())));
    for(int i = 0; i < CDF_tree_node_num; i++)
    {
        CDF_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    CDF_cheby_poly_pool[1]->coeffs = CDF_cheby_coeffs;
    CDF_cheby_poly_pool[1]->maxDegree = CDF_cheby_coeffs.size() - 1;
    {
        int degree = CDF_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, CDF_cheby_poly_pool);
    }
    cdf_x_max = 8;

    CDF_x_iter_2x_cheby_coeffs = {0.5, 0.5916790537958231, 3.27552742097307e-17, -0.10748392237973169, 
        1.067701425180745e-16, 0.01646575746789658, -9.446121881423479e-17, -0.0005779701033667785, 
        -1.7913841602730062e-16, -6.743358384228921e-05, -5.745378466006292e-17, -1.1670517637866644e-05, 
        -6.298564631679064e-17, -2.7882304899101934e-06, -4.080318978588528e-17, -1.0264486512245965e-06};
    int CDF_x_iter_2x_tree_node_num = 1<<int(ceil(log2(CDF_x_iter_2x_cheby_coeffs.size())));
    for(int i = 0; i < CDF_x_iter_2x_tree_node_num; i++)
    {
        CDF_x_iter_2x_cheby_poly_pool.push_back(new Chebyshev_Polynomial());
    }
    CDF_x_iter_2x_cheby_poly_pool[1]->coeffs = CDF_x_iter_2x_cheby_coeffs;
    CDF_x_iter_2x_cheby_poly_pool[1]->maxDegree = CDF_x_iter_2x_cheby_coeffs.size() - 1;
    {
        int degree = CDF_x_iter_2x_cheby_poly_pool[1]->degree();
        int logDegree = ceil(log2(degree));
        int logSplit = (logDegree >> 1);

        scheme_algo.call_prepareChebyshevCoeffsTree(logSplit, logDegree, 1, CDF_x_iter_2x_cheby_poly_pool);
    }

    temp_cipher_nonlinear = new Ciphertext(context.N, context.L, context.L, context.slots, NTL::to_RR(context.precision));
    
    softmax_x_max = 5;
    activate_x_max = 5;

    /**********************************************prepare softmax mask********************************************/
    // token length < 128
    int N = context.N;
    int L = context.L;
    int slots = context.slots;
    int column_num = slots / token_len;
    cuDoubleComplex* column_mask_buffer_host = new cuDoubleComplex[slots];
    cuDoubleComplex* column_mask_buffer_device;
    cudaMalloc(&column_mask_buffer_device, sizeof(cuDoubleComplex) * slots);

    cout<<"prepare reduce sum mask"<<endl;
    for(int i = 0; i < 1; i++){
        Plaintext* mask_i = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
        column_mask_reduce.push_back(mask_i);

        vector<int> mask_idx;
        for(int t = 0; t < column_num / d; t++){
            mask_idx.push_back(t * d + i);
        }

        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, *mask_i);
    }
    {
        Plaintext* mask_i = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
        column_mask_reduce.push_back(mask_i);

        vector<int> mask_idx;
        mask_idx.push_back(0);

        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, *mask_i);
    }

    
    /******************************************Inv and Square Inv*********************************************/
    // K_inv = {1.708209458521407, 1.3347202619383138, 1.0593437148393925, 1.0017626264306325, 1.0000034823730521};
    K_inv = {1.708209458521407, 1.3347202619383138, 1.0593437148393925, 1.0017626264306325};

    K_sqrt_inv = {1.472141915860566, 1.1254419159515154, 1.007928806890655};

    for(int i = 0; i < 2; i++){
        nonlieanr_buffer.push_back(new Ciphertext(N, L, L, slots, NTL::RR(context.precision)));
    }
    /******************************Q * K^T***********************************/
    // prepare mask for CCMM Q*K^T
    cout<<"prepare Q*K^T mask"<<endl;
    for(int i = 1; i < d; i<<=1){
        // Plaintext* mask_i = new Plaintext(N, L, L, slots, NTL::RR(pow(context.precision, 2.0/3)));
        Plaintext* mask_i = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
        column_mask_ccmm.push_back(mask_i);

        vector<int> mask_idx;
        for(int j = 0; 2*j*i < d; j++){
            for(int k = 0; k < i; k++){
                for(int t = 0; t < column_num / d; t++){
                    mask_idx.push_back(t * d + j * i * 2 + k);
                }
            }
        }
        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, *mask_i);
        
        // Plaintext* mask_i_other = new Plaintext(N, L, L, slots, NTL::RR(pow(context.precision, 0.6666)));
        Plaintext* mask_i_other = new Plaintext(N, L, L, slots, NTL::RR(context.precision));
        column_mask_ccmm.push_back(mask_i_other);
        for (int idx = 0; idx < mask_idx.size(); idx++){
            mask_idx[idx] += i;
        }
        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, *mask_i_other);
    }

    const int head_blocks = column_num / d;
    
    /******************************Q * K^T***********************************/
    // tmpcipher_buffer = (Ciphertext **)malloc(sizeof(Ciphertext) * (log2(d)+1));
    tmpcipher_buffer = new Ciphertext*[int(log2(d) + 1)];
    leafnode = scheme_algo.chebyshev_tree_pool[log2(d)+1];
    tmp_shift_K = scheme_algo.chebyshev_tree_pool[log2(d)+2];
    cout << "d=" << d << endl;
    cout << "scheme_algo.chebyshev_tree_pool.size() = " << scheme_algo.chebyshev_tree_pool.size() << endl;

    /****************************** begin *V ***********************************/
    rot_diag = new cuDoubleComplex[slots];
    cudaMalloc(&device_diag, sizeof(cuDoubleComplex) * slots);

    for (int j = 0; j < 8; j++){
        for(int i = 0; i < 8; i++){
            memset(rot_diag, 0, sizeof(cuDoubleComplex)*slots);
            const int repeats = slots / d;
            for (int k = 0; k < repeats; k++){
                rot_diag[j*8+i+k*64].x = 1;
                // printf("%d, ",j*8+i+k*64);
            }
            // printf("\n");
            cudaMemcpy(device_diag, rot_diag, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
            plain_tau_diag[j*8+i] = Plaintext(N, L, L, slots, NTL::RR(context.precision));
            context.encode(device_diag, plain_tau_diag[j*8+i]);
        }
    }
    // mask
    for(int zero_column_num = 1; zero_column_num < d; zero_column_num++){
        mask_ccmm_left[zero_column_num] = Plaintext(N, L, L, slots, NTL::RR(context.precision));
        memset(rot_diag, 0, sizeof(cuDoubleComplex)*slots);
        vector<int> mask_idx;
        // one ciphertext for four head
        for (int k = 0; k < head_blocks; k++){
            for (int j = 0; j < d-zero_column_num; j++){
                mask_idx.push_back(j+k*d);
            }
        }
        
        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, mask_ccmm_left[zero_column_num]);
    }

    for(int mask_column_num = 1; mask_column_num < d; mask_column_num++){
        mask_ccmm_right[mask_column_num] = Plaintext(N, L, L, slots, NTL::RR(context.precision));
        memset(rot_diag, 0, sizeof(cuDoubleComplex)*slots);
        vector<int> mask_idx;
        for (int k = 0; k < head_blocks; k++){
            for (int j = 0; j < mask_column_num; j++){
                mask_idx.push_back(d-j-1 + k*d);
            }
        }
        prepareMask(column_mask_buffer_host, mask_idx);
        cudaMemcpy(column_mask_buffer_device, column_mask_buffer_host, sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
        context.encode(column_mask_buffer_device, mask_ccmm_right[mask_column_num]);
    }

    for (int i = 0; i < mulV_gs; i++){
        mulV_gs_res[i] = scheme_algo.chebyshev_tree_pool[i];
    }

    delete[] column_mask_buffer_host;
    cudaFree(column_mask_buffer_device);

    /****************************** end *V ***********************************/
}

void Attention::prepareMask(cuDoubleComplex* mask_host, vector<int> idx_vector)
{
    int slots = context.slots;
    memset(mask_host, 0, sizeof(cuDoubleComplex) * slots);
    // for(int i = 0; i < slots; i++) mask_host[i].x = mask_host[i].y = 0;

    for(auto idx : idx_vector){
        for(int i = 0; i < token_len; i++){
            int column_num = slots / token_len;
            mask_host[i * column_num + idx].x = 1;
        }
    }
}

void Attention::addKey(SecretKey& sk)
{
    cout<<"add CCMM & reduce sum Keys"<<endl;
    printf("log2(d): %lf\n", log2(d));
    int slots = context.slots;
    int column_num = slots / token_len;
    int head_blocks = column_num / d;
    // CCMM Q*K^T
    // softmax reduce sum
    int max_pow = 0;
    while ((1 << max_pow) < d * head_blocks) {
        max_pow++;
    }
    for(int i = 0; i < max_pow + 1; i++){
        int shift = 1 << i;
        scheme.addLeftRotKey_23(sk, shift);
        printf("%d, ", shift);
        scheme.addLeftRotKey_23(sk, slots - shift);
        printf("%d, ", slots - shift);
    }
    scheme.addLeftRotKey_23(sk, slots / 2);
    printf("%d, ", slots / 2);
    scheme.addLeftRotKey_23(sk, column_num);
    printf("%d, ", column_num);
    scheme.addLeftRotKey_23(sk, 16 * column_num);
    printf("%d, ", 16 * column_num);
    // for tau gs
    for(int i = 0; i < 16; i++){
        scheme.addLeftRotKey_23(sk, column_num * 4 * i);
        printf("%d, ", column_num * 4 * i);
    }
    // for tau bs
    for(int i = 0; i < 4; i++){
        scheme.addLeftRotKey_23(sk, column_num * i);
        printf("%d, ", column_num * i);
    }
    // for *V
    scheme.addLeftRotKey_23(sk, slots - (column_num - 1));
    printf("%d, ", slots - (column_num - 1));
    cout<<endl;
    scheme.addLeftRotKey_23(sk, slots - d);
    printf("%d, ", slots - d);
    cout<<endl;
    for(int i = 0; i < mulV_bs; i++){
        scheme.addLeftRotKey_23(sk, column_num * i);
        printf("%d, ", column_num * i);
    }
}


void Attention::evalExp(Ciphertext& cipher)
{
    scheme.multConstAndEqual(cipher, 1./softmax_x_max);
    scheme.rescaleAndEqual(cipher);
    scheme.addConstAndEqual(cipher, 1);

    NTL::RR target_scale = cipher.scale;

    // First mapping: x \in [-10, 0] -> [-1, 1]
    // Evaluate the Chebyshev polynomial for exp(5(x-1))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, exp_cheby_poly_pool);
}

void Attention::evalExp_iter(Ciphertext& cipher, int iter)
{
    scheme.multConstAndEqual(cipher, 1./softmax_x_max/(1<<iter));
    scheme.rescaleAndEqual(cipher);
    scheme.addConstAndEqual(cipher, 1);

    NTL::RR target_scale = cipher.scale;

    // First mapping: x \in [-10, 0] -> [-1, 1]
    // Evaluate the Chebyshev polynomial for exp(5(x-1))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, exp_cheby_poly_pool);
    for (int i = 0; i < iter; i++){
        scheme.multAndEqual_23(cipher, cipher);
        scheme.rescaleAndEqual(cipher);
    }
}

// CDF function: 0.5 * (1 + erf(x / sqrt(2)))
void Attention::evalCDF(Ciphertext& cipher)
{
    int iter = 3;
    scheme.multConstAndEqual(cipher, 1./cdf_x_max/(1<<iter));
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, CDF_cheby_poly_pool);
    // Evaluate sigmoid(2x) from sigmoid(x)
    target_scale = cipher.scale;
    for(int i = 0; i < iter; i++){
        scheme.multConstAndEqual(cipher, 2);
        scheme.addConstAndEqual(cipher, -1);
        scheme_algo.evalPolynomialChebyshev(cipher, target_scale, CDF_x_iter_2x_cheby_poly_pool);
    }
}

// Sigmoid function: exp(x) / (1 + exp(x))
void Attention::evalSigmoid(Ciphertext& cipher)
{
    int iter = 3;
    scheme.multConstAndEqual(cipher, 1./sigmoid_x_max/pow(2, iter));
    scheme.rescaleAndEqual(cipher);
    NTL::RR target_scale = cipher.scale;

    // Evaluate the Chebyshev polynomial for 0.5 * (1 + erf(x / sqrt(2)))
    scheme_algo.evalPolynomialChebyshev(cipher, target_scale, sigmoid_cheby_poly_pool);

    // Evaluate sigmoid(2x) from sigmoid(x)
    target_scale = cipher.scale;
    for(int i = 0; i < iter; i++){
        scheme_algo.evalPolynomialChebyshev(cipher, target_scale, sigmoid_x_iter_2x_cheby_poly_pool);
    }
}

void Attention::evalGeLU(Ciphertext& cipher)
{
    Ciphertext* mult_buffer = temp_cipher_nonlinear;
    *mult_buffer = cipher;
    evalCDF(cipher);
    scheme.multAndEqual_23(cipher, *mult_buffer);
    scheme.rescaleAndEqual(cipher);
}

void Attention::evalSiLU(Ciphertext& cipher)
{
    Ciphertext* mult_buffer = temp_cipher_nonlinear;
    *mult_buffer = cipher;
    evalSigmoid(cipher);
    scheme.multAndEqual_23(cipher, *mult_buffer);
    scheme.rescaleAndEqual(cipher);
}

// only in [0, 1]
void Attention::evalInv(Ciphertext& cipher, double upper_bound)
{
    double scale = 1.0 / upper_bound;
    scheme.multConstAndEqual(cipher, scale);
    if(abs(scale - int(scale)) >= 1e-6) scheme.rescaleAndEqual(cipher);
    
    Ciphertext* bx = scheme_algo.chebyshev_tree_pool[0];
    Ciphertext* ax = &cipher;
    Ciphertext* temp = scheme_algo.chebyshev_tree_pool[1];
    temp->l = cipher.l;
    temp->scale = cipher.scale;
    bx->l = cipher.l;
    bx->scale = cipher.scale;
    cout << "cipher->scale = " << cipher.scale << endl;
    cout << "bx->scale = " << bx->scale << endl;
    NTL::RR target_scale = cipher.scale;

    // up scale
    scheme.multConstAndEqual(cipher, 2);
    cipher.scale *= 2;
    
    // b = (2 - k0*a)
    ax->scale = ax->scale / K_inv[0];
    scheme.constSub(*bx, *ax, 2);
    // cout << "after scheme.constSub(*bx, *ax, 2); bx->scale = " << bx->scale << endl;
    ax->scale = ax->scale * K_inv[0];
    // a = a * (2 - k0*a)
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);
    // a = k0 * a * (2 - k0*a)
    ax->scale = ax->scale / K_inv[0];
    // b = k0 * b * (2 - k0*a)
    // cout << "before bx->scale = bx->scale / K_inv[0]; bx->scale = " << bx->scale << endl;
    bx->scale = bx->scale / K_inv[0];
    // cout << "after bx->scale = bx->scale / K_inv[0]; bx->scale = " << bx->scale << endl;

    for(int i = 1; i < K_inv.size(); i++){
        // temp = (2 - k0*a)
        ax->scale = ax->scale / K_inv[i];
        scheme.constSub(*temp, *ax, 2);
        ax->scale = ax->scale * K_inv[i];

        // a = a * (2 - k0*a)
        scheme.multAndEqual_23(*ax, *temp);
        scheme.rescaleAndEqual(*ax);
        // a = k0 * a * (2 - k0*a)
        ax->scale = ax->scale / K_inv[i];
        // cout << "before scheme.multAndEqual_23(*bx, *temp); bx->scale = " << bx->scale << endl;
        // cout << "before scheme.multAndEqual_23(*bx, *temp); temp->scale = " << temp->scale << endl;
        // b = k0 * b * (2 - k0*a)
        scheme.multAndEqual_23(*bx, *temp);
        scheme.rescaleAndEqual(*bx);
        // cout << "after scheme.multAndEqual_23(*bx, *temp); bx->scale = " << bx->scale << endl;
        bx->scale = bx->scale / K_inv[i];
        // cout << "after bx->scale = bx->scale / K_inv[i]; bx->scale = " << bx->scale << endl;
    }
    // cout << "target_scale = " << target_scale << endl;
    // cout << "bx->scale = " << bx->scale << endl;
    // cout << "upper_bound = " << upper_bound << endl;
    double tt = to_double(target_scale / bx->scale / upper_bound);
    bx->scale *= target_scale / bx->scale;
    // cout << "after bx->scale *= target_scale / bx->scale; bx->scale = " << bx->scale << endl;
    // cout << "tt = " << tt << endl;
    scheme.multConstAndEqual(*bx, tt);
    // cout << "bx->scale = " << bx->scale << endl;
    scheme.rescaleAndEqual(*bx);
    // cout << "bx->scale = " << bx->scale << endl;
    cipher = *bx;
}

// only in [0, 1]
void Attention::evalSqrtInv(Ciphertext& cipher, SecretKey& sk, double upper_bound)
{
    double scale = 1.0 / upper_bound;
    scheme.multConstAndEqual(cipher, scale);
    if(abs(scale - int(scale)) >= 1e-6) scheme.rescaleAndEqual(cipher);
    
    Ciphertext* bx = scheme_algo.chebyshev_tree_pool[0];
    Ciphertext* ax = &cipher;
    Ciphertext* temp = scheme_algo.chebyshev_tree_pool[1];
    temp->l = cipher.l;
    temp->scale = cipher.scale;
    bx->l = cipher.l;
    bx->scale = cipher.scale;

    NTL::RR target_scale = cipher.scale;
    cout<<"ax.l: "<<ax->l<<endl;

    // b = (3 - k0*a)
    ax->scale = ax->scale / K_sqrt_inv[0];
    scheme.constSub(*bx, *ax, 3);
    ax->scale = ax->scale * K_sqrt_inv[0];
    // a = a * (3 - ki*a)**2
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);    
    // a = k0 / 4 * a * (3 - k0*a)**2
    ax->scale = ax->scale / (K_sqrt_inv[0] / 4);
    // b = ki**0.5 / 2 * b * (3 - k0*a)
    bx->scale = bx->scale / ((pow(K_sqrt_inv[0], 0.5) / 2));

    for(int i = 1; i < K_sqrt_inv.size(); i++){
        // temp = (3 - ki*a)
        ax->scale = ax->scale / K_sqrt_inv[i];
        scheme.constSub(*temp, *ax, 3);
        ax->scale = ax->scale * K_sqrt_inv[i];

        // a = a * (3 - ki*a)**2
        scheme.multAndEqual_23(*ax, *temp);
        scheme.rescaleAndEqual(*ax);
        scheme.multAndEqual_23(*ax, *temp);
        scheme.rescaleAndEqual(*ax);
        // a = ki / 4 * a * (3 - ki*a)**2
        ax->scale = ax->scale / (K_sqrt_inv[i] / 4);

        // b = ki**0.5 / 2 * b * (3 - ki*a)
        scheme.multAndEqual_23(*bx, *temp);
        scheme.rescaleAndEqual(*bx);
        bx->scale = bx->scale / ((pow(K_sqrt_inv[i], 0.5) / 2));
    }
    double tt = to_double(target_scale / bx->scale / pow(upper_bound, 0.5));
    cout << "tt = " << tt << endl;
    bx->scale *= target_scale / bx->scale;
    scheme.multConstAndEqual(*bx, tt);
    scheme.rescaleAndEqual(*bx);
    cipher = *bx;
}

// only in [0, 1]
void Attention::evalSqrtInv_without_mul_tt(Ciphertext& cipher, SecretKey& sk, double upper_bound)
{
    double scale = 1.0 / upper_bound;
    scheme.multConstAndEqual(cipher, scale);
    if(abs(scale - int(scale)) >= 1e-6) scheme.rescaleAndEqual(cipher);
    
    Ciphertext* bx = scheme_algo.chebyshev_tree_pool[0];
    Ciphertext* ax = &cipher;
    Ciphertext* temp = scheme_algo.chebyshev_tree_pool[1];
    temp->l = cipher.l;
    temp->scale = cipher.scale;
    bx->l = cipher.l;
    bx->scale = cipher.scale;

    NTL::RR target_scale = cipher.scale;
    cout<<"ax.l: "<<ax->l<<endl;

    // b = (3 - k0*a)
    ax->scale = ax->scale / K_sqrt_inv[0];
    scheme.constSub(*bx, *ax, 3);
    ax->scale = ax->scale * K_sqrt_inv[0];
    // a = a * (3 - ki*a)**2
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);
    scheme.multAndEqual_23(*ax, *bx);
    scheme.rescaleAndEqual(*ax);    
    // a = k0 / 4 * a * (3 - k0*a)**2
    ax->scale = ax->scale / (K_sqrt_inv[0] / 4);
    // b = ki**0.5 / 2 * b * (3 - k0*a)
    bx->scale = bx->scale / ((pow(K_sqrt_inv[0], 0.5) / 2));

    for(int i = 1; i < K_sqrt_inv.size(); i++){
        // temp = (3 - ki*a)
        ax->scale = ax->scale / K_sqrt_inv[i];
        scheme.constSub(*temp, *ax, 3);
        ax->scale = ax->scale * K_sqrt_inv[i];

        // a = a * (3 - ki*a)**2
        scheme.multAndEqual_23(*ax, *temp);
        scheme.rescaleAndEqual(*ax);
        scheme.multAndEqual_23(*ax, *temp);
        scheme.rescaleAndEqual(*ax);
        // a = ki / 4 * a * (3 - ki*a)**2
        ax->scale = ax->scale / (K_sqrt_inv[i] / 4);

        // b = ki**0.5 / 2 * b * (3 - ki*a)
        scheme.multAndEqual_23(*bx, *temp);
        scheme.rescaleAndEqual(*bx);
        bx->scale = bx->scale / ((pow(K_sqrt_inv[i], 0.5) / 2));
    }
    double tt = to_double(target_scale / bx->scale / pow(upper_bound, 0.5));
    bx->scale *= target_scale / bx->scale;
    // scheme.multConstAndEqual(*bx, tt);
    // scheme.rescaleAndEqual(*bx);
    cipher = *bx;
}

void Attention::evalSoftMax(vector<Ciphertext*>& cipher_P)
{
    if(cipher_P.size() != token_len / d){
        printf("Error: cipher_P size must be equal to token_len / d\n");
        return;
    }

    for(int i = 0; i < token_len / d; i++){
        scheme.addConstAndEqual(*cipher_P[i], -softmax_x_max);
        evalExp(*cipher_P[i]);
        // Bootstrapping here
    }
    // cout << "level after evalExp " << (*cipher_P[0]).l << endl;

    for(int i = 0; i < token_len / d; i++){
        *nonlieanr_buffer[1] = *cipher_P[i];

        // reduce sum and repeat
        // reduce sum
        for(int j = log2(d) - 1; j >= 0; j--){
            scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], 1 << j);
            // printf("reduce sum id: %d\n", 1<<i);
        }
        scheme.multConstAndEqual(*nonlieanr_buffer[1], *column_mask_reduce[0]);
        scheme.rescaleAndEqual(*nonlieanr_buffer[1]);

        // repeate
        for(int j = 0; j < log2(d); j++){
            scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], context.slots - (1 << j));
        }
        if(i == 0){
            *nonlieanr_buffer[0] = *nonlieanr_buffer[1];
        } else {
            scheme.addAndEqual(*nonlieanr_buffer[0], *nonlieanr_buffer[1]);
        }
    }

    // cout << "level after reduce repeat *nonlieanr_buffer[0] " << (*nonlieanr_buffer[0]).l << endl;
    // cout << "level after reduce repeat *nonlieanr_buffer[1] " << (*nonlieanr_buffer[1]).l << endl;

    // sum{exp(x)} < n / 4 ??
    evalInv(*nonlieanr_buffer[0], token_len / 4);

    // cout << "level after evalInv " << (*nonlieanr_buffer[0]).l << endl;

    for(int i = 0; i < token_len / d; i++){
        // exp(x) / sum(exp(x))
        scheme.multAndEqual_23(*cipher_P[i], *nonlieanr_buffer[0]);
        scheme.rescaleAndEqual(*cipher_P[i]);
    }
}

// use Fast and Accurate Homomorphic Softmax Evaluation method
void Attention::FASHE_evalSoftMax(vector<Ciphertext*>& cipher_P, SecretKey& sk)
{
    int iter_time = 2;
    double lambda_y[2] = {20.1};
    if(cipher_P.size() != token_len / d){
        printf("Error: cipher_P size must be equal to token_len / d\n");
        return;
    }
    for(int i = 0; i < token_len / d; i++){
        scheme.addConstAndEqual(*cipher_P[i], -(softmax_x_max*(1<<iter_time)));
        scheme.multConstAndEqual(*cipher_P[i], 1./softmax_x_max/(1<<iter_time));
        scheme.rescaleAndEqual(*cipher_P[i]);
        scheme.addConstAndEqual(*cipher_P[i], 1);

        NTL::RR target_scale = (*cipher_P[i]).scale;

        scheme_algo.evalPolynomialChebyshev(*cipher_P[i], target_scale, exp_cheby_poly_pool);
    }
    scheme.decrypt_display(sk, *cipher_P[0], "After exp:");
    for(int i = 0; i < token_len / d; i++){
        for (int t = 0; t < iter_time; t++){
            if (t < iter_time - 1 ){
                // mul lambda and then square
                scheme.multConstAndEqual(*cipher_P[i], lambda_y[t]);
                scheme.rescaleAndEqual(*cipher_P[i]);
                scheme.multAndEqual_23(*cipher_P[i], *cipher_P[i]);
                scheme.rescaleAndEqual(*cipher_P[i]);
                scheme.decrypt_display(sk, *cipher_P[i], "After mul lambda and square:");
            }
            else {
                *nonlieanr_buffer[1] = *cipher_P[i];
                scheme.multAndEqual_23(*nonlieanr_buffer[1], *nonlieanr_buffer[1]);
                scheme.rescaleAndEqual(*nonlieanr_buffer[1]);
                // reduce sum and repeat
                // reduce sum
                for(int j = log2(d) - 1; j >= 0; j--){
                    scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], 1 << j);
                    // printf("reduce sum id: %d\n", 1<<i);
                }
                scheme.multConstAndEqual(*nonlieanr_buffer[1], *column_mask_reduce[0]);
                scheme.rescaleAndEqual(*nonlieanr_buffer[1]);

                // repeate
                for(int j = 0; j < log2(d); j++){
                    scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], context.slots - (1 << j));
                }
                if(i == 0){
                    *nonlieanr_buffer[0] = *nonlieanr_buffer[1];
                } else {
                    scheme.addAndEqual(*nonlieanr_buffer[0], *nonlieanr_buffer[1]);
                }
            }
        }
    }
    scheme.decrypt_display(sk, *nonlieanr_buffer[0], "Sum before inv:");
    evalSqrtInv(*nonlieanr_buffer[0], sk, 1.1);
    scheme.decrypt_display(sk, *nonlieanr_buffer[0], "After inv:");
    for(int i = 0; i < token_len / d; i++){
        scheme.decrypt_display(sk, *cipher_P[i], "Before final mult:");
        scheme.multAndEqual_23(*cipher_P[i], *nonlieanr_buffer[0]);
        scheme.rescaleAndEqual(*cipher_P[i]);
        scheme.decrypt_display(sk, *cipher_P[i], "After final mult:");
        scheme.multAndEqual_23(*cipher_P[i], *cipher_P[i]);
        scheme.rescaleAndEqual(*cipher_P[i]);
    }
}

void Attention::evalSoftMax_phase1_iter(vector<Ciphertext*>& cipher_P, int iter)
{
    cout << "level at begin " << (*cipher_P[0]).l << endl;
    if(cipher_P.size() != token_len / d){
        printf("Error: cipher_P size must be equal to token_len / d\n");
        return;
    }

    for(int i = 0; i < token_len / d; i++){
        scheme.addConstAndEqual(*cipher_P[i], -(softmax_x_max*(1<<iter)));
        evalExp_iter(*cipher_P[i], iter);
        // Bootstrapping here
    }
}

void Attention::evalSoftMax_phase1(vector<Ciphertext*>& cipher_P)
{
    cout << "level at begin " << (*cipher_P[0]).l << endl;
    if(cipher_P.size() != token_len / d){
        printf("Error: cipher_P size must be equal to token_len / d\n");
        return;
    }

    for(int i = 0; i < token_len / d; i++){
        scheme.addConstAndEqual(*cipher_P[i], -softmax_x_max);
        evalExp(*cipher_P[i]);
        // Bootstrapping here
    }
}

void Attention::evalSoftMax_phase1_mul_sqrtd(vector<Ciphertext*>& cipher_P)
{
    cout << "level at begin " << (*cipher_P[0]).l << endl;
    if(cipher_P.size() != token_len / d){
        printf("Error: cipher_P size must be equal to token_len / d\n");
        return;
    }

    for(int i = 0; i < token_len / d; i++){
        // scheme.addConstAndEqual(*cipher_P[i], -softmax_x_max);
        // scheme.multConstAndEqual(*cipher_P[i], 1./softmax_x_max);
        scheme.multConstAndEqual(*cipher_P[i], 1./softmax_x_max/sqrt(d));
        scheme.addConstAndEqual(*cipher_P[i], -1);
        scheme.rescaleAndEqual(*cipher_P[i]);
        scheme.addConstAndEqual(*cipher_P[i], 1);

        NTL::RR target_scale = (*cipher_P[i]).scale;

        // First mapping: x \in [-10, 0] -> [-1, 1]
        // Evaluate the Chebyshev polynomial for exp(5(x-1))
        scheme_algo.evalPolynomialChebyshev(*cipher_P[i], target_scale, exp_cheby_poly_pool);
        // Bootstrapping here
    }
}

void Attention::evalSoftMax_phase2(vector<Ciphertext*>& cipher_P, SecretKey& sk)
{
    for(int i = 0; i < token_len / d; i++){
        *nonlieanr_buffer[1] = *cipher_P[i];

        // reduce sum and repeat
        // reduce sum
        for(int j = log2(d) - 1; j >= 0; j--){
            scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], 1 << j);
            // printf("reduce sum id: %d\n", 1<<i);
        }
        scheme.multConstAndEqual(*nonlieanr_buffer[1], *column_mask_reduce[0]);
        scheme.rescaleAndEqual(*nonlieanr_buffer[1]);

        // repeate
        for(int j = 0; j < log2(d); j++){
            scheme.leftRotateAddSelf_23(*nonlieanr_buffer[1], context.slots - (1 << j));
        }
        if(i == 0){
            *nonlieanr_buffer[0] = *nonlieanr_buffer[1];
        } else {
            scheme.addAndEqual(*nonlieanr_buffer[0], *nonlieanr_buffer[1]);
        }
    }
    cout << "before evalInv (*nonlieanr_buffer[0]).scale " << (*nonlieanr_buffer[0]).scale << endl;
    scheme.decrypt_display(sk, *nonlieanr_buffer[0], "Sum before inv:");
    // sum{exp(x)} < n / 4 ??
    evalInv(*nonlieanr_buffer[0], 1);
    cout << "after evalInv (*nonlieanr_buffer[0]).scale " << (*nonlieanr_buffer[0]).scale << endl;
    cout << "after evalInv (*cipher_P[0]).scale " << (*cipher_P[0]).scale << endl;
    for(int i = 0; i < token_len / d; i++){
        // exp(x) / sum(exp(x))
        scheme.multAndEqual_23(*cipher_P[i], *nonlieanr_buffer[0]);
        scheme.rescaleAndEqual(*cipher_P[i]);
    }
}

// LayerNorm function: y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
void Attention::LayerNorm(vector<Ciphertext*>& cipher, SecretKey& sk)
{
    int slots = context.slots;
    int column_num = slots / token_len;
    Ciphertext* sum_x = scheme_algo.chebyshev_tree_pool[3];
    vector<Ciphertext*> tmp_cipher;
    for (int i = 0; i < cipher.size(); i++){
        tmp_cipher.push_back(scheme_algo.chebyshev_tree_pool[i+3]);
    }
    // compute sum_x
    for (int i = 0; i < cipher.size(); i++){
        if(i==0)*sum_x = *cipher[i];
        else scheme.addAndEqual(*sum_x, *cipher[i]);
    }
    // scheme.decrypt_display(sk, *sum_x, "compute sum_x");

    // reduce sum and repeat
    // reduce sum
    for(int j = log2(column_num) - 1; j >= 0; j--){
        scheme.leftRotateAddSelf_23(*sum_x, 1 << j);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    scheme.multConstAndEqual(*sum_x, *column_mask_reduce[1]);
    scheme.rescaleAndEqual(*sum_x);

    // repeate
    for(int j = 0; j < log2(column_num); j++){
        scheme.leftRotateAddSelf_23(*sum_x, slots - (1 << j));
    }
    // scheme.decrypt_display(sk, *sum_x, "repeate");

    // compute n*x
    for (int i = 0; i < cipher.size(); i++){
        scheme.multConstAndEqual(*cipher[i], 1.0*column_num*cipher.size());
    }
    // compute n*x - sum_x
    for (int i = 0; i < cipher.size(); i++){
        scheme.subAndEqual(*cipher[i], *sum_x);
    }
    // square
    for (int i = 0; i < cipher.size(); i++){
        *tmp_cipher[i] = *cipher[i];
        scheme.multAndEqual_23(*tmp_cipher[i], *tmp_cipher[i]);
        scheme.rescaleAndEqual(*tmp_cipher[i]);
    }
    // compute sum
    for (int i = 0; i < tmp_cipher.size(); i++){
        if(i==0)*sum_x = *tmp_cipher[i];
        else scheme.addAndEqual(*sum_x, *tmp_cipher[i]);
    }

    // reduce sum and repeat
    // reduce sum
    for(int j = log2(column_num) - 1; j >= 0; j--){
        scheme.leftRotateAddSelf_23(*sum_x, 1 << j);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    scheme.multConstAndEqual(*sum_x, *column_mask_reduce[1]);
    scheme.rescaleAndEqual(*sum_x);

    // repeate
    for(int j = 0; j < log2(column_num); j++){
        scheme.leftRotateAddSelf_23(*sum_x, slots - (1 << j));
    }
    // add epsilon to avoid divide zero
    // scheme.addConstAndEqual(*sum_x, 1e-5);

    scheme.decrypt_display(sk, *sum_x, "before evalSqrtInv");
    // evalSqrtInv
    evalSqrtInv(*sum_x, sk, 12300);

    // mul
    for (int i = 0; i < cipher.size(); i++){
        scheme.multAndEqual_23(*cipher[i], *sum_x);
        scheme.rescaleAndEqual(*cipher[i]);
    }
    // (*gamma * sqrt(n)+ beta) has not been implemented in this function yet; it will be implemented outside of it.
    // n = column_num * cipher.size(). in bert n=768

}

// LayerNorm function: y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
void Attention::LayerNorm_Bert(vector<Ciphertext*>& cipher, SecretKey& sk)
{
    cout << "LayerNorm_Bert input scale " << cipher[0]->scale << endl;
    int slots = context.slots;
    int column_num = slots / token_len;
    Ciphertext* sum_x = scheme_algo.chebyshev_tree_pool[3];
    vector<Ciphertext*> tmp_cipher;
    for (int i = 0; i < cipher.size(); i++){
        tmp_cipher.push_back(scheme_algo.chebyshev_tree_pool[i+4]);
    }
    // compute sum_x
    for (int i = 0; i < cipher.size(); i++){
        if(i==0)*sum_x = *cipher[i];
        else scheme.addAndEqual(*sum_x, *cipher[i]);
    }
    cout << "LayerNorm_Bert after sum_x scale " << sum_x->scale << endl;
    

    // reduce sum and repeat
    // reduce sum
    for(int j = log2(column_num) - 1; j >= 0; j--){
        scheme.leftRotateAddSelf_23(*sum_x, 1 << j);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    scheme.multConstAndEqual(*sum_x, *column_mask_reduce[1]);
    scheme.rescaleAndEqual(*sum_x);
    cout << "LayerNorm_Bert after reduce sum scale " << sum_x->scale << endl;
    scheme.decrypt_display(sk, *sum_x, "compute sum_x");

    // repeate
    for(int j = 0; j < log2(column_num); j++){
        scheme.leftRotateAddSelf_23(*sum_x, slots - (1 << j));
    }
    scheme.decrypt_display(sk, *sum_x, "repeate");

    // compute n*x
    for (int i = 0; i < cipher.size(); i++){
        scheme.multConstAndEqual(*cipher[i], 1.0*column_num*cipher.size());
    }
    // compute n*x - sum_x
    for (int i = 0; i < cipher.size(); i++){
        scheme.subAndEqual(*cipher[i], *sum_x);
    }
    scheme.decrypt_display(sk, *cipher[0], "-sum_x");
    // square
    for (int i = 0; i < cipher.size(); i++){
        *tmp_cipher[i] = *cipher[i];
        scheme.multAndEqual_23(*tmp_cipher[i], *tmp_cipher[i]);
        scheme.rescaleAndEqual(*tmp_cipher[i]);
    }
    cout << "LayerNorm_Bert after square sum_x scale " << sum_x->scale << endl;
    cout << "LayerNorm_Bert after square tmp_cipher scale " << tmp_cipher[0]->scale << endl;
    // compute sum
    for (int i = 0; i < tmp_cipher.size(); i++){
        if(i==0)*sum_x = *tmp_cipher[i];
        else scheme.addAndEqual(*sum_x, *tmp_cipher[i]);
    }

    // reduce sum and repeat
    // reduce sum
    for(int j = log2(column_num) - 1; j >= 0; j--){
        scheme.leftRotateAddSelf_23(*sum_x, 1 << j);
        // printf("reduce sum id: %d\n", 1<<i);
    }
    cout << "LayerNorm_Bert after reduce sum phase1 scale " << sum_x->scale << endl;
    scheme.multConstAndEqual(*sum_x, *column_mask_reduce[1]);
    scheme.rescaleAndEqual(*sum_x);
    cout << "LayerNorm_Bert after reduce sum phase2 scale " << sum_x->scale << endl;

    // repeate
    for(int j = 0; j < log2(column_num); j++){
        scheme.leftRotateAddSelf_23(*sum_x, slots - (1 << j));
    }
    // add epsilon to avoid divide zero
    // scheme.addConstAndEqual(*sum_x, 1e-5);

    cout << "LayerNorm_Bert before sqrtInv sum_x scale " << sum_x->scale << endl;
    scheme.decrypt_display(sk, *sum_x, "before SqrtInv");
    // evalSqrtInv
    evalSqrtInv_without_mul_tt(*sum_x, sk, 2e+06);
    cout << "LayerNorm_Bert after sqrtInv sum_x scale " << sum_x->scale << endl;
    cout << "LayerNorm_Bert after sqrtInv cipher[0] scale " << cipher[0]->scale << endl;
    scheme.decrypt_display(sk, *sum_x, "after SqrtInv");

    // mul
    for (int i = 0; i < cipher.size(); i++){
        scheme.multAndEqual_23(*cipher[i], *sum_x);
        scheme.rescaleAndEqual(*cipher[i]);
    }
    // (*gamma * sqrt(n)+ beta) has not been implemented in this function yet; it will be implemented outside of it.
    // n = column_num * cipher.size(). in bert n=768
    cout << "LayerNorm_Bert finish scale " << cipher[0]->scale << endl;

}

void Attention::CCMM_QK(Ciphertext& Q, Ciphertext& K, Ciphertext& O1, Ciphertext& O2)
{
    cout << "const scale" << column_mask_ccmm[0]->scale << endl;
    for (int i = 0; i < log2(d)+1; i++){
        tmpcipher_buffer[i] = scheme_algo.chebyshev_tree_pool[i];
    }

    int slots = context.slots;
    int column_num = slots / token_len;

    if(d * token_len > slots){
        printf("Error QKV matrix must in one cipher\n");
    }
    *leafnode = K;
    Recursive_CCMM_reduce(Q, K, 0, log2(d), 0, column_num, tmpcipher_buffer);
    
    O1 = **tmpcipher_buffer;
    
    O2 = K;
    leftRotateAndEqual_23(O2, (int)(slots/2));
    *leafnode = O2;
    
    Recursive_CCMM_reduce(Q, O2, 0, log2(d), 0, column_num, tmpcipher_buffer);
    O2 = **tmpcipher_buffer;
}

void Attention::Recursive_CCMM_reduce(Ciphertext& Q, Ciphertext& K, int layer, int max_layer, int seq, int column_num, Ciphertext** cipher_stack)
{
    int slots = context.slots;
    if(layer == max_layer){
        **cipher_stack = *leafnode;
        if(seq){
            leftRotateAndEqual_23(**cipher_stack, column_num);
            *leafnode = **cipher_stack;
        }
        multAndEqual_23(**cipher_stack, Q);
        // if((**cipher_stack).scale > need_rescale)scheme.rescaleAndEqual(**cipher_stack);
        scheme.rescaleAndEqual(**cipher_stack);
        if (seq & 1)leftRotateAddSelf_23(**cipher_stack, slots - 1);
        else leftRotateAddSelf_23(**cipher_stack, 1);
        // scheme.decrypt_display(sk, **cipher_stack, "before mask");
        // cout << "before mult " << (**cipher_stack).scale << endl;
        scheme.multConstAndEqual(**cipher_stack, *column_mask_ccmm[(seq & 1)]);
        // cout << "after mult " << (**cipher_stack).scale << endl;
        // if((**cipher_stack).scale > need_rescale)scheme.rescaleAndEqual(**cipher_stack);
        scheme.rescaleAndEqual(**cipher_stack);
        return;
    }
    Recursive_CCMM_reduce(Q, K, layer + 1, max_layer, seq * 2, column_num, cipher_stack);
    Recursive_CCMM_reduce(Q, K, layer + 1, max_layer, seq * 2 + 1, column_num, cipher_stack+1);
    scheme.addAndEqual(**cipher_stack, **(cipher_stack+1));
    // if (layer == 5) scheme.decrypt_display(sk, **cipher_stack, "");
    if (layer > 0){
        if (seq & 1) leftRotateAddSelf_23(**cipher_stack, slots - (1 << (max_layer - layer)));
        else leftRotateAddSelf_23(**cipher_stack, (1 << (max_layer - layer)));
        // cout << "before mult " << (**cipher_stack).scale << endl;
        scheme.multConstAndEqual(**cipher_stack, *column_mask_ccmm[(seq & 1) + 2*(max_layer - layer)]);
        // cout << "after mult " << (**cipher_stack).scale << endl;
        // if((**cipher_stack).scale > need_rescale)scheme.rescaleAndEqual(**cipher_stack);
        scheme.rescaleAndEqual(**cipher_stack);
        // if (layer == max_layer - 1)scheme.decrypt_display(sk, **cipher_stack, "cipher_stack");
    }
    return;
}

void Attention::CCMM_QK_splited_heads(vector<Ciphertext *>& Q, vector<Ciphertext *>& K,vector<Ciphertext *>& O, int column_each_head)
{
    if (token_len % column_each_head != 0){
        printf("wrong splitd");
    }
    int yy = token_len / column_each_head;

    int q_len = Q.size();
    int k_len = K.size();
    int o_len = O.size();
    if (q_len != k_len){
        printf("wrong q_len and k_len");
    }

    for (int i = 0; i < log2(column_each_head)+1; i++){
        tmpcipher_buffer[i] = scheme_algo.chebyshev_tree_pool[i];
    }

    int slots = context.slots;
    int column_num = slots / token_len;

    if(d * token_len > slots){
        printf("Error QKV matrix must in one cipher\n");
    }

    *O[1] = *K[0];
    *leafnode = *K[0];
    Recursive_CCMM_reduce(*Q[0], *K[0], 0, log2(column_each_head), 0, column_num, tmpcipher_buffer);
    *O[0] = **tmpcipher_buffer;
    for (int time = 1; time < yy; time++){
        leftRotateAndEqual_23(*O[time], column_each_head*column_num);
        if(time+1<yy)*O[time+1] = *O[time];
        *leafnode = *O[time];
        
        Recursive_CCMM_reduce(*Q[0], *O[time], 0, log2(column_each_head), 0, column_num, tmpcipher_buffer);
        *O[time] = **tmpcipher_buffer;
    }

    
    for (int q_sed = 1; q_sed < q_len; q_sed++){
        *tmp_shift_K = *K[q_sed];
        *leafnode = *K[q_sed];
        Recursive_CCMM_reduce(*Q[q_sed], *K[q_sed], 0, log2(column_each_head), 0, column_num, tmpcipher_buffer);
        scheme.addAndEqual(*O[0], **tmpcipher_buffer);
        for (int time = 1; time < yy; time++){
            leftRotateAndEqual_23(*tmp_shift_K, column_each_head*column_num);
            *leafnode = *tmp_shift_K;
            
            Recursive_CCMM_reduce(*Q[q_sed], *tmp_shift_K, 0, log2(column_each_head), 0, column_num, tmpcipher_buffer);
            scheme.addAndEqual(*O[time], **tmpcipher_buffer);
        }
    }
}

void Attention::TauAndEqual(Ciphertext& A)
{
    Ciphertext* tmp_result_bs;
    Ciphertext* tmp_result_inside_bs;
    int diag_num = token_len * head_num;
    int baby_steps = 4;
    int giant_steps = 16;
    for (int i = 0; i < baby_steps; i++){
        tmpcipher_buffer[i] = scheme_algo.chebyshev_tree_pool[i];
    }
    tmp_result_bs = scheme_algo.chebyshev_tree_pool[baby_steps+1];
    tmp_result_inside_bs = scheme_algo.chebyshev_tree_pool[baby_steps+2];

    for (int i = 0; i < baby_steps; i++){
        *tmpcipher_buffer[i] = A;
        if(i>0)leftRotateAndEqual_23(*tmpcipher_buffer[i], (context.slots / token_len) * i);
    }

    for (int j = 0; j < giant_steps; j++){
        for (int i = 0; i < baby_steps; i++){
            if(i==0){
                *tmp_result_bs = *tmpcipher_buffer[i];
                scheme.multConstAndEqual(*tmp_result_bs, plain_tau_diag[baby_steps*j+i]);
                scheme.rescaleAndEqual(*tmp_result_bs);
            }
            else{
                *tmp_result_inside_bs = *tmpcipher_buffer[i];
                scheme.multConstAndEqual(*tmp_result_inside_bs, plain_tau_diag[baby_steps*j+i]);
                scheme.rescaleAndEqual(*tmp_result_inside_bs);
                scheme.addAndEqual(*tmp_result_bs, *tmp_result_inside_bs);
            }
        }
        if(j==0){
            A = *tmp_result_bs;
        }
        else{
            leftRotateAndEqual_23(*tmp_result_bs, (context.slots / token_len) * baby_steps * j);
            scheme.addAndEqual(A, *tmp_result_bs);
        }
    }
}

void Attention::CCMM_V(Ciphertext& sigma_O1, Ciphertext& sigma_O2, Ciphertext& tau_V, Ciphertext& O){
    int slots = context.slots;
    int column_num = slots / token_len;
    Ciphertext* mat1 = scheme_algo.chebyshev_tree_pool[mulV_gs];
    Ciphertext* mat2 = scheme_algo.chebyshev_tree_pool[mulV_gs + 1];
    Ciphertext* mat3 = scheme_algo.chebyshev_tree_pool[mulV_gs + 2];
    Ciphertext* mat4 = scheme_algo.chebyshev_tree_pool[mulV_gs + 3];
    Ciphertext* temp_rot1 = scheme_algo.chebyshev_tree_pool[mulV_gs + 4];
    Ciphertext* temp_rot2 = scheme_algo.chebyshev_tree_pool[mulV_gs + 5];
    Ciphertext* temp_mul1 = scheme_algo.chebyshev_tree_pool[mulV_gs + 6];
    Ciphertext* temp_mul2 = scheme_algo.chebyshev_tree_pool[mulV_gs + 7];
    Ciphertext* temp_res = scheme_algo.chebyshev_tree_pool[mulV_gs + 8];

    // gs
    *(mulV_gs_res[0]) = tau_V;
    for (int j = 1; j < mulV_gs; j++){
        *mulV_gs_res[j] = *mulV_gs_res[j-1];
        leftRotateAndEqual_23(*mulV_gs_res[j], mulV_bs*column_num);
    }

    *temp_rot1 = sigma_O1;
    *temp_rot2 = sigma_O2;
    // bs
    int offset = 0;
    for (int i = 0; i < mulV_bs; i++){
        if (i > 0){
            leftRotateAndEqual_23(*temp_rot1, slots - (column_num - 1));
            leftRotateAndEqual_23(*temp_rot2, slots - (column_num - 1));
        }
        *mat1 = *temp_rot1;
        *mat3 = *temp_rot2;
        for (int j = 0; j < mulV_gs/2; j++){
            offset = i + j * mulV_bs;
            if (j > 0){
                leftRotateAndEqual_23(*mat1, mulV_bs);
                leftRotateAndEqual_23(*mat3, mulV_bs);
            }
            *mat4 = *mat1;
            leftRotateAndEqual_23(*mat4, slots - d);
            *mat2 = *mat3;
            leftRotateAndEqual_23(*mat2, slots - d);

            *temp_mul1 = *mat1;
            *temp_mul2 = *mat2;
            
            if (offset){
                scheme.multConstAndEqual(*temp_mul1, mask_ccmm_left[offset]);
                scheme.rescaleAndEqual(*temp_mul1);
                
                scheme.multConstAndEqual(*temp_mul2, mask_ccmm_right[offset]);
                scheme.rescaleAndEqual(*temp_mul2);
                scheme.addAndEqual(*temp_mul1, *temp_mul2);
            }
            multAndEqual_23(*temp_mul1, *mulV_gs_res[j]);
            scheme.rescaleAndEqual(*temp_mul1);
            if (j==0) *temp_res = *temp_mul1;
            else scheme.addAndEqual(*temp_res, *temp_mul1);


            *temp_mul1 = *mat3;
            *temp_mul2 = *mat4;
            if (offset){
                scheme.multConstAndEqual(*temp_mul1, mask_ccmm_left[offset]);
                scheme.rescaleAndEqual(*temp_mul1);
                scheme.multConstAndEqual(*temp_mul2, mask_ccmm_right[offset]);
                scheme.rescaleAndEqual(*temp_mul2);
                scheme.addAndEqual(*temp_mul1, *temp_mul2);
            }
            multAndEqual_23(*temp_mul1, *mulV_gs_res[j+8]);
            scheme.rescaleAndEqual(*temp_mul1);
            scheme.addAndEqual(*temp_res, *temp_mul1);
            
        }
        if (i == 0){
            O = *temp_res;
        }
        else{
            leftRotateAndEqual_23(*temp_res, column_num * i);
            scheme.addAndEqual(O, *temp_res);
        }
    }
}
void Attention::leftRotateAndEqual_23(Ciphertext& A, int shift){
    scheme.leftRotateAndEqual_23(A, shift);
    KS_SV += 1;
}
void Attention::leftRotateAddSelf_23(Ciphertext& A, int shift){
    scheme.leftRotateAddSelf_23(A, shift);
    KS_SV += 1;
}
void Attention::multAndEqual_23(Ciphertext& A, Ciphertext& B){
    scheme.multAndEqual_23(A, B);
    KS_SV += 1;
}
