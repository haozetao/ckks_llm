#pragma once

#include "Bootstrapping_encoding.h"

#include <utility>


EncodingMatrix::EncodingMatrix(SecretKey &secretKey, Scheme_23 &scheme, int levelS2C, int levelC2S, int is_STC_first) : 
    secretKey(secretKey), scheme(scheme), levelBudgetDec(levelS2C), levelBudgetEnc(levelC2S), is_STC_first(is_STC_first)
{
    long slots = scheme.context.slots;
    vector<complex<double>> A = ComputeRoots(slots << 1, true);

    vector<uint32_tt> rotGroup(slots);
    rotateTemp_host = vector<cuDoubleComplex>(slots);
    cudaMalloc(&rotateTemp_device, sizeof(cuDoubleComplex) * slots);

    rotGroup[0] = 1;
    for (int i = 1; i < slots; i++)
    {
        rotGroup[i] = rotGroup[i - 1] * 5;
        rotGroup[i] &= (slots << 2) - 1;
    }

    m_paramsEnc = GetCollapsedFFTParams(slots, levelBudgetEnc, 0);
    m_paramsDec = GetCollapsedFFTParams(slots, levelBudgetDec, 0);

    /******************************************malloc double hoisting buffer**************************************** */
    int N = scheme.context.N;
    int L = scheme.context.L;

    int p_num = scheme.context.p_num;
    int q_num = scheme.context.q_num;
    int t_num = scheme.context.t_num;
    int Ri_blockNum = scheme.context.Ri_blockNum;
    int Qj_blockNum = scheme.context.Qj_blockNum;

    int gs = max(m_paramsEnc[5], m_paramsDec[5]);
    gs = max(gs, max(m_paramsEnc[8], m_paramsDec[8]));
    int bs = max(m_paramsEnc[4], m_paramsDec[4]);
    bs = max(bs, max(m_paramsEnc[7], m_paramsDec[7]));

    cout<<"m_paramsEnc gs: "<<m_paramsEnc[5]<<endl;
    cout<<"m_paramsEnc gsRem: "<<m_paramsEnc[8]<<endl;
    cout<<"m_paramsDec gs: "<<m_paramsDec[5]<<endl;
    cout<<"m_paramsDec gsRem: "<<m_paramsDec[8]<<endl;

    cudaMalloc(&cipher_buffer_PQ, sizeof(uint64_tt) * N * (p_num + q_num) * 2);
    cudaMalloc(&gs_cipher_T, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 * gs);
    cudaMalloc(&PQ_to_T_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);
    cudaMalloc(&T_to_PQ_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);

    cudaMalloc(&bs_cipher_T, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 * bs);
    /******************************************malloc double hoisting buffer**************************************** */

    // for(int i = 0; i < gs; i++)
    // {
    //     fastRotation_Enc_gs.push_back(new Ciphertext(gs_cipher_PQl, N, L, L, slots, NTL::RR(scheme.context.precision)));
    // }

    // modup -> cts -> evalmod -> stc
    // m_U0hatTPreFFT = EvalCoeffsToSlotsPrecompute(A, rotGroup, false, NTL::RR(scheme.context.precision), scheme.context.q_num - 1);
    // m_U0PreFFT     = EvalSlotsToCoeffsPrecompute(A, rotGroup, false, NTL::RR(scheme.context.precision), scheme.context.q_num - levelBudgetEnc - 2 - 6);

    if(levelBudgetEnc == 4)
    {
        is_sqrt_rescale = 1;
        rescale_times = 3;
    }
    else {
        is_sqrt_rescale = 0;
        rescale_times = levelBudgetEnc;
    }
    
    if(is_sqrt_rescale)
    {
        sqrt_rescale = NTL::RR(pow(scheme.context.precision, rescale_times/4.));
        // sqrt_rescale = NTL::RR(pow(scheme.context.precision, 1./2));

        cout<<"sqrt_rescale: "<<sqrt_rescale<<endl;
    }
    else
    {
        sqrt_rescale = NTL::RR(scheme.context.precision);
    }
    m_U0PreFFT     = EvalSlotsToCoeffsPrecompute(A, rotGroup, false, sqrt_rescale, scheme.context.q_num - 1);
    m_U0hatTPreFFT = EvalCoeffsToSlotsPrecompute(A, rotGroup, false, sqrt_rescale, scheme.context.q_num - 1 - 3);


    Precompute_rot_in_out_C2S();
    Precompute_rot_in_out_S2C();

    rotKey_pointer = vector<uint64_tt*>(gs, nullptr);
    cudaMalloc(&rotKey_pointer_device, sizeof(uint64_tt*) * gs);

    rotSlots = vector<int>(gs, -1);
    cudaMalloc(&rotSlots_device, sizeof(int) * gs);

    plaintextT_pointer = vector<uint64_tt*>(gs * bs, nullptr);
    cudaMalloc(&plaintextT_pointer_device, sizeof(uint64_tt*) * gs * bs);
    accNum_vec = vector<int>(bs, 0);
    cudaMalloc(&accNum_vec_device, sizeof(int) * bs);

    cudaStreamCreate(&stream_prefetch);
}

void EncodingMatrix::addC2SKey()
{
    int stop = -1;
    int flagRem = 0;
    int BASE_NUM_LEVELS_TO_DROP = 1;
    long N = scheme.context.N;
    long M = scheme.context.M;
    long slots = scheme.context.slots;

    long K = scheme.context.K;
    long logN = scheme.context.logN;

    int levelBudget = m_paramsEnc[0];
    int layersCollapse = m_paramsEnc[1];
    int remCollapse = m_paramsEnc[2];
    int numRotations = m_paramsEnc[3];
    int b = m_paramsEnc[4];
    int g = m_paramsEnc[5];
    int numRotationsRem = m_paramsEnc[6];
    int bRem = m_paramsEnc[7];
    int gRem = m_paramsEnc[8];

    if (remCollapse != 0)
    {
        stop = 0;
        flagRem = 1;
    }

    // input cipher is on Ql
    for(int s = levelBudget - 1; s > stop; s--)
    {
        cout<<"rot_in_C2S: "<<endl;
        for(int j = 0; j < g; j++)
        {
            cout<<rot_in_C2S[s][j]<<", ";
            if (rot_in_C2S[s][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_in_C2S[s][j]);
                rotIdx_C2S.push_back(rot_in_C2S[s][j]);
            }
        }
        cout<<endl;

        cout<<"rot_out_C2S: "<<endl;
        for(int j = 0; j < b; j++)
        {
            cout<<rot_out_C2S[s][j]<<", ";
            if (rot_out_C2S[s][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_out_C2S[s][j]);
                rotIdx_C2S.push_back(rot_out_C2S[s][j]);
            }  
        }
        cout<<endl;
    }

    if (flagRem)
    {
        cout<<"rot_in_C2S rem: "<<endl;
        for (int32_t j = 0; j < gRem; j++)
        {
            cout<<rot_in_C2S[stop][j]<<", ";
            if (rot_in_C2S[stop][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_in_C2S[stop][j]);
                rotIdx_C2S.push_back(rot_in_C2S[stop][j]);
            }
        }
        cout<<endl;

        cout<<"rot_out_C2S rem: "<<endl;
        for(int j = 0; j < bRem; j++)
        {
            cout<<rot_out_C2S[stop][j]<<", ";
            if (rot_out_C2S[stop][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_out_C2S[stop][j]);
                rotIdx_C2S.push_back(rot_out_C2S[stop][j]);
            }
        }
        cout<<endl;
    }
}

void EncodingMatrix::addS2CKey()
{
    int BASE_NUM_LEVELS_TO_DROP = 1;
    int levelBudget = m_paramsDec[0];
    int layersCollapse = m_paramsDec[1];
    int remCollapse = m_paramsDec[2];
    int numRotations = m_paramsDec[3];
    int b = m_paramsDec[4];
    int g = m_paramsDec[5];
    int numRotationsRem = m_paramsDec[6];
    int bRem = m_paramsDec[7];
    int gRem = m_paramsDec[8];
    int flagRem = 0;
    long N = scheme.context.N;
    long logN = scheme.context.logN;
    long M = scheme.context.M;
    long K = scheme.context.K;
    long slots = N / 2;
    if (remCollapse != 0)
    {
        flagRem = 1;
    }

    for (int32_t s = 0; s < levelBudget - flagRem; s++)
    {
        cout<<"rot_in_S2C: "<<endl;
        for (int32_t j = 0; j < g; j++)
        { // n1
            cout<<rot_in_S2C[s][j]<<", ";
            if (rot_in_S2C[s][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_in_S2C[s][j]);
                rotIdx_S2C.push_back(rot_in_S2C[s][j]);
            }
        }
        cout<<endl;
        
        cout<<"rot_out_S2C[s][i]: "<<endl;
        for (int32_t i = 0; i < b; i++)
        {
            cout<<rot_out_S2C[s][i]<<", ";
            if (rot_out_S2C[s][i] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_out_S2C[s][i]);
                rotIdx_S2C.push_back(rot_out_S2C[s][i]);
            }
        }
        cout<<endl;
    }

    if (flagRem)
    {
        int32_t stop = levelBudget - flagRem;
        cout<<"rot_in_S2C rem: "<<endl;
        for (int32_t j = 0; j < gRem; j++)
        {
            cout<<rot_in_S2C[stop][j]<<", ";
            if (rot_in_S2C[stop][j] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_in_S2C[stop][j]);
                rotIdx_S2C.push_back(rot_in_S2C[stop][j]);
            }
        }
        cout<<endl;

        cout<<"rot_out_S2C rem: "<<endl;
        for (int32_t i = 0; i < bRem; i++)
        {
            cout<<rot_out_S2C[stop][i]<<", ";
            if (rot_out_S2C[stop][i] != 0)
            {
                scheme.addLeftRotKey_23(secretKey, rot_in_S2C[stop][i]);
                rotIdx_S2C.push_back(rot_in_S2C[stop][i]);
            }
        }
        cout<<endl;
    }
}


void EncodingMatrix::addBootstrappingKey()
{
    addS2CKey();
    addC2SKey();

    set<int> rotIdx_C2S_set(rotIdx_C2S.begin(), rotIdx_C2S.end());
    set<int> rotIdx_S2C_set(rotIdx_S2C.begin(), rotIdx_S2C.end());
    set<int> res_C2S, res_S2C;
    set_difference(rotIdx_C2S_set.begin(), rotIdx_C2S_set.end(),
                    rotIdx_S2C_set.begin(), rotIdx_S2C_set.end(),
                    std::inserter(res_C2S, res_C2S.begin()));

    set_difference(rotIdx_S2C_set.begin(), rotIdx_S2C_set.end(),
                   rotIdx_C2S_set.begin(), rotIdx_C2S_set.end(),
                   std::inserter(res_S2C, res_S2C.begin()));

    rotIdx_C2S.assign(res_C2S.begin(), res_C2S.end());
    rotIdx_S2C.assign(res_S2C.begin(), res_S2C.end());
}

void EncodingMatrix::Precompute_rot_in_out_C2S()
{
    int stop = -1;
    int flagRem = 0;
    int BASE_NUM_LEVELS_TO_DROP = 1;
    long N = scheme.context.N;
    long M = scheme.context.M;
    long slots = scheme.context.slots;

    long K = scheme.context.K;
    long logN = scheme.context.logN;

    int levelBudget = m_paramsEnc[0];
    int layersCollapse = m_paramsEnc[1];
    int remCollapse = m_paramsEnc[2];
    int numRotations = m_paramsEnc[3];
    int b = m_paramsEnc[4];
    int g = m_paramsEnc[5];
    int numRotationsRem = m_paramsEnc[6];
    int bRem = m_paramsEnc[7];
    int gRem = m_paramsEnc[8];

    if (remCollapse != 0)
    {
        stop = 0;
        flagRem = 1;
    }

    rot_in_C2S  = vector<vector<int>>(levelBudget);
    rot_out_C2S = vector<vector<int>>(levelBudget);

    // precompute the inner and outer rotations
    // std::vector<std::vector<int32_t>> rot_in(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
    {
        if (flagRem == 1 && i == 0)
        {
            rot_in_C2S[i] = std::vector<int32_t>(numRotationsRem + 1);
        }
        else
        {
            rot_in_C2S[i] = std::vector<int32_t>(numRotations + 1);
        }
    }

    // std::vector<std::vector<int32_t>> rot_out(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
    {
        rot_out_C2S[i] = std::vector<int32_t>(b + bRem);
    }

    for (int32_t s = levelBudget - 1; s > stop; s--)
    {
        for (int32_t j = 0; j < g; j++)
        {
            rot_in_C2S[s][j] = ReduceRotation(
                (j - int32_t((numRotations + 1) / 2) + 1) * (1 << ((s - flagRem) * layersCollapse + remCollapse)),
                slots);
        }

        for (int32_t i = 0; i < b; i++)
        {
            rot_out_C2S[s][i] = ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
        }
    }

    if (flagRem)
    {
        for (int32_t j = 0; j < gRem; j++)
        {
            rot_in_C2S[stop][j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
        }

        for (int32_t i = 0; i < bRem; i++)
        {
            rot_out_C2S[stop][i] = ReduceRotation((gRem * i), M / 4);
        }
    }
}

void EncodingMatrix::Precompute_rot_in_out_S2C()
{
    int BASE_NUM_LEVELS_TO_DROP = 1;
    int levelBudget = m_paramsDec[0];
    int layersCollapse = m_paramsDec[1];
    int remCollapse = m_paramsDec[2];
    int numRotations = m_paramsDec[3];
    int b = m_paramsDec[4];
    int g = m_paramsDec[5];
    int numRotationsRem = m_paramsDec[6];
    int bRem = m_paramsDec[7];
    int gRem = m_paramsDec[8];
    int flagRem = 0;
    long N = scheme.context.N;
    long logN = scheme.context.logN;
    long M = scheme.context.M;
    long K = scheme.context.K;
    long slots = N / 2;
    if (remCollapse != 0)
    {
        flagRem = 1;
    }

    // precompute the inner and outer rotations
    // vector<vector<int32_t>> rot_in_S2C(levelBudget);
    rot_in_S2C  = vector<vector<int>>(levelBudget);
    rot_out_S2C = vector<vector<int>>(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
    {
        if (flagRem == 1 && i == uint32_t(levelBudget - 1))
        {
            // remainder corresponds to index 0 in encoding and to last index in decoding
            rot_in_S2C[i] = vector<int32_t>(numRotationsRem + 1);
        }
        else
        {
            rot_in_S2C[i] = vector<int32_t>(numRotations + 1);
        }
    }

    for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
    {
        rot_out_S2C[i] = vector<int32_t>(b + bRem);
    }

    for (int32_t s = 0; s < levelBudget - flagRem; s++)
    {
        for (int32_t j = 0; j < g; j++)
        {
            rot_in_S2C[s][j] = ReduceRotation(
                (j - int32_t((numRotations + 1) / 2) + 1) * (1 << (s * layersCollapse)),
                M / 4);
        }

        for (int32_t i = 0; i < b; i++)
        {
            rot_out_S2C[s][i] = ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4);
        }
    }
    

    if (flagRem)
    {
        int32_t s = levelBudget - flagRem;
        for (int32_t j = 0; j < gRem; j++)
        {
            rot_in_S2C[s][j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (s * layersCollapse)),
                                          M / 4);
        }

        for (int32_t i = 0; i < bRem; i++)
        {
            rot_out_S2C[s][i] = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4);
        }
    }

    
    // cout<<"S2C rot in:"<<endl;
    // for(auto v : rot_in_S2C)
    // {
    //     for(auto i : v)
    //     {
    //         cout<< i << ", ";
    //     }
    //     cout<<endl;
    // }

    // cout<<"S2C rot out:"<<endl;
    // for(auto v : rot_out_S2C)
    // {
    //     for(auto i : v)
    //     {
    //         cout<< i << ", ";
    //     }
    //     cout<<endl;
    // }
}


vector<complex<double>> EncodingMatrix::ComputeRoots(int N, bool a)
{
    int m = N << 1;
    vector<complex<double>> roots(m + 1);
    roots[0] = complex<double>(1, 0);
    double angle;
    for (int i = 1; i < m; i++)
    {
        angle = 2 * 3.14159265358979323846 * i / m;
        roots[i] = complex<double>(cos(angle), sin(angle));
    }
    roots[m] = roots[0];
    return roots;
}

vector<int32_t> EncodingMatrix::GetCollapsedFFTParams(uint32_tt slots, uint32_tt levelBudget, uint32_tt dim1)
{
    uint32_tt logSlots = log2(slots);
    // even for the case of a single slot we need one level for rescaling
    if (logSlots == 0)
    {
        logSlots = 1;
    }

    vector<uint32_tt> dims = SelectLayers(logSlots, levelBudget);
    // Need to compute how many layers are collapsed in each of the level from the budget.
    // If there is no exact division between the maximum number of possible levels (log(slots)) and the
    // level budget, the last level will contain the remaining layers collapsed.
    int32_t layersCollapse = dims[0];
    int32_t remCollapse = dims[2];

    bool flagRem = remCollapse != 0;

    uint32_tt numRotations    = (1 << (layersCollapse + 1)) - 1;
    uint32_tt numRotationsRem = (1 << (remCollapse + 1)) - 1;

    // Computing the baby-step b and the giant-step g for the collapsed layers for decoding.
    int32_t g;
    if (dim1 == 0 || dim1 > numRotations)
    {
        if (numRotations > 7)
        {
            g = (1 << (int32_t(layersCollapse / 2) + 2));
        }
        else
        {
            g = (1 << (int32_t(layersCollapse / 2) + 1));
        }
    }
    else
    {
        g = dim1;
    }
    int32_t b = (numRotations + 1) / g;

    int32_t bRem = 0;
    int32_t gRem = 0;
    if (flagRem)
    {
        if (numRotationsRem > 7)
        {
            gRem = (1 << (int32_t(remCollapse / 2) + 2));
        }
        else
        {
            gRem = (1 << (int32_t(remCollapse / 2) + 1));
        }
        bRem = (numRotationsRem + 1) / gRem;
    }

    // If this return statement changes then CKKS_BOOT_PARAMS should be altered as well
    return {int32_t(levelBudget), layersCollapse, remCollapse, int32_t(numRotations), b, g,
            int32_t(numRotationsRem), bRem, gRem};
}


uint32_tt EncodingMatrix::ReduceRotation(int index, int slots)
{
    int islots = slots;

    // if slots is a power of 2
    if ((slots & (slots - 1)) == 0)
    {
        int32_t n = log2(slots);
        if (index >= 0)
        {
            return index - ((index >> n) << n);
        }
        return index + islots + ((int(fabs(index)) >> n) << n);
    }
    return (islots + index % islots) % islots;
}


complex<double> *EncodingMatrix::Rotate(const vector<complex<double>> &a, int index, bool tr)
{
    int32_t slots = a.size();

    auto result = new complex<double>[slots]();

    if (index < 0 || index > slots)
    {
        index = ReduceRotation(index, slots);
    }

    if (index == 0)
    {
        for (int i = 0; i < slots; i++)
        {
            result[i] = a[i];
        }
    }
    else
    {
        // two cases: i+index <= slots and i+index > slots
        for (int32_t i = 0; i < slots - index; i++)
        {
            result[i] = a[i + index];
        }
        for (int32_t i = slots - index; i < slots; i++)
        {
            result[i] = a[i + index - slots];
        }
    }

    return result;
}
// openfhe

// OpenFHE的矩阵生成函数
vector<uint32_tt> EncodingMatrix::SelectLayers(uint32_tt logSlots, uint32_tt budget)
{
    uint32_tt layers = ceil(static_cast<double>(logSlots) / budget);
    uint32_tt rows = logSlots / layers;
    uint32_tt rem = logSlots % layers;

    uint32_tt dim = rows;
    if (rem != 0)
    {
        dim = rows + 1;
    }

    // the above choice ensures dim <= budget
    if (dim < budget)
    {
        layers -= 1;
        rows = logSlots / layers;
        rem = logSlots - rows * layers;
        dim = rows;

        if (rem != 0)
        {
            dim = rows + 1;
        }

        // the above choice endures dim >=budget
        while (dim != budget)
        {
            rows -= 1;
            rem = logSlots - rows * layers;
            dim = rows;
            if (rem != 0)
            {
                dim = rows + 1;
            }
        }
    }

    return {layers, rows, rem};
}

std::vector<std::vector<std::vector<std::complex<double>>>>
EncodingMatrix::CoeffEncodingCollapse(vector<std::complex<double>> pows, vector<uint32_t> rotGroup, uint32_t levelBudget,
                                      bool flag_i)
{
    uint32_t slots = rotGroup.size();
    // Need to compute how many layers are collapsed in each of the level from the budget.
    // If there is no exact division between the maximum number of possible levels (log(slots)) and the
    // level budget, the last level will contain the remaining layers collapsed.
    int32_t layersCollapse;
    int32_t remCollapse;

    std::vector<uint32_t> dims = SelectLayers(std::log2(slots), levelBudget);
    layersCollapse = dims[0];
    remCollapse = dims[2];

    int32_t dimCollapse = int32_t(levelBudget);
    int32_t stop = 0;
    int32_t flagRem = 0;

    if (remCollapse == 0)
    {
        stop = -1;
        flagRem = 0;
    }
    else
    {
        stop = 0;
        flagRem = 1;
    }

    uint32_t numRotations = (1 << (layersCollapse + 1)) - 1;
    uint32_t numRotationsRem = (1 << (remCollapse + 1)) - 1;

    // Computing the coefficients for encoding for the given level budget
    std::vector<std::vector<std::complex<double>>> coeff1 = CoeffEncodingOneLevel(pows, rotGroup, flag_i);

    // Coeff stores the coefficients for the given budget of levels
    std::vector<std::vector<std::vector<std::complex<double>>>> coeff(dimCollapse);
    for (uint32_t i = 0; i < uint32_t(dimCollapse); i++)
    {
        if (flagRem)
        {
            if (i >= 1)
            {
                // after remainder
                coeff[i] = std::vector<std::vector<std::complex<double>>>(numRotations);
                for (uint32_t j = 0; j < numRotations; j++)
                {
                    coeff[i][j] = std::vector<std::complex<double>>(slots);
                }
            }
            else
            {
                // remainder corresponds to the first index in encoding and to the last one in decoding
                coeff[i] = std::vector<std::vector<std::complex<double>>>(numRotationsRem);
                for (
                    uint32_t j = 0; j < numRotationsRem; j++)
                {
                    coeff[i][j] = std::vector<std::complex<double>>(slots);
                }
            }
        }
        else
        {
            std::vector<std::vector<std::complex<double>>> a(numRotations);
            coeff[i] = a;
            for (uint32_t j = 0; j < numRotations; j++)
            {
                coeff[i][j] = std::vector<std::complex<double>>(slots);
            }
        }
    }

    for (int32_t s = dimCollapse - 1; s > stop; s--)
    {
        int32_t top = int32_t(std::log2(slots)) - (dimCollapse - 1 - s) * layersCollapse - 1;

        for (int32_t l = 0; l < layersCollapse; l++)
        {
            if (l == 0)
            {
                coeff[s][0] = coeff1[top];
                coeff[s][1] = coeff1[top + std::log2(slots)];
                coeff[s][2] = coeff1[top + 2 * std::log2(slots)];
            }
            else
            {
                std::vector<std::vector<std::complex<double>>> temp = coeff[s];
                std::vector<std::vector<std::complex<double>>> zeros(numRotations,
                                                                     std::vector<std::complex<double>>(slots, 0.0));
                coeff[s] = zeros;
                uint32_t t = 0;

                for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++)
                {
                    for (uint32_t k = 0; k < slots; k++)
                    {
                        coeff[s][u + t][k] += coeff1[top - l][k] * temp[u][ReduceRotation(k - (1 << (top - l)), slots)];
                        coeff[s][u + t + 1][k] += coeff1[top - l + std::log2(slots)][k] * temp[u][k];
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * std::log2(slots)][k] *
                                                  temp[u][ReduceRotation(k + (1 << (top - l)), slots)];
                    }
                    t += 1;
                }
            }
        }
    }

    if (flagRem)
    {
        int32_t s = 0;
        int32_t top = int32_t(std::log2(slots)) - (dimCollapse - 1 - s) * layersCollapse - 1;

        for (int32_t l = 0; l < remCollapse; l++)
        {
            if (l == 0)
            {
                coeff[s][0] = coeff1[top];
                coeff[s][1] = coeff1[top + std::log2(slots)];
                coeff[s][2] = coeff1[top + 2 * std::log2(slots)];
            }
            else
            {
                std::vector<std::vector<std::complex<double>>> temp = coeff[s];
                std::vector<std::vector<std::complex<double>>> zeros(numRotationsRem,
                                                                     std::vector<std::complex<double>>(slots, 0.0));
                coeff[s] = zeros;
                uint32_t t = 0;

                for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++)
                {
                    for (uint32_t k = 0; k < slots; k++)
                    {
                        coeff[s][u + t][k] += coeff1[top - l][k] *
                                              temp[u][ReduceRotation(k - (1 << (top - l)), slots)];
                        coeff[s][u + t + 1][k] += coeff1[top - l + std::log2(slots)][k] * temp[u][k];
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * std::log2(slots)][k] *
                                                  temp[u][ReduceRotation(k + (1 << (top - l)), slots)];
                    }
                    t += 1;
                }
            }
        }
    }

    return coeff;
}

vector<vector<complex<double>>>
EncodingMatrix::CoeffEncodingOneLevel(vector<complex<double>>
                                          pows,
                                      vector<uint32_tt> rotGroup,
                                      bool flag_i)
{
    uint32_tt dim = pows.size() - 1;
    uint32_tt slots = rotGroup.size();

    complex<double> I(0, 1);

    // Each outer iteration from the FFT algorithm can be written a weighted sum of
    // three terms: the input shifted right by a power of two, the unshifted input,
    // and the input shifted left by a power of two. For each outer iteration
    // (log2(size) in total), the matrix coeff stores the coefficients in the
    // following order: the coefficients associated to the input shifted right,
    // the coefficients for the non-shifted input and the coefficients associated
    // to the input shifted left.
    vector<vector<complex<double>>> coeff(3 * log2(slots));

    for (uint32_tt i = 0; i < 3 * log2(slots); i++)
    {
        coeff[i] = vector<complex<double>>(slots);
    }

    for (uint32_tt m = slots; m > 1; m >>= 1)
    {
        uint32_tt s = log2(m) - 1;

        for (uint32_tt k = 0; k < slots; k += m)
        {
            uint32_tt lenh = m >> 1;
            uint32_tt lenq = m << 2;

            for (uint32_tt j = 0; j < lenh; j++)
            {
                uint32_tt jTwiddle = (lenq - (rotGroup[j] % lenq)) * (dim / lenq);

                if (flag_i && (m == 2))
                {
                    complex<double> w = exp(-M_PI / 2 * I) * pows[jTwiddle];
                    coeff[s + log2(slots)][j + k] = exp(-M_PI / 2 * I);     // not shifted
                    coeff[s + 2 * log2(slots)][j + k] = exp(-M_PI / 2 * I); // shifted left
                    coeff[s + log2(slots)][j + k + lenh] = -w;                   // not shifted
                    coeff[s][j + k + lenh] = w;                                       // shifted right
                }
                else
                {
                    complex<double> w = pows[jTwiddle];
                    coeff[s + log2(slots)][j + k] = 1;         // not shifted
                    coeff[s + 2 * log2(slots)][j + k] = 1;     // shifted left
                    coeff[s + log2(slots)][j + k + lenh] = -w; // not shifted
                    coeff[s][j + k + lenh] = w;                     // shifted right
                }
            }
        }
    }

    return coeff;
}

vector<vector<complex<double>>>
EncodingMatrix::CoeffDecodingOneLevel(vector<complex<double>>
                                          pows,
                                      vector<uint32_tt> rotGroup,
                                      bool flag_i)
{
    uint32_tt dim = pows.size() - 1;
    uint32_tt slots = rotGroup.size();
    complex<double> I(0, 1);

    // Each outer iteration from the FFT algorithm can be written a weighted sum of
    // three terms: the input shifted right by a power of two, the unshifted input,
    // and the input shifted left by a power of two. For each outer iteration
    // (log2(size) in total), the matrix coeff stores the coefficients in the
    // following order: the coefficients associated to the input shifted right,
    // the coefficients for the non-shifted input and the coefficients associated
    // to the input shifted left.
    vector<vector<complex<double>>> coeff(3 * log2(slots));

    for (uint32_tt i = 0; i < 3 * log2(slots); i++)
    {
        coeff[i] = vector<complex<double>>(slots);
    }

    for (uint32_tt m = 2; m <= slots; m <<= 1)
    {
        uint32_tt s = log2(m) - 1;

        for (uint32_tt k = 0; k < slots; k += m)
        {
            uint32_tt lenh = m >> 1;
            uint32_tt lenq = m << 2;

            for (uint32_tt j = 0; j < lenh; j++)
            {
                uint32_tt jTwiddle = (rotGroup[j] % lenq) * (dim / lenq);
                if (flag_i && (m == 2))
                {
                    complex<double> w = exp(M_PI / 2 * I) * pows[jTwiddle];
                    coeff[s + log2(slots)][j + k] = exp(M_PI / 2 * I); // not shifted
                    coeff[s + 2 * log2(slots)][j + k] = w;                  // shifted left
                    coeff[s + log2(slots)][j + k + lenh] = -w;              // not shifted
                    coeff[s][j + k + lenh] = exp(M_PI / 2 * I);             // shifted right
                }
                else
                {
                    complex<double> w = pows[jTwiddle];
                    coeff[s + log2(slots)][j + k] = 1;         // not shifted
                    coeff[s + 2 * log2(slots)][j + k] = w;     // shifted left
                    coeff[s + log2(slots)][j + k + lenh] = -w; // not shifted
                    coeff[s][j + k + lenh] = 1;                     // shifted right
                }
            }
        }
    }

    return coeff;
}

vector<vector<vector<complex<double>>>>
EncodingMatrix::CoeffDecodingCollapse(vector<complex<double>> pows, vector<uint32_tt> rotGroup,
                                      uint32_tt levelBudget, bool flag_i)
{
    uint32_tt slots = rotGroup.size();
    // Need to compute how many layers are collapsed in each of the level from the budget.
    // If there is no exact division between the maximum number of possible levels (log(slots)) and the
    // level budget, the last level will contain the remaining layers collapsed.
    int32_t layersCollapse;
    int32_t rowsCollapse;
    int32_t remCollapse;

    vector<uint32_tt> dims = SelectLayers(log2(slots), levelBudget);
    layersCollapse = dims[0];
    rowsCollapse = dims[1];
    remCollapse = dims[2];

    int32_t dimCollapse = int32_t(levelBudget);
    int32_t flagRem = 0;

    if (remCollapse == 0)
    {
        flagRem = 0;
    }
    else
    {
        flagRem = 1;
    }

    uint32_tt numRotations = (1 << (layersCollapse + 1)) - 1;
    uint32_tt numRotationsRem = (1 << (remCollapse + 1)) - 1;

    // Computing the coefficients for decoding for the given level budget
    vector<vector<complex<double>>> coeff1 = CoeffDecodingOneLevel(pows, rotGroup, flag_i);

    // Coeff stores the coefficients for the given budget of levels
    vector<vector<vector<complex<double>>>> coeff(dimCollapse);

    for (uint32_tt i = 0; i < uint32_tt(dimCollapse); i++)
    {
        if (flagRem)
        {
            if (i < levelBudget - 1)
            {
                // before remainder
                coeff[i] = vector<vector<complex<double>>>(numRotations);

                for (uint32_tt j = 0; j < numRotations; j++)
                {
                    coeff[i][j] = vector<complex<double>>(slots);
                }
            }
            else
            {
                // remainder corresponds to the first index in encoding and to the last one in decoding
                coeff[i] = vector<vector<complex<double>>>(numRotationsRem);

                for (uint32_tt j = 0; j < numRotationsRem; j++)
                {
                    coeff[i][j] = vector<complex<double>>(slots);
                }
            }
        }
        else
        {
            coeff[i] = vector<vector<complex<double>>>(numRotations);

            for (uint32_tt j = 0; j < numRotations; j++)
            {
                coeff[i][j] = vector<complex<double>>(slots);
            }
        }
    }

    for (int32_t s = 0; s < rowsCollapse; s++)
    {
        for (int32_t l = 0; l < layersCollapse; l++)
        {
            if (l == 0)
            {
                coeff[s][0] = coeff1[s * layersCollapse];
                coeff[s][1] = coeff1[log2(slots) + s * layersCollapse];
                coeff[s][2] = coeff1[2 * log2(slots) + s * layersCollapse];
            }
            else
            {
                vector<vector<complex<double>>> temp = coeff[s];
                vector<vector<complex<double>>> zeros(numRotations,
                                                                     vector<complex<double>>(slots, 0.0));
                coeff[s] = zeros;

                for (uint32_tt t = 0; t < 3; t++)
                {
                    for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++)
                    {
                        for (uint32_tt k = 0; k < slots; k++)
                        {
                            if (t == 0)
                                coeff[s][u][k] += coeff1[s * layersCollapse + l][k] * temp[u][k];
                            if (t == 1)
                                coeff[s][u + (1 << l)][k] +=
                                    coeff1[s * layersCollapse + l + log2(slots)][k] * temp[u][k];
                            if (t == 2)
                                coeff[s][u + (1 << (l + 1))][k] +=
                                    coeff1[s * layersCollapse + l + 2 * log2(slots)][k] * temp[u][k];
                        }
                    }
                }
            }
        }
    }

    if (flagRem)
    {
        int32_t s = rowsCollapse;

        for (int32_t l = 0; l < remCollapse; l++)
        {
            if (l == 0)
            {
                coeff[s][0] = coeff1[s * layersCollapse];
                coeff[s][1] = coeff1[log2(slots) + s * layersCollapse];
                coeff[s][2] = coeff1[2 * log2(slots) + s * layersCollapse];
            }
            else
            {
                vector<vector<complex<double>>> temp = coeff[s];
                vector<vector<complex<double>>> zeros(numRotationsRem,
                                                                     vector<complex<double>>(slots, 0.0));
                coeff[s] = zeros;

                for (uint32_tt t = 0; t < 3; t++)
                {
                    for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++)
                    {
                        for (uint32_tt k = 0; k < slots; k++)
                        {
                            if (t == 0)
                                coeff[s][u][k] += coeff1[s * layersCollapse + l][k] * temp[u][k];
                            if (t == 1)
                                coeff[s][u + (1 << l)][k] +=
                                    coeff1[s * layersCollapse + l + log2(slots)][k] * temp[u][k];
                            if (t == 2)
                                coeff[s][u + (1 << (l + 1))][k] +=
                                    coeff1[s * layersCollapse + l + 2 * log2(slots)][k] * temp[u][k];
                        }
                    }
                }
            }
        }
    }

    return coeff;
}

vector<vector<PlaintextT*>>
EncodingMatrix::EvalCoeffsToSlotsPrecompute(const vector<complex<double>> &A, const vector<uint32_tt> &rotGroup,
                                            bool flag_i, NTL::RR scale, uint32_tt target_level)
{
    // 抄的 scaleEnc = 1/28
    double ImportScale = 1/1;
    // double ImportScale = 1;

    // is_sqrt_rescale = 0;
    int BASE_NUM_LEVELS_TO_DROP = 1;

    int levelBudget = m_paramsEnc[0];
    int layersCollapse = m_paramsEnc[1];
    int remCollapse = m_paramsEnc[2];
    int numRotations = m_paramsEnc[3];
    int b = m_paramsEnc[4];
    int g = m_paramsEnc[5];
    int numRotationsRem = m_paramsEnc[6];
    int bRem = m_paramsEnc[7];
    int gRem = m_paramsEnc[8];

    cout<<"numRotations: "<<numRotations<<endl;
    cout<<"numRotationsRem: "<<numRotationsRem<<endl;

    int stop = -1;
    int flagRem = 0;

    long N = scheme.context.N;
    long M = scheme.context.M;
    long K = scheme.context.K;

    cout<<"N: "<< N<<endl;
    cout<<"M: "<< M<<endl;
    cout<<"K: "<< K<<endl;

    long slots = scheme.context.slots;

    if (remCollapse != 0)
    {
        stop = 0;
        flagRem = 1;
    }

    // result is the rotated plaintext version of the coefficients
    vector<vector<PlaintextT*>> result(levelBudget);
    for (int i = 0; i < levelBudget; i++)
    {
        vector<vector<complex<double>>> a1;
        if (flagRem == 1 && i == 0)
        {
            // remainder corresponds to index 0 in encoding and to last index in decoding
            for (int j = 0; j < numRotationsRem; j++)
            {
                vector<complex<double>> a2(slots, 0);

                // Plaintext plain(resultMid, N, slots, L + stop - levelBudget + 1 + K);

                PlaintextT* plain = new PlaintextT(N, scheme.context.t_num, slots, scale);
                // cout<<"level: "<<L + stop - levelBudget + 1 + K<<endl;
                result[i].push_back(plain);
                a1.push_back(a2);
            }
        }
        else
        {
            //            result[i] = vector<Plaintext>(numRotations);
            for (int j = 0; j < numRotations; j++)
            {
                vector<complex<double>> a2(slots, 0);

                // Plaintext plain(resultMid, N, slots, L + i - levelBudget + 1 + K);
                PlaintextT* plain = new PlaintextT(N, scheme.context.t_num, slots, scale);
                // cout<<"level: "<<L + i - levelBudget + 1 + K<<endl;
                result[i].push_back(plain);
                a1.push_back(a2);
            }
        }
    }

    if (slots == M / 4)
    {
        //------------------------------------------------------------------------------
        // fully-packed mode
        //------------------------------------------------------------------------------

        auto coeff = CoeffEncodingCollapse(A, rotGroup, levelBudget, flag_i);

        //        int sb = 0;
        for (int s = levelBudget - 1; s > stop; s--)
        {
            NTL::RR scaleReal(scale);
            for (int scaleI = 0; scaleI < scheme.context.q_num - target_level + levelBudget - s - 1; scaleI++)
            {
                scaleReal = scaleReal * scaleReal / scheme.context.qVec[scheme.context.q_num - scaleI - 1];
                cout<<"encode scaler index: "<<scheme.context.q_num - scaleI - 1<<endl;
            }

            // cout<<"encode num: " <<endl;
            for (int i = 0; i < b; i++)
            {
                for (int j = 0; j < g; j++)
                {
                    if (g * i + j != numRotations)
                    {
                        uint32_tt rot =
                            ReduceRotation(-g * i * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots);

                        if ((flagRem == 0) && (s == stop + 1)) {
                            // do the scaling only at the last set of coefficients
                            for (uint32_t k = 0; k < slots; k++) {
                                coeff[s][g * i + j][k] *= ImportScale;
                            }
                        }

                        auto rotateTemp = Rotate(coeff[s][g * i + j], rot, true);
                        if(s == levelBudget - 1 && is_STC_first){
                            for (int iVals = 0; iVals < slots; iVals++)
                            {
                                rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/2.);
                                rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/2.);
                                // if(iVals < slots/2 + 4 && iVals > slots/2) printf("%lf+i*%lf, ", rotateTemp[iVals].real(), rotateTemp[iVals].imag());
                            } 
                        }
                        else if(is_STC_first) {
                            for (int iVals = 0; iVals < slots; iVals++)
                            {
                                rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/2.);
                                rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/2.);

                                // if(iVals < slots/2 + 4 && iVals > slots/2) printf("%lf+i*%lf, ", rotateTemp[iVals].real(), rotateTemp[iVals].imag());
                            }
                        } else {
                            for (int iVals = 0; iVals < slots; iVals++)
                            {
                                rotateTemp_host[iVals].x = rotateTemp[iVals].real();
                                rotateTemp_host[iVals].y = rotateTemp[iVals].imag();

                                // if(iVals < slots/2 + 4 && iVals > slots/2) printf("%lf+i*%lf, ", rotateTemp[iVals].real(), rotateTemp[iVals].imag());
                            }
                        }
                        // cout<<endl;
                        // scheme.context.EncodePQ(result[s][g * i + j].mx, rotateTemp, scaleReal, slots,
                        //                         L + s - levelBudget + 1, K);
                        cudaMemcpy(rotateTemp_device, rotateTemp_host.data(), sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
                        scheme.context.encode_T(rotateTemp_device, *result[s][g * i + j], scale);

                        // cout<<K + target_level + s - levelBudget + 1 << ", ";
                    }
                }
            }
        }

        if (flagRem)
        {
            NTL::RR scaleReal = scale;
            for (int scaleI = 0; scaleI < scheme.context.q_num - target_level + levelBudget - stop - 1; scaleI++)
            {
                scaleReal = scaleReal * scaleReal / scheme.context.qVec[scheme.context.q_num - scaleI - 1];
            }
            
            // cout<<"encode num: " <<endl;
            for (int32_t i = 0; i < bRem; i++)
            {
                for (int32_t j = 0; j < gRem; j++)
                {
                    if (gRem * i + j != int32_t(numRotationsRem))
                    {
                        uint32_tt rot = ReduceRotation(-gRem * i, slots);
                        for (uint32_tt k = 0; k < slots; k++)
                        {
                            coeff[stop][gRem * i + j][k] *= ImportScale;
                        }

                        auto rotateTemp = Rotate(coeff[stop][gRem * i + j], rot, true);
                        if (is_STC_first)
                        {
                            for (int iVals = 0; iVals < slots; iVals++)
                            {
                                rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/2. * scheme.context.eval_sine_K);
                                rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/2. * scheme.context.eval_sine_K);
                            }
                        }
                        else
                        {
                            for (int iVals = 0; iVals < slots; iVals++)
                            {
                                rotateTemp_host[iVals].x = rotateTemp[iVals].real();
                                rotateTemp_host[iVals].y = rotateTemp[iVals].imag();
                            }
                        }
                        // scheme.context.EncodePQ(result[stop][gRem * i + j], rotateTemp, scaleReal, slots,
                        //                         L + stop - levelBudget + 1, K);
                        cudaMemcpy(rotateTemp_device, rotateTemp_host.data(), sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
                        scheme.context.encode_T(rotateTemp_device, *result[stop][gRem * i + j], scale);

                        // cout<< K + target_level + stop - levelBudget + 1<< ", ";
                    }
                }
            }
        }
    }

    return result;
}

vector<vector<PlaintextT*>>
EncodingMatrix::EvalSlotsToCoeffsPrecompute(const vector<complex<double>> &A, const vector<uint32_tt> &rotGroup,
                                            bool flag_i, NTL::RR scale, uint32_tt target_level)
{
    
    int BASE_NUM_LEVELS_TO_DROP = 1;

    int levelBudget = m_paramsDec[0];
    int layersCollapse = m_paramsDec[1];
    int remCollapse = m_paramsDec[2];
    int numRotations = m_paramsDec[3];
    int b = m_paramsDec[4];
    int g = m_paramsDec[5];
    int numRotationsRem = m_paramsDec[6];
    int bRem = m_paramsDec[7];
    int gRem = m_paramsDec[8];

    int flagRem = 0;
    long N = scheme.context.N;
    long M = scheme.context.M;
    long K = scheme.context.K;

    long slots = N / 2;
    // double ImportScale = 1/50;
    double ImportScale = 1;


    if (remCollapse != 0)
    {
        flagRem = 1;
    }

    // result is the rotated plaintext version of coeff
    vector<vector<PlaintextT*>> result(levelBudget);
    for (uint32_tt i = 0; i < uint32_tt(levelBudget); i++)
    {
        if (flagRem == 1 && i == uint32_tt(levelBudget - 1))
        {
            // remainder corresponds to index 0 in encoding and to last index in decoding
            //            result[i] = vector<Plaintext>(numRotationsRem);
            for (int j = 0; j < numRotationsRem; j++)
            {
                // Plaintext plain(resultMid, N, slots, L - i + K);
                PlaintextT* plain = new PlaintextT(N, scheme.context.t_num, slots, scale);
                result[i].push_back(plain);
            }
        }
        else
        {
            //            result[i] = vector<Plaintext>(numRotations);
            for (int j = 0; j < numRotations; j++)
            {
                PlaintextT* plain = new PlaintextT(N, scheme.context.t_num, slots, scale);
                result[i].push_back(plain);
            }
        }
    }
    auto coeff = CoeffDecodingCollapse(A, rotGroup, levelBudget, flag_i);

    for (int32_t s = 0; s < levelBudget - flagRem; s++)
    {
        // cout<<"decode num: " <<endl;
        for (int32_t i = 0; i < b; i++)
        {
            for (int32_t j = 0; j < g; j++)
            {
                if (g * i + j != int32_t(numRotations))
                {
                    uint32_tt rot = ReduceRotation(-g * i * (1 << (s * layersCollapse)), slots);
                    if ((flagRem == 0) && (s == levelBudget - flagRem - 1))
                    {
                        // do the scaling only at the last set of coefficients
                        // for (uint32_tt k = 0; k < slots; k++) {
                        //     coeff[s][g * i + j][k] *= ImportScale;
                        // }
                    }

                    auto rotateTemp = Rotate(coeff[s][g * i + j], rot, true);
                    if(s == 0 && is_STC_first){
                        for (int iVals = 0; iVals < slots; iVals++)
                        {
                            rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/16.);
                            rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/16.);
                        }
                    }else if(is_STC_first){
                        for (int iVals = 0; iVals < slots; iVals++)
                        {
                            rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/8.);
                            rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/8.);
                        }
                    }
                    else {
                        for (int iVals = 0; iVals < slots; iVals++)
                        {
                            rotateTemp_host[iVals].x = rotateTemp[iVals].real();
                            rotateTemp_host[iVals].y = rotateTemp[iVals].imag();
                        }
                    }
                    cudaMemcpy(rotateTemp_device, rotateTemp_host.data(), sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
                    scheme.context.encode_T(rotateTemp_device, *result[s][g * i + j], scale);

                    // cout<< K + target_level - s<<", ";
                }
            }
        }
    }
    if (flagRem)
    {
        int32_t s = levelBudget - flagRem;

        // cout<<"decode num: " <<endl;
        for (int32_t i = 0; i < bRem; i++)
        {
            for (int32_t j = 0; j < gRem; j++)
            {
                if (gRem * i + j != int32_t(numRotationsRem))
                {
                    uint32_tt rot = ReduceRotation(-gRem * i * (1 << (s * layersCollapse)), slots);
                    //                    for (uint32_tt k = 0; k < slots; k++) {
                    //                        coeff[s][gRem * i + j][k] *= scale;
                    //                    }
                    auto rotateTemp = Rotate(coeff[s][gRem * i + j], rot, true);
                    for (int iVals = 0; iVals < slots; iVals++)
                    if (is_STC_first)
                    {
                        rotateTemp_host[iVals].x = rotateTemp[iVals].real()/(b*g/4.);
                        rotateTemp_host[iVals].y = rotateTemp[iVals].imag()/(b*g/4.);
                    }
                    else
                    {
                        rotateTemp_host[iVals].x = rotateTemp[iVals].real();
                        rotateTemp_host[iVals].y = rotateTemp[iVals].imag();
                    }
                    // scheme.context.EncodePQ(result[s][gRem * i + j].mx, rotateTemp, scale, slots, L - s, K);
                    cudaMemcpy(rotateTemp_device, rotateTemp_host.data(), sizeof(cuDoubleComplex) * slots, cudaMemcpyHostToDevice);
                    scheme.context.encode_T(rotateTemp_device, *result[s][gRem * i + j], scale);

                    // cout<< K + target_level - s<<", ";
                }
            }
        }
    }

    return result;
}
