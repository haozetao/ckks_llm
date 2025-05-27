#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Context_23.cuh"
#include "include/TimeUtils.cuh"
#include "include/pcmm/PCMM_Context.cuh"
#include "include/pcmm/PCMM_Scheme.cuh"

int main(int argc, char* argv[])
{
    if(argc != 4) return 0;

    int logN = atoi(argv[1]);
    int poly_num = atoi(argv[2]);
    int mod_num = atoi(argv[3]);
    int logslots = logN - 1;
    int N = 1 << logN;
    int ngpus;
    cudaGetDeviceCount(&ngpus);

    Context_23 context(logN, logslots, 192);
    Scheme_23 scheme(context);
    cudaDeviceSynchronize();
    cout<<"Generate Context OK"<<endl;
    printf("logN: %d Pnum: %d Qnum: %d Tnum: %d dnum: %d gamma: %d\n", logN, context.p_num, context.q_num, context.t_num, context.dnum, context.gamma);

    int PCMM_N1 = 256;
    int mlwe_rank = N / PCMM_N1;
    // ring packing always works on level0 ???
    vector<uint64_tt> p_ringpack = {context.pVec[0]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();

    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme);
    cout<<pcmm_context.psi_pq_ringpack[0]<<endl;

    SecretKey sk(context);
    MLWESecretKey mlwe_sk(PCMM_N1, pcmm_context.pq_ringpack.size(), mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);

    uint64_tt* ax = new uint64_tt[PCMM_N1 * mod_num * poly_num];

    std::mt19937_64 gen(0);
    for(int t = 0; t < mod_num * poly_num; t++)
    {
        for(int i = 0; i < PCMM_N1; i++) ax[i + t*PCMM_N1] = i%1000 % q_ringpack[0];
    }

    uint64_tt* ax_device;
    cudaMalloc(&ax_device, sizeof(uint64_tt) * PCMM_N1 * mod_num * poly_num);
    cudaMemcpy(ax_device, ax, sizeof(uint64_tt) * PCMM_N1 * mod_num * poly_num, cudaMemcpyHostToDevice);

    double ntt = 1000, intt = 1000;
    double ntt_all = 0, intt_all = 0;

    CUDATimer cuTimer;
    print_device_array(ax_device, PCMM_N1, 1, "ax");

    for (int i = 0; i < 1; i++)
    {
        cuTimer.start();
            pcmm_context.ToNTTInplace(ax_device, poly_num, mod_num, 0, p_ringpack_count);
        ntt = min(cuTimer.stop(), ntt);
        ntt_all += ntt;
        
        barrett_batch_device(ax_device, ax_device, PCMM_N1*poly_num, 0, 0, context.K, mod_num);

        cuTimer.start();
            pcmm_context.FromNTTInplace(ax_device, poly_num, mod_num, 0, p_ringpack_count);
        intt = min(cuTimer.stop(), intt);
        intt_all += intt;
    }
    for(int idx = 0; idx < mod_num; idx++)
        // for(int i = poly_num - 3; i < poly_num; i++)
        print_device_array(ax_device + idx*poly_num*PCMM_N1, PCMM_N1, poly_num, "ax");

    printf("Time: 4step  ntt: %f us intt: %f us\n", ntt*1000, intt*1000);
    printf("Throughput: 4step  ntt: %f OPs intt: %f OPs\n", 1000/double(ntt)*poly_num*mod_num, 1000/double(intt)*poly_num*mod_num);
    printf("Time: total ntt: %f ms intt: %f ms\n", ntt_all, intt_all);
    return 0;
}