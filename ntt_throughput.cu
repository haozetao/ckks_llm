#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Context_23.cuh"
#include "include/TimeUtils.cuh"

int main(int argc, char* argv[])
{
    if(argc != 4) return 0;

    int logN = atoi(argv[1]);
    int poly_num = atoi(argv[2]);
    int mod_num = atoi(argv[3]);
    int logslots = logN - 1;

    Context_23 context(logN, logslots);
    cudaDeviceSynchronize();
    cout<<"Generate Context OK"<<endl;
    printf("logN: %d Pnum: %d Qnum: %d Tnum: %d dnum: %d gamma: %d\n", logN, context.p_num, context.q_num, context.t_num, context.dnum, context.gamma);

    int N = context.N;
    int L = context.L;
    int K = context.K;
    int slots = context.slots;

    // int poly_num = 128;

    uint64_tt* ax = new uint64_tt[N * mod_num * poly_num];

    std::mt19937_64 gen(0);
    for(int t = 0; t < mod_num * poly_num; t++)
    {
        for(int i = 0; i < N; i++) ax[i + t*N] = i % context.qVec[0];
    }

    uint64_tt* ax_device;
    cudaMalloc(&ax_device, sizeof(uint64_tt) * N * mod_num * poly_num);
    cudaMemcpy(ax_device, ax, sizeof(uint64_tt) * N * mod_num * poly_num, cudaMemcpyHostToDevice);

    double ntt = 1000, intt = 1000;
    double ntt_all = 0, intt_all = 0;

    CUDATimer cuTimer;

    for (int i = 0; i < 1; i++)
    {
        cuTimer.start();
            context.FromNTTInplace(ax_device, 0, 0, poly_num, mod_num, mod_num);
        intt = min(cuTimer.stop(), intt);
        intt_all += intt;

        cuTimer.start();
            context.ToNTTInplace(ax_device, 0, 0, poly_num, mod_num, mod_num);
        ntt = min(cuTimer.stop(), ntt);
        ntt_all += ntt;
    }
    print_device_array(ax_device, N, poly_num, "ax");

    printf("Time: 4step  ntt: %f us intt: %f us\n", ntt*1000, intt*1000);
    printf("Throughput: 4step  ntt: %f OPs intt: %f OPs\n", 1000/double(ntt)*poly_num*mod_num, 1000/double(intt)*poly_num*mod_num);
    printf("Time: total ntt: %f ms intt: %f ms\n", ntt_all, intt_all);
    return 0;
}