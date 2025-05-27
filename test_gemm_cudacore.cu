#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#include "include/Context_23.cuh"
#include "include/TimeUtils.cuh"
#include "include/pcmm/PCMM_Context.cuh"
#include "include/pcmm/PCMM_Scheme.cuh"
#include "include/precision.cuh"


#define block_256_256 16
#define block_4096_65536 32
#define threads_one_warp 32
#define warps_one_block 32

int M_ = 4096;
int N_ = 4096;
int K_ = 65536;

// 最native的方法,矩阵乘法 N1*N1*N1
__global__ void matMultCUDA_kernel(uint64_tt* mlwe_plant_result_device, uint64_tt* mlwe_plant_1_device, uint64_tt* mlwe_plant_2_device, int row, int mid, int col, int mlwe_l, int p_num)
{
    int index_mod = blockIdx.y;// 0~1
    int global_idx = blockIdx.x * ringSwitch_block + threadIdx.x;// 0~255

    int row_res = global_idx / col;
    int col_res = global_idx % row;

    uint64_tt q = pqt_cons[p_num + index_mod];
    uint128_tt mu = {pqt_mu_cons_high[p_num + index_mod], pqt_mu_cons_low[p_num + index_mod]};

    // uint64_tt* plaint_1_this_mod = mlwe_plant_1_device + index_mod * N1;

    uint128_tt acc = 0;
    if(row_res<row&&col_res<col)
    {
        for(int i=0;i<mid;i++)
        {
            madc_uint64_uint64_uint128(mlwe_plant_1_device[row_res*mid+i], mlwe_plant_2_device[i * mid + col_res], acc);
        }
        singleBarrett_new(acc, q, mu);
        mlwe_plant_result_device[global_idx] = acc.low;
    }


}
// shared memory  width*width*width
template<int TILE_WIDTH>
__global__ void matMultCUDA_shared_kernel(uint64_tt* P, uint64_tt* M, uint64_tt* N, int width, int mlwe_l, int p_num)
{
	// shared memory
	__shared__ uint64_tt Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ uint64_tt Nds[TILE_WIDTH][TILE_WIDTH];

    int index_mod = blockIdx.z;
    uint64_tt q = pqt_cons[p_num + index_mod];
    uint128_tt mu = {pqt_mu_cons_high[p_num + index_mod], pqt_mu_cons_low[p_num + index_mod]}; 


	int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;
	int col = by*TILE_WIDTH + ty;
	int row = bx*TILE_WIDTH + tx;
 
	uint128_tt acc = 0;
	for (int i = 0; i < width / TILE_WIDTH; ++i)
	{
		// every thread write corresponding value to shared memory
		Mds[ty][tx] = M[row*width + i*TILE_WIDTH + tx];
		Nds[ty][tx] = N[col + width*(ty + TILE_WIDTH*i)];
 
		// wait for all threads
		__syncthreads();
 
		for (int j = 0; j < TILE_WIDTH; ++j)
		{
            madc_uint64_uint64_uint128(Mds[ty][j], Nds[j][tx], acc);

		}
        singleBarrett_new(acc, q, mu);
        __syncthreads(); 
        P[row*width + col] = acc.low;
	}
}
// shared memory  r*m*c
template<int TILE_WIDTH>
__global__ void matMultCUDA_shared_k_kernel(uint64_tt* P, uint64_tt* M, uint64_tt* N, int r, int m, int c, int mlwe_l, int p_num)
{

	// shared memory
	__shared__ uint64_tt Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ uint64_tt Nds[TILE_WIDTH][TILE_WIDTH];

    int index_mod = blockIdx.z;
    uint64_tt q = pqt_cons[p_num + index_mod];
    uint128_tt mu = {pqt_mu_cons_high[p_num + index_mod], pqt_mu_cons_low[p_num + index_mod]}; 


	int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

    int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;

    

	uint128_tt acc = 0;
    for (int i = 0; i < m / TILE_WIDTH; ++i)
	{
		// every thread write corresponding value to shared memory
		Mds[ty][tx] = M[row*m + i*TILE_WIDTH + tx];
		Nds[ty][tx] = N[col + c*(ty + TILE_WIDTH*i)];
 
		// wait for all threads
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; ++j)
		{
            madc_uint64_uint64_uint128(Mds[ty][j], Nds[j][tx], acc);
		}

        singleBarrett_new(acc, q, mu);
        
		// __syncthreads();
	}
    P[row*c + col] = acc.low;
    // printf(",%d",row*c + col);
	
	
}


template<int mlwe_rank>
__global__ void mlwePlantMult_kernel(uint64_tt* mlwe_plant_result_device, uint64_tt* mlwe_plant_1_device, uint64_tt* mlwe_plant_2_device, int N, int N1, int mlwe_L, int L, int p_num)
{
    int index_mod = blockIdx.y;// 0~1
    int global_idx = blockIdx.x * ringSwitch_block + threadIdx.x;// 0~255
    int index_plant = blockIdx.z;//0~255

    uint64_tt q = pqt_cons[p_num + index_mod];

    uint64_tt* plant_1_mx = mlwe_plant_1_device + index_plant * (N1 * index_mod);
    uint64_tt* plant_2_mx = mlwe_plant_2_device + index_plant * (N1 * index_mod);


}


__host__ void PCMM_Scheme::mlwePlantMult(vector<MLWEPlaintext*> mlwe_plant_result, vector<MLWEPlaintext*> mlwe_plant_1, vector<MLWEPlaintext*> mlwe_plant_2)
{
    int N = context.N;
    int L = mlwe_plant_1[0]->L;
    int N1 = pcmm_context.N1;
    int mlwe_rank = pcmm_context.mlwe_rank;
    int p_num = pcmm_context.p_ringpack.size();
    int l = mlwe_plant_1[0]->l;
    int extract_num = mlwe_plant_1.size();



    pcmm_context.FromNTTInplace(mlwe_plant_1[0]->mx_device, 1, l+1, 0, pcmm_context.ringpack_p_count, extract_num);


    dim3 rlweCipherDecompose_dim(N1 / ringSwitch_block, l + 1, extract_num);
    if(mlwe_rank == 256){
        mlwePlantMult_kernel <256> <<< rlweCipherDecompose_dim, ringSwitch_block >>> (mlwe_plant_result[0]->mx_device, mlwe_plant_1[0]->mx_device, mlwe_plant_2[0]->mx_device, N, N1, l, L, p_num);
    } else {
        cout << "mlwe rank not supported!" << endl;
    }
    pcmm_context.ToNTTInplace(mlwe_plant_1[0]->mx_device, 1, l+1, 0, pcmm_context.ringpack_p_count, extract_num);

 
}

// 计算res[0][0] = a[0][i]*b[i][0]
void mat_mul(uint64_tt* res, uint64_tt* a, uint64_tt* b, uint64_tt M, uint64_tt K, uint64_tt N, uint64_tt q)
{
    for(int i=0;i<1;i++)
    {
        for(int j=0;j<5;j++)
        {
            for(int k=0;k<K;k++)
            {
                // cout<<a[i*N+k]<<"*"<<b[k*N+j]<<"=";
                // __uint128_t qwe = a[i*N+k]*b[k*N+j];
                // cout<<static_cast<uint64_t>(qwe%q)<<",";
                // res[i*N+j] = (res[i*N+j] + (a[i*N+k]*b[k*N+j])%q)%q;
                res[i*N+j] = (res[i*N+j] +  mulMod128(a[i*N+k],b[k*N+j],q)) % q;
                
            }
        }
    }
    cout<<endl;
}


int main(int argc, char* argv[])
{
    if(argc != 2) return 0;

    int logN = atoi(argv[1]);
    int logslots = logN - 1;

    Context_23 context(logN, logslots, 192);
    Scheme_23 scheme(context);
    cudaDeviceSynchronize();
    cout<<"Generate Context OK"<<endl;
    printf("logN: %d Pnum: %d Qnum: %d Tnum: %d dnum: %d gamma: %d\n", logN, context.p_num, context.q_num, context.t_num, context.dnum, context.gamma);

    int N = context.N;
    int slots = context.slots;
    int L = context.L;
    int K = context.K;

    int PCMM_N1 = 256;
    int mlwe_rank = N / PCMM_N1;
    // ring packing always works on level0 ???
    vector<uint64_tt> p_ringpack = {context.pVec[0]};
    vector<uint64_tt> q_ringpack = {context.qVec[0], context.qVec[1]};
    int p_ringpack_count = p_ringpack.size();
    int q_ringpack_count = q_ringpack.size();


    PCMM_Context pcmm_context(PCMM_N1, mlwe_rank, p_ringpack, q_ringpack, context);
    PCMM_Scheme pcmm_scheme(pcmm_context, scheme);

    int pq_ringpack_count = pcmm_context.pq_ringpack.size();

    SecretKey sk(context);
    MLWESecretKey mlwe_sk(PCMM_N1, pq_ringpack_count, mlwe_rank);
    pcmm_scheme.convertMLWESKfromRLWESK(mlwe_sk, sk);
    scheme.addEncKey(sk);
    
    double* real_mes_host = new double[N];
    randomDoubleArray(real_mes_host, N, 1./10);
    for(int i = 0; i < 8; i++){
        printf("%lf, ", real_mes_host[i]);
    }
    cout<<endl;

    
    cudaEvent_t start1, end2;

    cudaEventCreate(&start1);
    cudaEventCreate(&end2);
    CUDATimer cuTimer;
    

    
    float gen_swk = 1000;
    float enc = 1000, dec = 1000, resc = 1000, ntt = 1000, intt = 1000, ecd = 1000, dcd = 1000;
    float global_little = 1000, global_big = 1000000000, shared_little = 1000, shared_big = 1000000000;
    float rlwe2mlwe = 1000;
    float temp_little = 0, temp_big = 0;
    int target_level = 1;


    uint64_tt* test_mul_host = new uint64_tt[N];
    uint64_tt* test_mul_cpu_host = new uint64_tt[N];

    uint64_tt* test_res = new uint64_tt[N];

    uint64_tt* test_mat_1_host = new uint64_tt[M_*N_];
    uint64_tt* test_mat_2_host = new uint64_tt[N_*K_];
    uint64_tt* test_mat_res_host = new uint64_tt[M_*K_];
    uint64_tt* test_mat_res_cpu_host = new uint64_tt[N_];


    std::random_device rd;  // 硬件随机数生成器（如果可用）
    std::mt19937_64 gen(rd()); // 使用 64 位版本的梅森旋转算法
    std::uniform_int_distribution<uint64_t> distribution(0, 562949979504640); // 生成 [0, 2^64-1] 的随机数

    for(int i=0;i<M_*N_;i++)
    {
        test_mat_1_host[i] = distribution(gen);
    }
    for(int i=0;i<N_*K_;i++)
    {
        test_mat_2_host[i] = distribution(gen);
    }

    for(int i=0;i<N;i++)
    {
        test_mul_host[i] = 562949979504641 - i%10;
    }    
    auto start = std::chrono::high_resolution_clock::now();
    mat_mul(test_mat_res_cpu_host, test_mat_1_host, test_mat_2_host, 4096, 4096, 65536, 562949979504641); // 太慢了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "big CPU_time: " << duration.count() << " us to execute." << std::endl;
    {
        uint64_tt* plaint_res_big_test;
        uint64_tt* plaint_1_big_test;
        uint64_tt* plaint_2_big_test;


        cudaMalloc(&plaint_res_big_test, sizeof(uint64_tt) * M_*K_);
        cudaMalloc(&plaint_1_big_test, sizeof(uint64_tt) * M_*N_);// 4096*4096
        cudaMalloc(&plaint_2_big_test, sizeof(uint64_tt) * N_*K_);// 4096*65536
        cudaMemcpy(plaint_1_big_test, test_mat_1_host, sizeof(uint64_tt) * M_*N_, cudaMemcpyHostToDevice);
        cudaMemcpy(plaint_2_big_test, test_mat_2_host, sizeof(uint64_tt) * N_*K_, cudaMemcpyHostToDevice); 
 
        
        for(int i=0;i<4;i++)
        {
            // big mat
            cudaEventRecord(start1);

                // dim3 block_mult_big(4096*65536/1024, 1);
                // matMultCUDA_kernel<<<block_mult_big, 1024>>>(plaint_res_big_test, plaint_1_big_test, plaint_2_big_test, 4096, 4096, 65536, 0, context.p_num);

            cudaEventRecord(end2);
            cudaEventSynchronize(end2);
            cudaEventElapsedTime(&temp_big, start1, end2);        
            global_big = min(global_big, temp_big);


            cudaEventRecord(start1);

                // cudaFuncSetCacheConfig(&matMultCUDA_shared_k_kernel, )
                cudaFuncSetAttribute(&matMultCUDA_shared_k_kernel<block_4096_65536>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32*1024);//设置shared_memory 大小


                // dim3 dimGrid_big(4096/32, 65536/32, 1);
                dim3 dimGrid_big(K_/block_4096_65536, M_/block_4096_65536, 1);

                dim3 dimBlock_big(warps_one_block, threads_one_warp);
                matMultCUDA_shared_k_kernel<block_4096_65536><<<dimGrid_big, dimBlock_big>>>(plaint_res_big_test, plaint_1_big_test, plaint_2_big_test, 4096, 4096, 65536, 0, context.p_num);

            cudaEventRecord(end2);
            cudaEventSynchronize(end2);
            cudaEventElapsedTime(&temp_big, start1, end2);        
            shared_big = min(shared_big, temp_big);

        }
        cudaMemcpy(test_mat_res_host, plaint_res_big_test, sizeof(double) * M_*K_, cudaMemcpyDeviceToHost);// 4096*65536
    }

    // for(int i=0;i<33;i++)
    // {
    //     for(int j=0;j<33;j++)
    //     {
    //         cout<<test_mat_res_host[i*K_+j]<<",";
    //     }
    //     cout<<endl;
    // } 



    cout<<"cpu: "<<endl;
    for(int i=0;i<5;i++)
    {
        printf("%llu,",test_mat_res_cpu_host[i]);
    }
    printf("\n");
    cout<<"gpu: "<<endl;
    for(int i=0;i<5;i++)
    {
        printf("%llu,",test_mat_res_host[i]);
    }
    printf("\n");

    cout<<"GPU_global_big_time:"<<global_big<<" ms "<<endl;
    cout<<"GPU_shared_big_time:"<<shared_big<<" ms "<<endl;


    return 0;
}