#pragma once

#include "Context_23.h"
#include "Params.cuh"
#include "Encoder.cuh"
#include "BasisConv.cuh"
#include "ExternalProduct.cuh"
#include <assert.h>

using namespace std;

Context_23::Context_23(long logN, long logslots, long h, double sigma) :
logN(logN), logslots(logslots), h(h), sigma(sigma)
{
	// qVec, pVec, qMuVec, pMuVec, qPsi, pPsi
	getPrimeCKKS(h);

	N = 1L << logN;
	M = N << 1;
	logNh = logN - 1;
    slots = 1L << logslots;
	Nh = N >> 1;
	// Encryption parameters
	q_num = qVec.size();
	p_num = pVec.size();
	t_num = tVec.size();
	mod_num = p_num + q_num + t_num;
	L = q_num - 1;
	K = pVec.size();
	dnum = q_num / K;
	alpha = K;

	Ri_blockNum = ceil(double(p_num + q_num) / gamma);
	Qj_blockNum = ceil(double(q_num) / p_num);

	assert(Ri_blockNum <= max_Riblock_num);
	assert(Qj_blockNum <= max_Qjblock_num);
	// cout<<"Ri_blockNum: "<<Ri_blockNum<<"  Qj_blockNum: "<<Qj_blockNum<<endl;
	assert(t_num <= max_tnum);
	// cout<<"t_num: "<<t_num<<endl;


	randomArray_len = 0;

	NTL::RR::SetPrecision(1024);

	preComputeOnCPU();
	// printf("preComputeOnCPU OK\n");
	copyMemoryToGPU();
	// printf("copyMemoryToGPU OK\n");

    printf("logN: %d Pnum: %d Qnum: %d Tnum: %d gamma: %d\n", logN, p_num, q_num, t_num, gamma);
    printf("dnum: %d Ri_blockNum: %d, Qj_blockNum: %d\n", dnum, Ri_blockNum, Qj_blockNum);
}

void Context_23::preComputeOnCPU()
{
	/****************************************[p0,p1,...,p{k-1},q0,q1,...,qL]************************************************/
	NTL::ZZ mulP(1);
	NTL::ZZ mulPQ_gamma(1);
	NTL::ZZ mulQ_alpha(1);
	NTL::ZZ mulT(1);
	for(int i = 0; i < K; i++)
	{
		pqtVec.push_back(pVec[i]);
		pqt2Vec.push_back(pVec[i]*2);
		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= pVec[i];
		pMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(pMuVec[i].high);
		pqtMuVec_low.push_back(pMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, pVec[i]);
		pPsi.push_back(root);
		pqtPsi.push_back(root);

		mulP *= pVec[i];
	}
	for(int i = 0; i <= L; i++)
	{
		pqtVec.push_back(qVec[i]);
		pqt2Vec.push_back(qVec[i]*2);
		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= qVec[i];
		qMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(qMuVec[i].high);
		pqtMuVec_low.push_back(qMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, qVec[i]);
		qPsi.push_back(root);
		pqtPsi.push_back(root);

		if(i < p_num)
			mulQ_alpha *= qVec[i];
	}
	NTL::ZZ halfT(1);
	for(int i = 0; i < t_num; i++)
	{
		pqtVec.push_back(tVec[i]);
		pqt2Vec.push_back(tVec[i]*2);
		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= tVec[i];
		tMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(tMuVec[i].high);
		pqtMuVec_low.push_back(tMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, tVec[i]);
		tPsi.push_back(root);
		pqtPsi.push_back(root);

		halfT *= tVec[i];
		mulT *= tVec[i];
	}

	for(int i = 0; i < gamma; i++) mulPQ_gamma *= pqtVec[i];

	// cout<<mulT<<endl;
	// cout<<mulP * mulPQ_gamma * N * dnum<<endl;
	assert(mulT > mulQ_alpha * mulPQ_gamma * N * dnum);
	cout << NTL::conv<RR>(mulT) / NTL::conv<RR>(mulQ_alpha * mulPQ_gamma * N * dnum) << endl;
	assert(mulP > mulQ_alpha);
	cout << NTL::conv<RR>(mulP) / NTL::conv<RR>(mulQ_alpha) << endl;
	// cout<<"T / (Q[:alpha] * PQ[:gamma] * N * d): "<<mulT / (mulQ_alpha*mulPQ_gamma*dnum*N)<<endl;
	// cout<<"mulP / mulQ_alpha: "<<mulP / mulQ_alpha<<endl;

	halfT /= 2;
	for(int i = 0; i < p_num + q_num + t_num; i++)
	{
		halfTmodpqti.push_back(halfT % pqtVec[i]);
	}

	/*****************************************************pq_psi_related*****************************************************/
	for (int i = 0; i < K; i++)
		pqtPsiInv.push_back(modinv128(pPsi[i], pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		pqtPsiInv.push_back(modinv128(qPsi[i], qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		pqtPsiInv.push_back(modinv128(tPsi[i], tVec[i])); // tPsiInv

	/*****************************************************100x_ntt*****************************************************/
	for (int i = 0; i < K; i++)
		n_inv_host.push_back(modinv128(N, pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		n_inv_host.push_back(modinv128(N, qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		n_inv_host.push_back(modinv128(N, tVec[i])); // tPsiInv

	for (int i = 0; i < K; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i], pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i+K], qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i+K+L+1], tVec[i])); // tPsiInv
	/******************************************base convert from P x Ql to Ql************************************************/

	for(int l = 0; l <= L; l++)
	{
		pHatVecModq_23.push_back({});
		for(int i = 0; i < K; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii < K; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, pVec[ii], qVec[l]);
			}
			pHatVecModq_23[l].push_back(temp);
		}
	}

	for(int i = 0; i < K; i++)
	{
		uint64_tt temp = 1;
		for(int ii = 0; ii < K; ii++)
		{
			if(ii == i) continue;
			temp = mulMod128(temp, pVec[ii], pVec[i]);
		}
		temp = modinv128(temp, pVec[i]);
		pHatInvVecModp_23.push_back(temp);
		pHatInvVecModp_23_shoup.push_back(x_Shoup(temp, pVec[i]));
	}


	for(int l = 0; l <= L; l++)
	{
		PQ_inv_mod_qi_better.push_back({});
		for(int t = 0; t <= l; t++)
		{
			int upper = ceil(double(l+1) / K);

			uint64_tt temp = 1;
			for(int i = 0; i < K; i++)
			{
				temp = mulMod128(temp, pVec[i], qVec[t]);
			}
			for(int i = l+1; i < upper*K; i++)
			{
				temp = mulMod128(temp, qVec[i], qVec[t]);
			}
			temp = modinv128(temp, qVec[t]);
			PQ_inv_mod_qi_better[l].push_back(temp);
		}
	}

	for(int l = 0; l <= L; l++)
	{
		PQ_div_Qj_modqi.push_back({});
		for(int k = 0; k <= l; k++)
		{
			uint64_tt temp = 1;
			uint64_tt mod = qVec[k];
			for(int i = 0; i < K; i++)
			{
				temp = mulMod128(temp, pVec[i], mod);
			}
			for(int i = l+1; i < q_num; i++)
			{
				temp = mulMod128(temp, qVec[i], mod);
			}
			PQ_div_Qj_modqi[l].push_back(temp);
		}
	}

	/************************************base convert from Ri to T******************************************/
	for(int i = 0; i < Ri_blockNum; i++)
	{
		RiHatInvVecModRi_23.push_back({});
		RiHatInvVecModRi_23_shoup.push_back({});
		for(int j = 0; j < gamma && i*gamma + j < p_num + q_num; j++)
		{
			uint64_tt temp = 1;
			for(int jj = 0; jj < gamma && i*gamma + jj < p_num + q_num; jj++)
			{
				if(jj == j) continue;
				temp = mulMod128(temp, pqtVec[i*gamma + jj], pqtVec[i*gamma + j]);
			}
			temp = modinv128(temp, pqtVec[i*gamma + j]);
			RiHatInvVecModRi_23[i].push_back(temp);
			RiHatInvVecModRi_23_shoup[i].push_back(x_Shoup(temp, pqtVec[i*gamma + j]));
		}
	}

	for(int i = 0; i < Ri_blockNum; i++)
	{
		RiHatVecModT_23.push_back({});
		for(int k = 0; k < t_num; k++)
		{
			uint64_tt mod = tVec[k];
			RiHatVecModT_23[i].push_back({});
			for(int j = 0; j < gamma && i*gamma + j < p_num + q_num; j++)
			{
				uint64_tt temp = 1;
				for(int jj = 0; jj < gamma && i*gamma + jj < p_num + q_num; jj++)
				{
					if(jj == j) continue;
					temp = mulMod128(temp, pqtVec[i*gamma + jj], mod);
				}
				RiHatVecModT_23[i][k].push_back(temp);
			}
		}
	}

	for(int k = 0; k < t_num; k++)
	{
		Rimodti.push_back({});
		uint64_tt mod = tVec[k];
		for(int i = 0; i < Ri_blockNum; i++)
		{
			uint64_tt temp = 1;
			for(int j = 0; j < gamma && i*gamma+j < p_num + q_num; j++)
			{
				temp = mulMod128(temp, pqtVec[i*gamma + j], mod);
			}
			Rimodti[k].push_back(temp);
		}
	}

	/************************************base convert from Qj to T******************************************/
	for(int l = 0; l <= L; l++)
	{
		QjHatInvVecModQj_23.push_back({});
		QjHatInvVecModQj_23_shoup.push_back({});
		int block_num = ceil(double(l+1) / p_num);
		for(int i = 0; i < block_num; i++)
		{
			QjHatInvVecModQj_23[l].push_back({});
			QjHatInvVecModQj_23_shoup[l].push_back({});
			for(int j = 0; j < p_num && i*p_num + j <= l; j++)
			{
				uint64_tt temp = 1;
				for(int jj = 0; jj < p_num && i*p_num + jj <= l; jj++)
				{
					if(jj == j) continue;
					temp = mulMod128(temp, qVec[i*p_num + jj], qVec[i*p_num + j]);
				}
				temp = modinv128(temp, qVec[i*p_num + j]);
				QjHatInvVecModQj_23[l][i].push_back(temp);
				QjHatInvVecModQj_23_shoup[l][i].push_back(x_Shoup(temp, qVec[i*p_num + j]));
			}
		}
	}

	for(int l = 0; l <= L; l++)
	{
		QjHatVecModT_23.push_back({});
		int block_num = ceil(double(l+1) / p_num);
		for(int i = 0; i < block_num; i++)
		{
			QjHatVecModT_23[l].push_back({});
			for(int k = 0; k < t_num; k++)
			{
				uint64_tt mod = tVec[k];
				QjHatVecModT_23[l][i].push_back({});
				for(int j = 0; j < p_num && i*p_num + j <= l; j++)
				{
					uint64_tt temp = 1;
					for(int jj = 0; jj < p_num && i*p_num + jj <= l; jj++)
					{
						if(jj == j) continue;
						temp = mulMod128(temp, qVec[i*p_num + jj], mod);
					}
					QjHatVecModT_23[l][i][k].push_back(temp);
				}
			}
		}
	}

	for(int l = 0; l <= L; l++)
	{
		Qjmodti.push_back(vector<uint64_tt>(Qj_blockNum*t_num, 0));
		for(int k = 0; k < t_num; k++)
		{
			int block_num = ceil(double(l+1) / p_num);
			for(int i = 0; i < block_num; i++)
			{
				uint64_tt temp = 1;
				for(int j = 0; j < p_num && i*p_num + j <= l; j++)
				{
					temp = mulMod128(temp, pqtVec[p_num + i*p_num + j], tVec[k]);
				}
				Qjmodti[l][i*t_num + k] = temp;
			}
		}
	}

	/************************************base convert from T to Ri******************************************/
	{
		for(int i = 0; i < t_num; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii < t_num; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, tVec[ii], tVec[i]);
			}
			temp = modinv128(temp, tVec[i]);
			THatInvVecModti_23.push_back(temp);
			THatInvVecModti_23_shoup.push_back(x_Shoup(temp, tVec[i]));
		}
	}

	for(int i = 0; i < p_num + q_num; i++)
	{
		THatVecModRi_23.push_back({});
		for(int j = 0; j < t_num; j++)
		{
			uint64_tt temp = 1;
			for(int jj = 0; jj < t_num; jj++)
			{
				if(jj == j) continue;
				temp = mulMod128(temp, tVec[jj], pqtVec[i]);
			}
			THatVecModRi_23[i].push_back(temp);
		}
	}

	for(int i = 0; i < p_num + q_num; i++)
	{
		uint64_tt temp = 1;
		for(int k = 0; k < t_num; k++)
		{
			temp = mulMod128(temp, tVec[k], pqtVec[i]);
		}
		Tmodpqi.push_back(temp);
	}

	/******************************************Fast_conv_related***************************************************/
	/*********************************************P Inv mod qi*****************************************************/
	for(int i = 0; i <= L; i++)
	{
		uint64_tt temp = 1;
		for(int j = 0; j < K; j++)
		{
			temp = mulMod128(temp, pVec[j], qVec[i]);
		}
		PModqt.push_back(temp);
		PModqt_shoup.push_back(x_Shoup(temp, qVec[i]));

		temp = modinv128(temp, qVec[i]);
		PinvModq.push_back(temp);
		PinvModq_shoup.push_back(x_Shoup(temp, qVec[i]));
	}

	for(int i = 0; i < t_num; i++)
	{
		uint64_tt temp = 1;
		for(int j = 0; j < K; j++)
		{
			temp = mulMod128(temp, pVec[j], tVec[i]);
		}
		PModqt.push_back(temp);
		PModqt_shoup.push_back(x_Shoup(temp, tVec[i]));
	}

	/*********************************************Rescale_related***************************************************/
	/**********************************************ql Inv mod qi****************************************************/
	for(int l = 1; l <= L; l++)
	{
		qiInvVecModql.push_back({});
		qiInvVecModql_shoup.push_back({});
		for(int j = 0; j < l; j++)
		{
			uint64_tt qj_inv_mod_qi = modinv128(qVec[l], qVec[j]);
			qiInvVecModql.back().push_back(qj_inv_mod_qi);
			qiInvVecModql_shoup.back().push_back(x_Shoup(qj_inv_mod_qi, qVec[j]));
		}
	}

	/************************************************decode*********************************************************/
	/**********************************************ql Inv mod qi****************************************************/
	for(int l = 0; l <= L; l++)
	{
		QlInvVecModqi.push_back({});
		for(int i = 0; i <= l; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii <= l; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, qVec[ii], qVec[i]);
			}
			temp = modinv128(temp, qVec[i]);
			QlInvVecModqi[l].push_back(temp);
		}
	}

	for(int l = 0; l <= L; l++)
	{
		QlHatVecModt0.push_back({});
		for(int i = 0; i <= l; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii <= l; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, qVec[ii], tVec[0]);
			}
			QlHatVecModt0[l].push_back(temp);
		}
	}

	{
		
	}
}

/**************************************memory malloc & copy on GPU**********************************************/
void Context_23::copyMemoryToGPU()
{
    //pqPsiTable and pqPsiInvTable
    uint64_tt** pqtPsiTable = new uint64_tt*[(K+L+1+t_num)];
	uint64_tt** pqtPsiInvTable = new uint64_tt*[(K+L+1+t_num)];

	/*******************************************100x_NTT******************************************************/
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		pqtPsiTable[i] = new uint64_tt[N];
		pqtPsiInvTable[i] = new uint64_tt[N];
        fillTablePsi128_special(pqtPsi[i], pqtVec[i], pqtPsiInv[i], pqtPsiTable[i], pqtPsiInvTable[i], N, n_inv_host[i]);
    }

	uint64_tt** psi_shoup_table = new uint64_tt*[(K+L+1+t_num)];
    uint64_tt** psiinv_shoup_table = new uint64_tt*[(K+L+1+t_num)];
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		psi_shoup_table[i] = new uint64_tt[N];
		psiinv_shoup_table[i] = new uint64_tt[N];
        fillTablePsi_shoup128(pqtPsiTable[i], pqtVec[i], pqtPsiInvTable[i], psi_shoup_table[i], psiinv_shoup_table[i], N);
    }

	cudaMalloc(&psi_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
	cudaMalloc(&psiinv_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		cudaMemcpy(psi_table_device + i * N, pqtPsiTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(psiinv_table_device + i * N, pqtPsiInvTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);

		delete pqtPsiTable[i];
		delete pqtPsiInvTable[i];
	}
	delete pqtPsiTable;
	delete pqtPsiInvTable;

	cudaMalloc(&psi_shoup_table_device, sizeof(uint64_tt) *  N * (K+L+1+t_num));
	cudaMalloc(&psiinv_shoup_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		cudaMemcpy(psi_shoup_table_device + i * N, psi_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(psiinv_shoup_table_device + i * N, psiinv_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		delete psi_shoup_table[i];
		delete psiinv_shoup_table[i];
	}
	delete psi_shoup_table;
	delete psiinv_shoup_table;

	cudaMalloc(&n_inv_device, sizeof(uint64_tt)  * (K+L+1+t_num));
	cudaMalloc(&n_inv_shoup_device, sizeof(uint64_tt)  * (K+L+1+t_num));

	cudaMemcpy(n_inv_device, n_inv_host.data(), sizeof(uint64_tt) * (K+L+1+t_num), cudaMemcpyHostToDevice);
	cudaMemcpy(n_inv_shoup_device, n_inv_shoup_host.data(), sizeof(uint64_tt) * (K+L+1+t_num), cudaMemcpyHostToDevice);
	/************************************base convert from PQl to Ql****************************************/
	// P/pk					
	// [P/p0 P/p1 ... P/pk] mod qi
	// P/pk mod qi
	// size = (L + 1) * K
	// ok
	cudaMalloc(&pHatVecModq_23_device, sizeof(uint64_tt) * K*(L+1));
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(pHatVecModq_23_device + l*K, pHatVecModq_23[l].data(), sizeof(uint64_tt) * pHatVecModq_23[l].size(), cudaMemcpyHostToDevice);		
	}

	// // pk/P
	// // inv[p012...k/p0] inv[p012...k/p1] ... inv[p012...k/pk]
	// // pk/P mod pk
	// // size = K
	// // ok
	// cudaMalloc(&pHatInvVecModp_23_device, sizeof(uint64_tt) * K);
	// cudaMemcpy(pHatInvVecModp_23_device, pHatInvVecModp_23.data(), sizeof(uint64_tt) * pHatInvVecModp_23.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&PQ_inv_mod_qi_better_device, sizeof(uint64_tt) * (L+1)*(L+1));
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(PQ_inv_mod_qi_better_device + l*(L+1), PQ_inv_mod_qi_better[l].data(), sizeof(uint64_tt) * PQ_inv_mod_qi_better[l].size(), cudaMemcpyHostToDevice);
	}
	
	cudaMalloc(&PQ_div_Qj_modqi_device, sizeof(uint64_tt) * q_num * q_num);
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(PQ_div_Qj_modqi_device + l*(L+1), PQ_div_Qj_modqi[l].data(), sizeof(uint64_tt) * PQ_div_Qj_modqi[l].size(), cudaMemcpyHostToDevice);
	}

	// qi mod qj
	// inv[q1]mod qi inv[q2]mod qi inv[q3]mod qi inv[q4]mod qi ... inv[qL]mod qi
	// ql mod qi [l(l-1)/2 + i]
	// size = L*(L+1)/2
	cudaMalloc(&qiInvVecModql_device, sizeof(uint64_tt) * L * (L + 1) / 2);
	cudaMalloc(&qiInvVecModql_shoup_device, sizeof(uint64_tt) * L * (L + 1) / 2);
	for(int l = 0; l < L; l++)
	{
		cudaMemcpy(qiInvVecModql_device + (l+1)*l/2, qiInvVecModql[l].data(), sizeof(uint64_tt) * qiInvVecModql[l].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(qiInvVecModql_shoup_device + (l+1)*l/2, qiInvVecModql_shoup[l].data(), sizeof(uint64_tt) * qiInvVecModql_shoup[l].size(), cudaMemcpyHostToDevice);
	}

	/************************************base convert from Ri to T******************************************/
	cudaMalloc(&RiHatInvVecModRi_23_device, sizeof(uint64_tt) * gamma * Ri_blockNum);
	cudaMalloc(&RiHatInvVecModRi_23_shoup_device, sizeof(uint64_tt) * gamma * Ri_blockNum);
	for(int i = 0; i < Ri_blockNum; i++)
	{
		cudaMemcpy(RiHatInvVecModRi_23_device + i * gamma, RiHatInvVecModRi_23[i].data(), sizeof(uint64_tt) * RiHatInvVecModRi_23[i].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(RiHatInvVecModRi_23_shoup_device + i * gamma, RiHatInvVecModRi_23_shoup[i].data(), sizeof(uint64_tt) * RiHatInvVecModRi_23_shoup[i].size(), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&RiHatVecModT_23_device, sizeof(uint64_tt) * gamma * t_num * Ri_blockNum);
	for(int i = 0; i < Ri_blockNum; i++)
	{
		for(int j = 0; j < t_num; j++)
		{
			cudaMemcpy(RiHatVecModT_23_device + i * gamma * t_num + j * gamma,
			RiHatVecModT_23[i][j].data(), sizeof(uint64_tt) * RiHatVecModT_23[i][j].size(), cudaMemcpyHostToDevice);
		}
	}

	/************************************base convert from Qj to T******************************************/
	cudaMalloc(&QjHatInvVecModQj_23_device, sizeof(uint64_tt) * q_num * p_num * Qj_blockNum);
	cudaMalloc(&QjHatInvVecModQj_23_shoup_device, sizeof(uint64_tt) * q_num * p_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		for(int i = 0; i < QjHatInvVecModQj_23[l].size(); i++)
		{
			cudaMemcpy(QjHatInvVecModQj_23_device + l*p_num*Qj_blockNum + i*p_num,
			QjHatInvVecModQj_23[l][i].data(), sizeof(uint64_tt) * QjHatInvVecModQj_23[l][i].size(), cudaMemcpyHostToDevice);
			cudaMemcpy(QjHatInvVecModQj_23_shoup_device + l*p_num*Qj_blockNum + i*p_num,
			QjHatInvVecModQj_23_shoup[l][i].data(), sizeof(uint64_tt) * QjHatInvVecModQj_23_shoup[l][i].size(), cudaMemcpyHostToDevice);
		}
	}

	cudaMalloc(&QjHatVecModT_23_device, sizeof(uint64_tt) * q_num * p_num * t_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		for(int i = 0; i < QjHatVecModT_23[l].size(); i++)
		{
			for(int j = 0; j < QjHatVecModT_23[l][i].size(); j++)
			{
				cudaMemcpy(QjHatVecModT_23_device + l*p_num*t_num*Qj_blockNum + i*p_num*t_num + j*p_num,
				QjHatVecModT_23[l][i][j].data(), sizeof(uint64_tt) * QjHatVecModT_23[l][i][j].size(), cudaMemcpyHostToDevice);
			}
		}
	}

	cudaMalloc(&Qjmodti_device, sizeof(uint64_tt) * q_num * t_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(Qjmodti_device + l*t_num*Qj_blockNum, Qjmodti[l].data(), sizeof(uint64_tt) * Qjmodti[l].size(), cudaMemcpyHostToDevice);
	}

	/************************************base convert from T to Ri******************************************/
	cudaMalloc(&THatInvVecModti_23_device, sizeof(uint64_tt) * t_num);
	cudaMemcpy(THatInvVecModti_23_device, THatInvVecModti_23.data(), sizeof(uint64_tt) * THatInvVecModti_23.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&THatInvVecModti_23_shoup_device, sizeof(uint64_tt) * t_num);
	cudaMemcpy(THatInvVecModti_23_shoup_device, THatInvVecModti_23_shoup.data(), sizeof(uint64_tt) * THatInvVecModti_23_shoup.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&THatVecModRi_23_device, sizeof(uint64_tt) * t_num * (K+L+1));
	for(int i = 0; i < K+L+1; i++)
	{
		cudaMemcpy(THatVecModRi_23_device + i * t_num, THatVecModRi_23[i].data(), sizeof(uint64_tt) * t_num, cudaMemcpyHostToDevice);
	}

	/**********************************************BaseConv decode Ql to T0**************************************************/
	cudaMalloc(&QlInvVecModqi_device, sizeof(uint64_tt) * (L+1)*(L+1));
	for(int i = 0; i <= L; i++)
	{
		cudaMemcpy(QlInvVecModqi_device + i * L, QlInvVecModqi[i].data(), sizeof(uint64_tt) * QlInvVecModqi[i].size(), cudaMemcpyHostToDevice);
	}
	cudaMalloc(&QlHatVecModt0_device, sizeof(uint64_tt) * (L+1)*(L+1));
	for(int i = 0; i <= L; i++)
	{
		cudaMemcpy(QlHatVecModt0_device + i * L, QlHatVecModt0[i].data(), sizeof(uint64_tt) * QlHatVecModt0[i].size(), cudaMemcpyHostToDevice);
	}

	/*************************rotGroups and ksiPows***********************/
	//rotGroups
	uint64_tt* rotGroups = new uint64_tt[Nh];
	long fivePows = 1;
	for (long i = 0; i < Nh; ++i) {
		rotGroups[i]=fivePows;
		fivePows *= 5;
		fivePows %= M;
	}
	
	cudaMalloc(&rotGroups_device, sizeof(uint64_tt) * Nh);
	cudaMemcpy(rotGroups_device, rotGroups, sizeof(uint64_tt) * Nh, cudaMemcpyHostToDevice);
	
	//ksiPows
	cuDoubleComplex* ksiPows = new cuDoubleComplex[M + 1];
	for (long j = 0; j < M; ++j) {
		double angle = 2.0 * M_PIl * j / M;
		ksiPows[j].x=cos(angle);
		ksiPows[j].y=sin(angle);
	}
	ksiPows[M] = ksiPows[0];
	
	cudaMalloc(&ksiPows_device, sizeof(cuDoubleComplex) * (M + 1));
    cudaMemcpy(ksiPows_device, ksiPows, sizeof(cuDoubleComplex)* (M + 1), cudaMemcpyHostToDevice);

	// pq
	cudaMemcpyToSymbol(pqt_cons, pqtVec.data(), sizeof(uint64_tt) * pqtVec.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pqt2_cons, pqt2Vec.data(), sizeof(uint64_tt) * pqt2Vec.size(), 0, cudaMemcpyHostToDevice);
	// pqt_mu_high
	cudaMemcpyToSymbol(pqt_mu_cons_high, pqtMuVec_high.data(), sizeof(uint64_tt) * pqtMuVec_high.size(), 0, cudaMemcpyHostToDevice);
	// pqt_mu_low
	cudaMemcpyToSymbol(pqt_mu_cons_low, pqtMuVec_low.data(), sizeof(uint64_tt) * pqtMuVec_low.size(), 0, cudaMemcpyHostToDevice);
	// T//2 mod pqti
	cudaMemcpyToSymbol(halfTmodpqti_cons, halfTmodpqti.data(), sizeof(uint64_tt) * halfTmodpqti.size(), 0, cudaMemcpyHostToDevice);
	// P mod qi
	cudaMemcpyToSymbol(Pmodqt_cons, PModqt.data(), sizeof(uint64_tt) * PModqt.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Pmodqt_shoup_cons, PModqt_shoup.data(), sizeof(uint64_tt) * PModqt_shoup.size(), 0, cudaMemcpyHostToDevice);
	// P^-1 mod qi
	cudaMemcpyToSymbol(Pinvmodqi_cons, PinvModq.data(), sizeof(uint64_tt) * PinvModq.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Pinvmodqi_shoup_cons, PinvModq_shoup.data(), sizeof(uint64_tt) * PinvModq_shoup.size(), 0, cudaMemcpyHostToDevice);
	// pk/P mod pk
	cudaMemcpyToSymbol(pHatInvVecModp_cons, pHatInvVecModp_23.data(), sizeof(uint64_tt) * pHatInvVecModp_23.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pHatInvVecModp_shoup_cons, pHatInvVecModp_23_shoup.data(), sizeof(uint64_tt) * pHatInvVecModp_23_shoup.size(), 0, cudaMemcpyHostToDevice);
	// Ri mod ti
	uint64_tt* temp_mem_device;
	cudaMalloc(&temp_mem_device, sizeof(uint64_tt) * t_num * Ri_blockNum);
	for(int i = 0; i < t_num; i++)
	{
		cudaMemcpy(temp_mem_device + i*Ri_blockNum, Rimodti[i].data(), sizeof(uint64_tt) * Rimodti[i].size(), cudaMemcpyHostToDevice);
	}
	cudaMemcpyToSymbol(Rimodti_cons, temp_mem_device, sizeof(uint64_tt) * t_num * Ri_blockNum, 0, cudaMemcpyDeviceToDevice);
	cudaFree(temp_mem_device);
	// T mod pqi
	cudaMemcpyToSymbol(Tmodpqi_cons, Tmodpqi.data(), sizeof(uint64_tt) * Tmodpqi.size(), 0, cudaMemcpyHostToDevice);
	// for sk <- HWT(h)
	randomArray_len += sizeof(uint32_tt) * h + sizeof(uint8_tt) * h;
	// for pk.a <- R_{QL}^2
	randomArray_len += sizeof(uint64_tt) * N * (L+1);
	// for pk.e <- X_{QL}
	randomArray_len += sizeof(uint32_tt) * N;
	// for swk.a <- R_{PQL}^2
	randomArray_len += sizeof(uint64_tt) * dnum * N * (L+1+K);
	// for swk.e <- X_{PQL}
	randomArray_len += sizeof(uint32_tt) * dnum * N;
	randomArray_len += sizeof(uint32_tt) * dnum * N;

	// cout<<"randomArray_len: "<<randomArray_len<<endl;

	cudaMalloc(&randomArray_device, randomArray_len);
	RNG::generateRandom_device(randomArray_device, randomArray_len);
	randomArray_sk_device = randomArray_device;
	randomArray_pk_device = randomArray_sk_device + h * sizeof(uint32_tt) / sizeof(uint8_tt) + h * sizeof(uint8_tt) / sizeof(uint8_tt);
	randomArray_e_pk_device = randomArray_pk_device + N * (L+1) * sizeof(uint64_tt) / sizeof(uint8_tt);

	randomArray_swk_device = randomArray_e_pk_device + dnum * N * sizeof(uint32_tt) / sizeof(uint8_tt);
	randomArray_e_swk_device = randomArray_swk_device + dnum * N * (L+1+K) * sizeof(uint64_tt) / sizeof(uint8_tt);

/******************************************for encode & decode********************************************/
	cudaMalloc(&decode_buffer_device, sizeof(uint64_tt) * N * (L+1));
	decode_buffer_host = new uint64_tt[N * (L+1)];
	cudaMalloc(&encode_buffer, sizeof(cuDoubleComplex) * (N>>1));
	cudaMalloc(&encode_coeffs_buffer, sizeof(double) * N);
}


// __host__ void Context_23::forwardNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num)
// {
//     uint32_tt num = poly_num * mod_num;
//     uint64_tt* device_target = device_a + (N * idx_poly); 
//     uint64_tt* psi_powers_target = pqtPsiTable_device + (N * idx_mod);
//     if(N == 65536)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(16, num);
//         CTBasedNTTInner_batch<1, 65536, 15> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<2, 65536, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<4, 65536, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<8, 65536, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

//         CTBasedNTTInnerSingle_batch<16, 65536, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 32768)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(8, num);
//         CTBasedNTTInner_batch<1, 32768, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<2, 32768, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<4, 32768, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

//         CTBasedNTTInnerSingle_batch<8, 32768, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 16384)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(4, num);
//         CTBasedNTTInner_batch<1, 16384, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//         CTBasedNTTInner_batch<2, 16384, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

//         CTBasedNTTInnerSingle_batch<4, 16384, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 8192)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(2, num);
//         CTBasedNTTInner_batch<1, 8192, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

//         CTBasedNTTInnerSingle_batch<2, 8192, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 4096)
//     {
//         dim3 single_dim(1, num);
//         CTBasedNTTInnerSingle_batch<1, 4096, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 2048)
//     {
//         dim3 single_dim(1, num);
//         CTBasedNTTInnerSingle_batch<1, 2048, 10> << <single_dim, 1024, 2048 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
//     }
// }

// __host__ void Context_23::inverseNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num)
// {
//     uint32_tt num = poly_num * mod_num;
//     uint64_tt* device_target = device_a + (N * idx_poly);
//     uint64_tt* psiinv_powers_target = pqtPsiInvTable_device + (N * idx_mod);
//     if (N == 65536)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(16, num);
//         GSBasedINTTInnerSingle_batch<16, 65536, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
//         GSBasedINTTInner_batch<8, 65536, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<4, 65536, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<2, 65536, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<1, 65536, 15> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 32768)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(8, num);
//         GSBasedINTTInnerSingle_batch<8, 32768, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
//         GSBasedINTTInner_batch<4, 32768, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<2, 32768, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<1, 32768, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 16384)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(4, num);
//         GSBasedINTTInnerSingle_batch<4, 16384, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
//         GSBasedINTTInner_batch<2, 16384, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//         GSBasedINTTInner_batch<1, 16384, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 8192)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(2, num);
//         GSBasedINTTInnerSingle_batch<2, 8192, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
//         GSBasedINTTInner_batch<1, 8192, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 4096)
//     {
//         dim3 multi_dim(N / 1024 / 2, num);
//         dim3 single_dim(1, num);
//         GSBasedINTTInnerSingle_batch<1, 4096, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
//     else if (N == 2048)
//     {
//         dim3 single_dim(1, num);
//         GSBasedINTTInnerSingle_batch<1, 2048, 10> << <single_dim, 1024, 2048 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
//     }
// }



/*****************************************************new_batch_ntt***************************************************************/
__host__ void Context_23::ToNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);

	const int tw_shmem_size = block_size / 16 * 8 * 2 * sizeof(uint64_tt);
    NTT8pointPerThread_kernel1<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)
							   + tw_shmem_size>>>
                             (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len);
    NTT8pointPerThread_kernel2<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len);
}

__host__ void Context_23::FromNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_kernel1<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len);
    INTT8pointPerThread_kernel2<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, n_inv_device, n_inv_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len);
}

__host__ void Context_23::ToNTTInplace_for_externalProduct(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len, int cipher_mod_num, int batch_size)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num, batch_size);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    NTT8pointPerThread_for_ext_kernel1<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                             (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, pad, poly_mod_len, cipher_mod_num);
    NTT8pointPerThread_for_ext_kernel2<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, second_radix_size, poly_mod_len, cipher_mod_num);
}

__host__ void Context_23::FromNTTInplace_for_externalProduct(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len, int cipher_mod_num, int batch_size)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num, 2);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num, batch_size);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_for_ext_kernel1<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len, cipher_mod_num);
    INTT8pointPerThread_for_ext_kernel2<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, n_inv_device, n_inv_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len, cipher_mod_num);
}

__host__ void Context_23::divByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num)
{
    uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);
	
	dim3 divByiAndEqual_dim(N / poly_block, mod_num, 2);
	divByiAndEqual_kernel <<< divByiAndEqual_dim, poly_block >>>(device_a, N, q_num, idx_mod, psi_powers_target);
}

__host__ void Context_23::mulByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num)
{
    uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);

	dim3 mulByiAndEqual_dim(N / poly_block, mod_num, 2);
	mulByiAndEqual_kernel <<< mulByiAndEqual_dim, poly_block >>>(device_a, N, q_num, idx_mod, psi_powers_target);
}

__host__ void Context_23::poly_add_complex_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_buffer, int idx_a, int idx_mod, int mod_num)
{
	uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);
	uint64_tt* psi_powers_shoup_target = psi_shoup_table_device + (N * idx_mod);

    dim3 add_dim(N / poly_block, mod_num);
    poly_add_complex_const_batch_device_kernel<<< add_dim, poly_block >>>(device_a, add_const_buffer, N, psi_powers_target, psi_powers_shoup_target, idx_a, L, idx_mod);
}

__host__ void Context_23::poly_mul_const_batch_device(uint64_tt* device_a, uint64_tt* const_real, int idx_mod, int mod_num)
{
    dim3 mul_dim(N / poly_block, mod_num, 2);
    poly_mul_const_batch_device_kernel<<< mul_dim, poly_block >>>(device_a, const_real, N, q_num, idx_mod);
}

// c1 += c2 * const
__host__ void Context_23::poly_mul_const_add_cipher_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, uint64_tt target_scale, int idx_mod, int mod_num)
{
    dim3 mul_dim(N / poly_block, mod_num, 2);
    poly_mul_const_batch_andAdd_device_kernel<<< mul_dim, poly_block >>>(device_a, device_b, const_real, target_scale, N, q_num, idx_mod);
}