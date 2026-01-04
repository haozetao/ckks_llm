# CKKS MMA Ringpacking 项目文档

## 项目概述

本项目是一个基于 **CKKS (Cheon-Kim-Kim-Song)** 同态加密方案的高性能矩阵乘法加速框架，专门针对 **LLM (Large Language Model)** 和 **BERT** 等大模型的隐私推理场景优化。项目采用了 **Ring Packing**、**PCMM (Packed Ciphertext Matrix Multiplication)** 和 **MLWE (Module Learning With Errors)** 等先进技术，实现了密文状态下的高效矩阵运算。

### 核心特性

- 🚀 **高性能密文矩阵乘法**：通过 PCMM 算法实现密文-明文矩阵乘法
- 🔄 **Ring Packing 技术**：将多个密文打包到一个多项式中，提升计算效率
- 🔐 **MLWE 加密方案**：基于模块学习错误问题的加密方案
- 🤖 **LLM/BERT 推理支持**：支持大语言模型的隐私推理
- ⚡ **GPU 加速**：基于 CUDA 的高性能实现
- 🎯 **Bootstrapping**：支持密文自举，实现无限深度计算

## 项目结构

```
ckks_llm/
├── src/
│   ├── ckks/
│   │   ├── include/
│   │   │   ├── Context_23.cuh/h          # CKKS 上下文管理
│   │   │   ├── Scheme_23.cuh/h           # CKKS 加密方案
│   │   │   ├── Ciphertext.cuh            # 密文数据结构
│   │   │   ├── Plaintext.cuh             # 明文数据结构
│   │   │   ├── SecretKey.cuh             # 密钥管理
│   │   │   ├── Key.cuh                   # 密钥切换
│   │   │   ├── ntt_60bit.cuh             # 数论变换 (NTT)
│   │   │   ├── poly_arithmetic.cuh       # 多项式运算
│   │   │   ├── pcmm/                     # PCMM 模块
│   │   │   │   ├── PCMM_Context.cuh/h   # PCMM 上下文
│   │   │   │   ├── PCMM_Scheme.cuh/h     # PCMM 方案
│   │   │   │   ├── PPMM_kernel.cuh       # PPMM GPU 核函数
│   │   │   │   ├── MLWECiphertext.cuh    # MLWE 密文
│   │   │   │   ├── MLWEPlaintext.cuh     # MLWE 明文
│   │   │   │   ├── MLWESecretKey.cuh     # MLWE 密钥
│   │   │   │   └── ringSwitch.cuh        # 环切换
│   │   │   ├── bootstrapping/            # 自举模块
│   │   │   │   ├── Bootstrapper.cuh/h    # 自举器
│   │   │   │   ├── Bootstrapping.cuh     # 自举实现
│   │   │   │   └── Bootstrapping_encoding.cuh  # 自举编码
│   │   │   └── attention/               # 注意力机制
│   │   │       ├── Attention.cuh/h       # 注意力实现
│   │   ├── bert/                         # BERT 相关
│   │   │   ├── Bert.cuh/h                # BERT 模型
│   │   │   └── Load_bert.cuh             # BERT 模型加载
│   │   └── llm/                          # LLM 相关
│   │       ├── llm_infer.cuh             # LLM 推理
│   │       ├── llama-3-8b.cuh            # Llama-3-8B 模型
│   │       ├── opt.cuh                   # OPT 模型
│   │       └── load_bert.cuh             # 模型加载
│   └── test_*.cu                         # 测试文件
├── python/
│   ├── ccmm.py                           # CCMM 算法实现
│   ├── ringpacking.py                    # Ring Packing 实现
│   ├── *.ipynb                           # Jupyter notebooks
│   └── data/                             # 测试数据
├── run.sh                                # 编译运行脚本
└── README.md
```

## 核心模块详解

### 1. Context_23 - CKKS 上下文管理

**文件**: [Context_23.cuh](src/ckks/include/Context_23.cuh), [Context_23.h](src/ckks/include/Context_23.h)

**功能**:
- 管理加密参数（环维度 N、模数链、精度等）
- 预计算 NTT 相关参数（psi, psi_inv 等）
- 管理模数切换和基转换所需的预计算表
- 编码/解码功能（复数和系数编码）

**核心参数**:
```cpp
int logN;              // 环维度的对数
int N;                 // 环维度 (2^logN)
int L;                 // 最大支持层数
int q_num;             // q 模数数量
int p_num;             // p 模数数量
int K;                 // 特殊模数数量
int gamma;             // gamma 参数
vector<uint64_tt> qVec; // q 模数向量
vector<uint64_tt> pVec; // p 模数向量
```

**核心方法**:
- `encode()`: 编码复数到明文
- `decode()`: 解码明文到复数
- `forwardNTT_batch()`: 批量 NTT 变换
- `inverseNTT_batch()`: 批量逆 NTT 变换
- `modUpPQtoT_23()`: 模数提升 (PQ -> T)
- `modDownPQltoQl_23()`: 模数降低 (PQl -> Ql)

### 2. Scheme_23 - CKKS 加密方案

**文件**: [Scheme_23.cuh](src/ckks/include/Scheme_23.cuh), [Scheme_23.h](src/ckks/include/Scheme_23.h)

**功能**:
- 实现完整的 CKKS 加密/解密操作
- 密文运算（加法、乘法、旋转等）
- 密钥生成和密钥切换
- Rescale 操作

**核心方法**:
- `encrypt()`: 加密
- `decrypt()`: 解密
- `add()`: 密文加法
- `mult()`: 密文乘法
- `addConst()`: 密文加常数
- `multConst()`: 密文乘常数
- `rescale()`: Rescale 操作
- `rotate()`: 槽位旋转
- `conjugate()`: 共轭操作

### 3. PCMM_Context - 密文矩阵乘法上下文

**文件**: [PCMM_Context.cuh](src/ckks/include/pcmm/PCMM_Context.cuh), [PCMM_Context.h](src/ckks/include/pcmm/PCMM_Context.h)

**功能**:
- 管理 PCMM 算法的上下文参数
- Ring Packing 相关的预计算
- MLWE 多项式的 NTT 变换

**核心参数**:
```cpp
int N1;                    // MLWE 多项式维度
int mlwe_rank;             // MLWE 秩
int ringpack_p_count;      // Ring packing p 模数数量
int ringpack_q_count;      // Ring packing q 模数数量
vector<uint64_tt> p_ringpack;  // Ring packing p 模数
vector<uint64_tt> q_ringpack;  // Ring packing q 模数
```

**核心方法**:
- `encodeCoeffs()`: 编码系数到 MLWE 明文
- `decodeCoeffs()`: 解码 MLWE 明文到系数
- `ToNTTInplace()`: 原地 NTT 变换
- `FromNTTInplace()`: 原地逆 NTT 变换

### 4. PCMM_Scheme - 密文矩阵乘法方案

**文件**: [PCMM_Scheme.cuh](src/ckks/include/pcmm/PCMM_Scheme.cuh), [PCMM_Scheme.h](src/ckks/include/pcmm/PCMM_Scheme.h)

**功能**:
- 实现 RLWE 到 MLWE 的分解
- 实现 MLWE 到 RLWE 的打包
- 实现 PPMM (Plain-Packed Matrix Multiplication)

**核心方法**:
- `rlweCipherDecompose()`: 将 RLWE 密文分解为多个 MLWE 密文
- `mlweCipherPacking()`: 将多个 MLWE 密文打包为一个 RLWE 密文
- `PPMM()`: 密文-明文矩阵乘法
- `addRepakcingKey()`: 添加重打包密钥
- `convertMLWESKfromRLWESK()`: 从 RLWE 密钥转换到 MLWE 密钥

**PPMM 函数签名**:
```cpp
__host__ void PPMM(
    float* plain_mat,                          // 明文矩阵
    vector<MLWECiphertext*> mlwe_cipher_decomposed,  // 分解后的 MLWE 密文
    int mat_M,                                 // 矩阵 M 维度
    int mat_N,                                 // 矩阵 N 维度
    int mat_K,                                 // 矩阵 K 维度
    int mlwe_num                               // MLWE 数量
);
```

### 5. Bootstrapper - 自举器

**文件**: [Bootstrapper.cuh](src/ckks/include/bootstrapping/Bootstrapper.cuh), [Bootstrapper.h](src/ckks/include/bootstrapping/Bootstrapper.h)

**功能**:
- 实现 CKKS 自举操作
- 密文刷新，支持无限深度计算
- 槽位到系数 (C2S) 和系数到槽位 (S2C) 转换

**核心方法**:
- `bootstrap()`: 自举操作
- `addBootstrappingKey()`: 添加自举密钥
- `C2S()`: 槽位到系数转换
- `S2C()`: 系数到槽位转换

### 6. Attention - 注意力机制

**文件**: [Attention.cuh](src/ckks/include/attention/Attention.cuh), [Attention.h](src/ckks/include/attention/Attention.h)

**功能**:
- 实现 Transformer 的注意力机制
- 支持 BERT 和 LLM 的注意力计算
- 密文状态下的注意力计算

**核心方法**:
- `computeAttention()`: 计算注意力
- `addKey()`: 添加注意力密钥
- `softmax()`: 密文 Softmax 操作

## 技术原理

### Ring Packing

Ring Packing 是将多个较小的多项式打包到一个更大的多项式中的技术。在本项目中：

1. **分解**: 将一个 RLWE 密文（环维度 N）分解为多个 MLWE 密文（环维度 N1）
   - `N = N1 * mlwe_rank`
   - 例如：N=32768, N1=256, mlwe_rank=128

2. **打包**: 将多个 MLWE 密文重新打包为一个 RLWE 密文

**优势**:
- 减少密文数量，降低通信开销
- 提升并行计算效率
- 优化内存访问模式

### PCMM (Packed Ciphertext Matrix Multiplication)

PCMM 是密文-明文矩阵乘法的高效实现：

```
C = A × B
```

其中：
- A 是密文矩阵（打包后的 MLWE 密文）
- B 是明文矩阵（普通浮点数矩阵）
- C 是结果密文矩阵

**实现步骤**:
1. 将 RLWE 密文分解为 MLWE 密文
2. 对每个 MLWE 密文执行矩阵乘法
3. 将结果打包回 RLWE 密文

**性能优化**:
- GPU 并行计算
- Shared Memory 优化
- Double Buffering 技术
- Coalesced Memory Access

### MLWE (Module Learning With Errors)

MLWE 是 LWE 的模块化版本，具有以下特点：

- **结构**: 多个 LWE 实例的向量
- **安全性**: 基于模块格问题
- **效率**: 支持批处理和并行计算

在本项目中，MLWE 用于：
- Ring Packing 的中间表示
- 优化矩阵乘法计算

### NTT (Number Theoretic Transform)

NTT 是快速傅里叶变换在有限域上的实现，用于加速多项式乘法：

- **正向 NTT**: 将多项式从系数域转换到点值域
- **逆向 NTT**: 将多项式从点值域转换回系数域
- **优势**: 多项式乘法从 O(N²) 降低到 O(N log N)

**实现特点**:
- 支持 60-bit 模数
- 预计算 psi 和 psi_inv
- 批量处理优化

## 编译和运行

### 编译

使用提供的 `run.sh` 脚本：

```bash
./run.sh <test_file>
```

例如：
```bash
./run.sh test_rlwe_decomp.cu
```

### 测试文件

项目提供了多个测试文件：

- `test_rlwe_decomp.cu`: 测试 RLWE 分解和 MLWE 打包
- `test_ccmm.cu`: 测试密文矩阵乘法
- `test_boot.cu`: 测试自举功能
- `test_bert_infer.cu`: 测试 BERT 推理
- `test_nonlinear.cu`: 测试非线性函数

### 运行示例

```bash
# 测试 RLWE 分解
./run.sh test_rlwe_decomp.cu

# 测试 CCMM
./run.sh test_ccmm.cu

# 测试 BERT 推理
./run.sh test_bert_infer.cu
```

## 性能优化

### GPU 优化

1. **Shared Memory**: 利用共享内存减少全局内存访问
2. **Coalesced Access**: 合并内存访问模式
3. **Warp Shuffle**: 使用 warp shuffle 指令
4. **Double Buffering**: 双缓冲技术隐藏延迟

### 算法优化

1. **Batch Processing**: 批量处理多个密文
2. **Precomputation**: 预计算常用参数
3. **Memory Reuse**: 内存复用减少分配开销
4. **NTT Optimization**: 优化的 NTT 实现

## 应用场景

### 1. 隐私推理

在不解密的情况下对加密数据进行模型推理：

- **医疗诊断**: 保护患者隐私
- **金融风控**: 保护用户数据
- **推荐系统**: 保护用户行为数据

### 2. 联邦学习

多方协作训练模型，保护各方数据隐私：

- **模型聚合**: 密文模型参数聚合
- **梯度计算**: 密文梯度计算
- **模型更新**: 密文模型更新

### 3. 安全外包计算

将计算任务外包给云端，保护数据隐私：

- **矩阵运算**: 密文矩阵乘法
- **深度学习**: 密文神经网络推理
- **数据分析**: 密文统计分析

## 依赖项

- CUDA Toolkit
- C++ 编译器 (支持 C++11 或更高版本)
- NTL (Number Theory Library)
- cuBLAS (可选，用于矩阵运算)

## 常见问题

### Q1: 编译时出现 "declaration is incompatible" 错误

**原因**: 函数声明和定义的参数不匹配

**解决**: 确保头文件和实现文件的函数签名一致

例如，在 `PCMM_Scheme.h` 中：
```cpp
void PPMM(float* plain_mat, vector<MLWECiphertext*> mlwe_cipher_decomposed, int mat_M, int mat_N, int mat_K, int mlwe_num);
```

在 `PPMM_kernel.cuh` 中：
```cpp
void PCMM_Scheme::PPMM(float* plain_mat, vector<MLWECiphertext*> mlwe_cipher_decomposed, int mat_M, int mat_N, int mat_K, int mlwe_num)
```

### Q2: 运行时出现 "too few arguments in function call" 错误

**原因**: 函数调用时参数数量不足

**解决**: 确保调用时传递所有必需的参数

例如：
```cpp
pcmm_scheme.PPMM(plain_mat_device, mlwe_cipher_decomposed, mat_M, mat_N, PCMM_N1, decomp_num);
```

### Q3: GPU 内存不足

**原因**: 密文和预计算表占用大量 GPU 内存

**解决**:
- 减少环维度 N
- 减少模数链长度 L
- 使用更小的批次大小

## 未来工作

- [ ] 支持更多 LLM 模型（GPT、Llama 等）
- [ ] 优化 Bootstrapping 性能
- [ ] 支持分布式计算
- [ ] 添加更多非线性函数
- [ ] 提供更友好的 Python API

## 参考文献

1. Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. ASIACRYPT.
2. Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2016). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. PKC.
3. Albrecht, M. R., et al. (2018). Homomorphic Encryption Standardization.

## 许可证

请查看项目根目录下的 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]

---

**最后更新**: 2025-12-29
