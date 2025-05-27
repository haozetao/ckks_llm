#include <iostream>
#include <random>

using namespace std;

int32_t q = 0x3ffc0001;
// , 0x3fde0001, 0x3fd20001, 0x3fac0001, 0x3f820001, 0x3f760001, 0x3f5a0001, 0x3f540001, 0x3f3a0001, 0x3ef80001, 0x3ef40001, 0x3ee60001, 0x3ed60001,


uint64_t karatsuba(uint8_t x, uint8_t y) {
    // 处理简单的单字节乘法（递归终止条件）
    return static_cast<uint16_t>(x) * static_cast<uint16_t>(y);
}

void karatsuba_32(){
    int32_t a, b;

    random_device seed;//硬件生成随机数种子
    mt19937 gen(seed()); // Mersenne Twister 引擎
    uniform_int_distribution<> distrib(0, q);//设置随机数范围，并为均匀分布
    a = distrib(gen);//随机数
    b = distrib(gen);

    uint64_t c = uint64_t(a) * b;
    
    uint64_t temp_a = a, temp_b = b;

    uint8_t ra[4];
    uint8_t rb[4];

    for(int i = 0; i < 4; i++){
        ra[i] = a & 0xff;
        a >>= 8;

        rb[i] = b & 0xff;
        b >>= 8;
        // printf("%x, %x\n", ra[i], rb[i]);
    }

    uint64_t acc = 0;
    // karatsuba 核心部分
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         acc += karatsuba(ra[i], rb[j]) << (8 * (i + j));
    //     }
    // }

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < i+1; j++){
            int idx1 = i - j;
            int idx2 = j;

            acc += karatsuba(ra[idx1], rb[idx2]) << (8 * i);
            printf("%d %d | %d %d %d\n", i, j, idx1, idx2, 8 * (10 - i));
        }
        printf("\n");
    }

    for(int i = 4; i < 7; i++) {   // Outer loop (0, 1, 2)
        for(int j = 4; j < i+1; j++) {  // Inner loop (decreasing length as i increases)
            int idx1 = (7 - j);
            int idx2 = (j - i + 3);

            acc += karatsuba(ra[idx1], rb[idx2]) << (8 * (10 - i));
            printf("%d %d | %d %d %d\n", i, j, idx1, idx2, 8 * (10 - i));
        }
        printf("\n");
    }

    printf("Direct: %llx * %llx = %llx\n", temp_a, temp_b, uint64_t(temp_a)*temp_b);
    printf("Karatsuba: %llx\n", acc);
}


void karatsuba_64()
{
    uint64_t a, b;

    // 生成随机数
    std::random_device seed;
    std::mt19937_64 gen(seed());
    std::uniform_int_distribution<uint64_t> distrib(0, UINT64_MAX);
    a = distrib(gen);
    b = distrib(gen);

    // 直接计算（用于验证）
    __uint128_t c = static_cast<__uint128_t>(a) * b;
    
    // 分解 a 和 b 为 8 个 uint8_t
    uint8_t ra[8];
    uint8_t rb[8];

    uint64_t temp_a = a;
    uint64_t temp_b = b;
    for (int i = 0; i < 8; i++) {
        ra[i] = temp_a & 0xff;
        temp_a >>= 8;

        rb[i] = temp_b & 0xff;
        temp_b >>= 8;
    }

    // 128 位累加器（低 64 位 + 高 64 位）
    __uint128_t acc = 0;

    // Karatsuba 核心部分
    // 1. 处理前 8 个部分（0 ≤ i < 8）
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j <= i; j++) {
            int idx1 = i - j;
            int idx2 = j;
            acc += static_cast<__uint128_t>(karatsuba(ra[idx1], rb[idx2])) << (8 * i);
            printf("%d %d | %d %d %d\n", i, j, idx1, idx2, 8 * i);

        }
        printf("\n");

    }

    // 2. 处理后 7 个部分（8 ≤ i < 15）
    for (int i = 8; i < 15; i++) {
        for (int j = 8; j < i + 1; j++) {
            int idx1 = 15 - j;
            int idx2 = j - i + 7;
            acc += static_cast<__uint128_t>(karatsuba(ra[idx1], rb[idx2])) << (8 * (22 - i));
            printf("%d %d | %d %d %d\n", i, j, idx1, idx2, 8 * (22 - i));

        }
        printf("\n");
    }
    printf("Direct: %llx * %llx = %llx %llx\n", a, b, (uint64_t)(c >> 64), (uint64_t)c);
    // 输出结果
    printf("Karatsuba: %llx %llx\n", (uint64_t)(acc >> 64), (uint64_t)acc);
}

int main()
{
    // karatsuba_32();
    karatsuba_64();
}