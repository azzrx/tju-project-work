#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint> // 用于 uint64_t
#include <windows.h>
/**
 * @brief 设计一个 k-ary (k进制) 表示的分解算法。
 * * 将指数 e 分解成一系列 k-bit 的系数 di。
 * e = sum(di * (2^k)^i)
 *
 * @param e 输入的指数，e < 2^63
 * @param k 窗口宽度，即每个系数的二进制位数 (1 <= k <= 63)
 * @return std::vector<uint32_t> 包含 k-ary 系数 (d0, d1, d2, ...) 的向量。
 */
std::vector<uint32_t> k_ary_decomposition(uint64_t e, int k) {
    // 1. 检查 k 的有效性
    if (k <= 0 || k >= 64) {
        std::cerr << "错误：k 必须在 1 到 63 之间。" << std::endl;
        return {};
    }

    // 2. 计算模数 (Modulus) M = 2^k
    // M 用于取模操作，提取当前块的值。
    // 由于 e < 2^63，我们可以使用 uint64_t 来处理 e 和 M。
    uint64_t M = 1ULL << k; // 2^k

    std::vector<uint32_t> decomposition;

    // 3. 循环分解：使用 e = (e / M) * M + (e % M) 的原理
    uint64_t current_e = e;
    while (current_e > 0) {
        // 取当前 e 的最低 k 位作为系数 di
        uint64_t di_64 = current_e % M;
        
        // 将系数存入向量。由于 di < 2^k 且 k <= 63，它可以安全地存入 uint32_t (甚至 uint64_t)
        // 使用 static_cast<uint32_t> 确保类型匹配，因为 di_64 必定小于 2^63
        // 且 k <= 32 时 di_64 < 2^32，可存入 uint32_t。为了通用性，若 k>32，则需使用 uint64_t。
        // 为了简化和常用性（一般 k<=32），这里使用 uint32_t，但若 k > 32 且要保证兼容性，应使用 uint64_t。
        // 考虑到题目要求 e < 2^63 且通常 k 在 4 到 16 之间，uint32_t 是够用的。
        decomposition.push_back(static_cast<uint32_t>(di_64));
        
        // 更新 e：去掉已提取的 k 位 (即整除 M)
        current_e /= M;
    }

    // 4. 结果是低位在前 (d0, d1, ...) 的顺序。
    // 如果想要高位在前 (..., d2, d1, d0) 的表示，需要反转。
    // 在模幂运算中，通常从高位开始处理，但分解结果通常是低位在前。
    // 保持低位在前，但如果需要高位在前，可以取消下面的注释。
    // std::reverse(decomposition.begin(), decomposition.end());
    
    return decomposition;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    // e < 2^63，使用 uint64_t 存储
    uint64_t e = 123456789012345ULL; // 一个大数字
    int k = 8; // 窗口宽度 k=8，即分解为 8-bit 的块 (相当于 256-ary 表示)

    // 从用户输入获取 e 和 k (十进制输入)
    std::cout << "请输入指数 e (< 2^63): ";
    std::cin >> e;
    std::cout << "请输入窗口宽度 k (1 <= k <= 63): ";
    std::cin >> k;

    // 进行 k-ary 分解
    std::vector<uint32_t> result = k_ary_decomposition(e, k);

    // 输出结果
    std::cout << "--- k-ary 和窗口译码表示 ---" << std::endl;
    std::cout << "输入 e (十进制): " << e << std::endl;
    std::cout << "窗口宽度 k: " << k << std::endl;
    
    if (!result.empty()) {
        std::cout << "分解结果 (d0, d1, d2, ...): [";
        // 打印结果向量，即每个 di 的十进制值
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i];
            if (i < result.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        
        // 验证分解是否正确：e = d0 + d1*M + d2*M^2 + ...
        uint64_t check_e = 0;
        uint64_t M = 1ULL << k; // 2^k
        uint64_t M_power_i = 1; // 相当于 M^i
        
        for (uint32_t di : result) {
            check_e += static_cast<uint64_t>(di) * M_power_i;
            // 更新 M^i 到 M^(i+1)。防止溢出：M_power_i <= 2^63。
            if (M_power_i > (UINT64_MAX / M) && check_e != e) { 
                // 仅用于防止理论上的乘法溢出检查，对于 e < 2^63 通常不会发生
                break;
            }
            M_power_i *= M; 
        }

        std::cout << "分解验证 (重构的 e): " << check_e << std::endl;
        std::cout << "验证结果: " << (check_e == e ? "通过" : "失败") << std::endl;
    }

    return 0;
}