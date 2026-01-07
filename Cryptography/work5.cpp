#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <windows.h>
// 保持 find_odd_coefficient 函数不变，因为它只负责系数提取

int find_odd_coefficient(uint64_t x, int w, uint64_t M_w, int B_w) {
    if ((x & 1) == 0) {
        // 通常在主循环中处理
        throw std::logic_error("find_odd_coefficient 必须只对奇数 x 调用。"); 
    }

    uint64_t t_mod_M = x % M_w;

    int t;
    if (t_mod_M >= B_w) {
        t = static_cast<int>(t_mod_M) - static_cast<int>(M_w);
    } else {
        t = static_cast<int>(t_mod_M);
    }
    
    return t;
}

/**
 * @brief 设计一个 wNAF 译码算法，并按照高位->低位顺序输出。
 * * 将指数 k 分解成 wNAF 形式：k = sum(di * 2^i)
 *
 * @param k 输入的指数 (标量)，k < 2^63
 * @param w 窗口宽度 (w >= 2, 通常 w <= 8)
 * @return std::vector<int> 包含 wNAF 系数 (d_{m-1}, ..., d0) 的向量。
 */
std::vector<int> wNAF_encoding(uint64_t k, int w) {
    // 1. 检查 w 的有效性
    if (w < 2 || w > 63) {
        std::cerr << "错误：w 必须在 2 到 63 之间（wNAF通常要求 w >= 2）。" << std::endl;
        return {};
    }

    // 2. 计算必要的常量
    uint64_t M_w = 1ULL << w;
    int B_w = 1 << (w - 1); 

    std::vector<int> wnaf_representation;
    uint64_t current_k = k;

    // 3. 循环提取系数 (D0, D1, D2, ... 低位优先)
    while (current_k > 0) {
        if (current_k & 1) {
            // 奇数位，提取非零系数 t
            int t;
            try {
                t = find_odd_coefficient(current_k, w, M_w, B_w);
            } catch (const std::logic_error& e) {
                std::cerr << "内部错误: " << e.what() << std::endl;
                return {};
            }
            
            wnaf_representation.push_back(t);
            current_k = (current_k - static_cast<uint64_t>(t)) >> 1;

        } else {
            // 偶数位，系数 d_i = 0
            wnaf_representation.push_back(0);
            current_k >>= 1;
        }
    }
    
    // 关键修改：反转向量，使顺序变为 (D_{m-1}, ..., D0) 高位优先
    std::reverse(wnaf_representation.begin(), wnaf_representation.end());
    
    return wnaf_representation;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    uint64_t k;
    int w;

    std::cout << "请输入指数 k (< 2^63): ";
    if (!(std::cin >> k)) { return 1; }
    
    std::cout << "请输入窗口宽度 w (w >= 2): ";
    if (!(std::cin >> w)) { return 1; }

    std::cout << "\n--- wNAF 译码表示 (高位->低位) ---" << std::endl;
    std::cout << "输入 k (十进制): " << k << std::endl;
    std::cout << "窗口宽度 w: " << w << std::endl;

    // 进行 wNAF 编码
    std::vector<int> wnaf_result = wNAF_encoding(k, w);

    // 输出结果
    if (!wnaf_result.empty()) {
        std::cout << "wNAF 译码表示 (d_{m-1}, ..., d0): [";
        // 打印结果向量
        for (size_t i = 0; i < wnaf_result.size(); ++i) {
            std::cout << wnaf_result[i];
            if (i < wnaf_result.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        
        // 验证分解是否正确：k = sum(di * 2^i)
        // 注意：验证过程不变，因为它依赖于系数和 2^i 的乘积，与存储顺序无关。
        int64_t check_k = 0;
        // 注意：这里需要从高位（左侧）开始计算 2^i
        
        // 重新计算 2^i，但为了简单和通用性，我们使用向量的下标 i 来计算 2^i。
        // 最高位系数 d_{m-1} 对应 2^{m-1}
        // 最低位系数 d_0 对应 2^0
        int m = wnaf_result.size(); // 序列长度
        
        for (int i = 0; i < m; ++i) {
            // d_i 实际上是 wnaf_result[m - 1 - i]
            // wnaf_result[0] 是 d_{m-1}，对应 2^{m-1}
            // wnaf_result[m-1] 是 d_0，对应 2^0
            int di = wnaf_result[m - 1 - i]; // 提取 d_i
            int64_t power_of_2 = 1ULL << i; // 计算 2^i
            
            check_k += static_cast<int64_t>(di) * power_of_2;
        }

        std::cout << "分解验证 (重构的 k): " << check_k << std::endl;
        std::cout << "验证结果: " << (check_k == static_cast<int64_t>(k) ? "通过" : "失败") << std::endl;
    }

    return 0;
}