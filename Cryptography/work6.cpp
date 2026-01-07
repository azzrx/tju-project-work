#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric> // 用于 std::max

// 保留此行以支持中文输出，但这不是标准C++特性
#include <windows.h> 

using namespace std;

// 定义无符号长整型，用于处理模运算，P <= 2^32
typedef unsigned long long ull;

// --- 辅助函数：核心数学操作 ---

// --- 辅助函数：核心数学操作 ---

// 辅助函数 1：模乘法 (a * b) mod p
ull ModMul(ull a, ull b, ull p) {
    return (a * b) % p;
}

// 注意：ExtendedGcd 的参数 x 和 y 必须使用 long long 确保能存储负值
// 辅助函数 2：扩展欧几里得算法 (Extended Euclidean Algorithm)
// 计算 ax + my = gcd(a, m)
ull ExtendedGcd(ull a, ull m, long long& x, long long& y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return m;
    }
    long long x1, y1;
    // 递归调用
    ull gcd = ExtendedGcd(m % a, a, x1, y1);
    
    // 关键修正：确保在 long long 范围内进行计算
    x = y1 - (long long)(m / a) * x1;
    y = x1;
    return gcd;
}

// 辅助函数 3：模逆元 (a^{-1} mod m)
// 传入的 a, m 是 ull，但计算 x, y 使用 long long
ull ModInverse(ull a, ull m) {
    long long x, y; // 必须是 long long
    ull g = ExtendedGcd(a, m, x, y);
    
    if (g != 1) {
        // 模逆元不存在
        return 0; 
    }
    
    // 关键修正：将可能为负的 x 转换为 [0, m-1] 范围内的正数
    // (x % m + m) % m 适用于 x 为 long long 的情况
    return (ull)((x % (long long)m + (long long)m) % (long long)m);
}
// ... 剩下的辅助函数 NAF 保持不变 ...
// 辅助函数 4：NAF 编码 (Non-Adjacent Form)
/**
 * @brief 将指数 e 编码为 NAF 序列。
 * @param e 要编码的指数
 * @return vector<int> NAF 序列，元素为 {0, 1, -1}
 */
vector<int> NAF(ull e) {
    vector<int> naf_seq;
    while (e > 0) {
        if (e & 1) { // 奇数
            // z = e mod 4，如果 z=1，则 z=1；如果 z=3，则 z=-1
            int z = (e % 4 == 1) ? 1 : -1; 
            naf_seq.push_back(z);
            e = (e - z) / 2;
        } else { // 偶数
            naf_seq.push_back(0);
            e = e / 2;
        }
    }
    return naf_seq;
}

// --- 算法 1：Shamir 窍门 (总乘法次数统计) ---

/**
 * @brief 算法1 Shamir窍门：计算 R = g^a * h^b mod p，统计平方和乘法。
 */
ull ShamirTrick_TotalCount(ull g, ull h, ull a, ull b, ull p, int& mul_count) {
    mul_count = 0; 
    
    ull gh = ModMul(g, h, p); 
    ull A = 1;

    int t_a = (a > 0) ? floor(log2(a)) : 0;
    int t_b = (b > 0) ? floor(log2(b)) : 0;
    int t = max(t_a, t_b); 

    // 预处理 a 和 b 的位
    vector<int> a_bits(t + 1);
    vector<int> b_bits(t + 1);
    for (int i = 0; i <= t; ++i) {
        a_bits[i] = (a >> i) & 1;
        b_bits[i] = (b >> i) & 1;
    }

    // 循环从最高位 t down to 0
    for (int i = t; i >= 0; --i) {
        
        // (3.1) 平方操作：A <- A * A
        if (i != t) {
            A = ModMul(A, A, p);
            mul_count++; // 统计平方操作
        }

        // (3.2) 乘法操作：A <- A * g^a_i * h^b_i
        int a_i = a_bits[i];
        int b_i = b_bits[i];

        if (a_i == 1 && b_i == 0) { // 乘 g
            A = ModMul(A, g, p); mul_count++;
        } else if (a_i == 0 && b_i == 1) { // 乘 h
            A = ModMul(A, h, p); mul_count++;
        } else if (a_i == 1 && b_i == 1) { // 乘 g*h
            A = ModMul(A, gh, p); mul_count++;
        }
    }
    return A;
}

// --- 算法 2：扩展 Shamir 窍门 (真正实现 NAF 优化) ---

/**
 * @brief 算法2 扩展Shamir窍门：计算 R = g^a * h^b mod p (基于 NAF 优化)
 * @param d_naf, f_naf: 预先计算好的 NAF 序列
 * @param mul_count 传引用，统计总乘法次数（包含平方和乘法）
 * @return ull 结果 R
 */
ull ExtendedShamirTrick_NAF(ull g, ull h, ull p, 
                            const vector<int>& d_naf, const vector<int>& f_naf, 
                            int& mul_count) {
    mul_count = 0; 

    // --- 预计算 ---
    
    // 计算并存储 g^{-1}, h^{-1}
    ull g_inv = ModInverse(g, p);
    ull h_inv = ModInverse(h, p);

    // 预计算所有可能的乘数 (共 8 种组合)
    ull P_table[8];
    P_table[0] = ModMul(g, h, p);           // g^1 * h^1
    P_table[1] = ModMul(g, h_inv, p);       // g^1 * h^{-1}
    P_table[2] = ModMul(g_inv, h, p);       // g^{-1} * h^1
    P_table[3] = ModMul(g_inv, h_inv, p);   // g^{-1} * h^{-1}
    P_table[4] = g;                         // g^1 * h^0
    P_table[5] = g_inv;                     // g^{-1} * h^0
    P_table[6] = h;                         // g^0 * h^1
    P_table[7] = h_inv;                     // g^0 * h^{-1}

    ull A = 1;

    // 循环的最大长度 T
    int T = max(d_naf.size(), f_naf.size()); 

    // (4) For i from T-1 down to 0
    for (int i = T - 1; i >= 0; --i) {
        
        // (4.1) 平方操作：A <- A * A
        if (i != T - 1) { 
            A = ModMul(A, A, p);
            mul_count++; // 统计平方操作
        }

        // (4.2) 乘法操作：A <- A * g^d_i * h^f_i
        int d_i = (i < d_naf.size()) ? d_naf[i] : 0;
        int f_i = (i < f_naf.size()) ? f_naf[i] : 0;
        
        if (d_i != 0 || f_i != 0) { // 如果 (d_i, f_i) 不全为零
            int idx = -1;
            
            // 查表逻辑 (9 种情况，排除 (0,0))
            if (d_i == 1) {
                if (f_i == 1) idx = 0;
                else if (f_i == -1) idx = 1;
                else idx = 4; // f_i == 0
            } else if (d_i == -1) {
                if (f_i == 1) idx = 2;
                else if (f_i == -1) idx = 3;
                else idx = 5; // f_i == 0
            } else { // d_i == 0
                if (f_i == 1) idx = 6;
                else if (f_i == -1) idx = 7;
            }

            if (idx != -1) {
                A = ModMul(A, P_table[idx], p);
                mul_count++; // 统计乘法操作
            }
        }
    }
    return A;
}

// --- 算法 3：CRT 加速 RSA 算法 ---

// 模幂运算 (base^exp) mod p (内部使用，不统计乘法次数)
ull ModPow(ull base, ull exp, ull p) {
    ull res = 1;
    base %= p;
    
    while (exp > 0) {
        if (exp & 1) { 
            res = ModMul(res, base, p);
        }
        base = ModMul(base, base, p);
        exp >>= 1; 
    }
    return res;
}

/**
 * @brief 算法3 CRT加速RSA解密：计算 M = C^d mod (p*q)
 */
ull CRT_Decryption(ull C, ull p, ull q, ull d1, ull d2, ull q_inv) {
    
    // (1) 计算 M1 = C^d1 mod p
    ull M1 = ModPow(C, d1, p); 

    // (2) 计算 M2 = C^d2 mod q
    ull M2 = ModPow(C, d2, q);

    // (3) 计算 M = M2 + [((M1 - M2) * q^{-1} mod p) mod p] * q
    
    // 计算 (M1 - M2) mod p
    ull diff = (M1 >= M2) ? (M1 - M2) : (M1 + p - M2);
    
    // 计算 h = (M1 - M2) * q^{-1} mod p
    ull h = ModMul(diff, q_inv, p);
    
    // 计算 M = M2 + h * q
    ull M = M2 + h * q; 
    
    return M;
}

// --- 主函数：测试与运行 ---

// --- 主函数：测试与运行 ---

int main() {
    // 设置控制台编码为 UTF-8 以正确显示中文（仅在 Windows 下）
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
    #endif

    // --- 测试 Shamir 窍门和扩展 Shamir 窍门 (算法 1 & 2) ---
    cout << "--- 测试 Shamir 窍门和扩展 Shamir 窍门 (总乘法次数) ---" << endl;
    
    // 测试用例输入
    ull g, h, a, b, p;
    // 推荐使用图中的测试用例: 2 5 569858951 734233321 3586654197
    cout << "请输入g,h,a,b,p: ";
    if (!(cin >> g >> h >> a >> b >> p)) {
        // 如果输入失败，使用默认值
        g = 2; h = 5; a = 569858951; b = 734233321; p = 3586654197;
        cout << "输入失败或不足，使用默认测试值: " << g << " " << h << " " << a << " " << b << " " << p << endl;
    }

    int m_count_shamir = 0;
    int n_count_extended = 0;

    // --- 逆元检查 (检查算法2的基石) ---
    ull g_inv = ModInverse(g, p);
    ull h_inv = ModInverse(h, p);

    cout << "--- 关键验证 (模逆元) ---" << endl;
    if (g_inv == 0 || h_inv == 0) {
        cout << "❗ 警告：g 或 h 与模数 p 不互素，无法进行 NAF 优化！" << endl;
        cout << " g=" << g << ", h=" << h << ", p=" << p << endl;
    }
    cout << "g 的逆元 g_inv (mod p) = " << g_inv << endl;
    cout << "h 的逆元 h_inv (mod p) = " << h_inv << endl;
    cout << "------------------------------------------" << endl;

    // --- NAF 预计算 (解决作用域问题) ---
    vector<int> d_naf = NAF(a);
    vector<int> f_naf = NAF(b);
    
    // 运行 Shamir 窍门 (算法 1)
    ull R_shamir = ShamirTrick_TotalCount(g, h, a, b, p, m_count_shamir);
    cout << "Shamir 窍门 (算法 1):" << endl;
    cout << "  g^a * h^b (mod p) = " << R_shamir << endl;
    cout << "  总乘法次数 m = " << m_count_shamir << " (预期 54, 图中 53)" << endl; 
    
    // 运行 扩展 Shamir 窍门 (算法 2 - 真正 NAF)
    ull R_extended = ExtendedShamirTrick_NAF(g, h, p, d_naf, f_naf, n_count_extended);
    cout << "扩展 Shamir 窍门 (算法 2 - NAF 优化):" << endl;
    cout << "  g^a * h^b (mod p) = " << R_extended << endl;
    cout << "  总乘法次数 n = " << n_count_extended << " (预期 45-46)" << endl; 
    
    // 🚀 最终结论：如果两个结果一致，说明修正成功。
    if (R_shamir == R_extended) {
        cout << "\n✅ 校验成功：两个算法的结果一致！" << endl;
    } else {
        cout << "\n❌ 校验失败：两个算法的结果不一致！" << endl;
    }
    
    cout << "------------------------------------------" << endl;
    
    // 额外的 NAF 统计信息
    cout << "NAF 编码长度 (a) = " << d_naf.size() << " (标准二进制位宽: " << (int)ceil(log2(a+1)) << ")" << endl;
    cout << "NAF 编码长度 (b) = " << f_naf.size() << " (标准二进制位宽: " << (int)ceil(log2(b+1)) << ")" << endl;
    
    cout << "------------------------------------------" << endl;

    // --- 测试 CRT 加速 RSA 算法 (算法 3) ---
    cout << "--- 测试 CRT 加速 RSA 算法 (算法 3) ---" << endl;
    
    // 假设的 RSA 参数 (使用小素数便于验证)
    ull p_rsa = 101; 
    ull q_rsa = 113; 
    ull d = 6597; 
    ull C = 1234; 

    // 预计算值
    ull d1 = d % (p_rsa - 1); // 97
    ull d2 = d % (q_rsa - 1); // 105
    ull q_inv_crt = ModInverse(q_rsa, p_rsa); // 113^{-1} mod 101 = 49
    
    // 运行 CRT 加速解密 (算法 3)
    ull M_crt = CRT_Decryption(C, p_rsa, q_rsa, d1, d2, q_inv_crt);
    cout << "CRT 加速 RSA 解密 (算法 3):" << endl;
    cout << "  密文 C = " << C << endl;
    cout << "  p=" << p_rsa << ", q=" << q_rsa << ", d=" << d << endl;
    cout << "  q^{-1} mod p = " << q_inv_crt << endl;
    cout << "  明文 M = " << M_crt << " (预期结果 41)" << endl;
    
    return 0;
}