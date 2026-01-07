//RSA 小加密指数 e=3 的攻击算法（实验2）

#include <iostream>
#include <string>
#include <cstdint>
#include <windows.h>

using ull = unsigned long long;

// GCD 函数
ull gcd(ull a, ull b) {
    while (b != 0) {
        ull temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// 扩展欧几里得算法求模逆（修复完整逻辑）
long long ext_gcd_mod_inverse(ull a, ull m) {
    ull m0 = m;
    long long y = 0, x = 1;
    if (m == 1) return 0;
    while (a > 1) {
        ull q = a / m;
        ull t = m;
        m = a % m;
        a = t;
        t = y;
        y = x - q * y;
        x = t;
    }
    // 现在 a == gcd
    if (a > 1) return -1;  // 无逆元
    if (x < 0) x += m0;
    return x;
}

// 中国剩余定理（完全修复）
ull chinese_remainder(const ull n[3], const ull c[3]) {
    __int128 prod = (__int128)n[0] * n[1] * n[2];
    __int128 sum = 0;

    for (int i = 0; i < 3; ++i) {
        __int128 M = prod / n[i];
        ull Mi_mod_ni = (ull)(M % (__int128)n[i]);  // 安全取模
        long long inv_ll = ext_gcd_mod_inverse(Mi_mod_ni, n[i]);
        if (inv_ll == -1) {
            std::cerr << "【错误】求逆失败！" << std::endl;
            return 0;
        }
        ull inv = (ull)inv_ll;
        sum += (__int128)c[i] * M * inv;
    }

    ull x = (ull)(sum % prod);
    return x;
}

// 整数立方根
ull integer_cube_root(ull target) {
    ull low = 0;
    ull high = 1000000ULL;
    while (low < high) {
        ull mid = low + (high - low + 1) / 2;
        __int128 cube = (__int128)mid * mid * mid;
        if (cube <= target) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "========================================\n";
    std::cout << " RSA 小加密指数 e=3 的攻击算法（实验2）\n";
    std::cout << "========================================\n";
    std::cout << "用法：work2.exe n1 n2 n3 c1 c2 c3\n";
    std::cout << "示例：work2.exe 763813 828083 720761 352596 408368 6728\n";
    std::cout << "----------------------------------------\n";

    if (argc != 7) {
        std::cerr << "【错误】参数不足！需要 6 个参数。\n";
        return 1;
    }

    ull n[3], c[3];
    try {
        n[0] = std::stoull(argv[1]);
        n[1] = std::stoull(argv[2]);
        n[2] = std::stoull(argv[3]);
        c[0] = std::stoull(argv[4]);
        c[1] = std::stoull(argv[5]);
        c[2] = std::stoull(argv[6]);
    } catch (...) {
        std::cerr << "【错误】参数转换失败！\n";
        return 1;
    }

    std::cout << "【互素测试】\n";
    if (gcd(n[0], n[1]) != 1 || gcd(n[0], n[2]) != 1 || gcd(n[1], n[2]) != 1) {
        std::cerr << "【错误】n1、n2、n3 不两两互素！\n";
        return 1;
    }
    std::cout << "    通过！\n";

    __int128 N_big = (__int128)n[0] * n[1] * n[2];
    ull N = (ull)N_big;
    ull x = chinese_remainder(n, c);

    std::cout << "【CRT 计算】\n";
    std::cout << "    N = " << N << "\n";
    std::cout << "    x = " << x << "\n";

    ull m = integer_cube_root(x);
    std::cout << "【明文恢复】\n";
    std::cout << "    m = " << m << "\n";

    __int128 computed = (__int128)m * m * m;
    if (computed == x) {
        std::cout << "【验证】通过！\n";
    } else {
        std::cout << "【警告】验证失败（" << (ull)computed << " ≠ " << x << "）\n";
    }

    std::cout << "========================================\n";
    std::cout << "攻击完成，明文 m = " << m << "\n";
    std::cout << "========================================\n";

    return 0;
}