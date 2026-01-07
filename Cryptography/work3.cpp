#include <iostream>
#include <random>
#include <cstdint>
#include <windows.h>

using ull = unsigned long long;

// 手动计算一个 unsigned long long 的比特长度（最高位1的位置+1）
unsigned int bit_length(ull n) {
    if (n == 0) return 0;
    unsigned int len = 1;  // 从1开始，因为 n >=1 时至少1位
    while (n >>= 1) ++len;
    return len;
}

// 简单试除法判断素数（适合 l <= 32）
bool is_prime(ull n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (ull i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// 生成指定比特长度范围内的随机素数
ull generate_random_prime(int min_bits, int max_bits, std::mt19937_64& gen) {
    while (true) {
        int bits = min_bits + (gen() % (max_bits - min_bits + 1));
        ull num = (1ULL << (bits - 1)) | (gen() % (1ULL << (bits - 1)));
        if (num % 2 == 0) num += 1;  // 确保奇数
        if (num < 3) continue;
        if (is_prime(num)) return num;
    }
}

// 生成强素数
void generate_strong_prime(int l) {
    if (l <= 0 || l > 32) {
        std::cerr << "【错误】比特长度 l 必须在 1 到 32 之间！" << std::endl;
        return;
    }

    int half = l / 2;
    int low = std::max(4, half - 4);
    int high = half + 4;

    std::random_device rd;
    std::mt19937_64 gen(999);

    ull p = 0, s = 0, r = 0, t = 0;

    std::cout << "【正在生成比特长度为 " << l << " 的强素数】请稍等..." << std::endl;

    bool found = false;
    while (!found) {
        // 生成 s 和 t
        s = generate_random_prime(low, high, gen);
        t = generate_random_prime(low, high, gen);

        // 生成 r = 2*t*i + 1 是素数
        ull i = 1;
        while (true) {
            __int128 temp_r = (__int128)2 * t * i + 1;
            if (temp_r >= (1ULL << 40)) break;
            r = (ull)temp_r;
            if (is_prime(r)) break;
            ++i;
            if (i > 1000000) break;
        }
        if (i > 1000000) continue;

        // 生成 p = 2*s*r*j + 1 是素数，且比特长度正好为 l
        ull j = 1;
        while (true) {
            __int128 temp_p = (__int128)2 * s * r * j + 1;
            if (temp_p >= (__int128)1 << (l + 1)) break;  // 超过 l+1 位
            p = (ull)temp_p;
            if (bit_length(p) == static_cast<unsigned int>(l) && is_prime(p)) {
                found = true;
                break;
            }
            ++j;
            if (j > 1000000) break;
        }
    }

    std::cout << "【生成成功】" << std::endl;
    std::cout << "强素数 p = " << p << " (比特长度: " << bit_length(p) << ")" << std::endl;
    std::cout << "p+1 的大素因子 s = " << s << std::endl;
    std::cout << "p-1 的大素因子 r = " << r << std::endl;
    std::cout << "r-1 的大素因子 t = " << t << std::endl;

    std::cout << "【验证】" << std::endl;
    std::cout << "    p 是素数: " << (is_prime(p) ? "是" : "否") << std::endl;
    std::cout << "    (p-1) 可被 2*r 整除: " << (((p-1) % (2*r)) == 0 ? "是" : "否") << std::endl;
    std::cout << "    (p+1) 可被 2*s 整除: " << (((p+1) % (2*s)) == 0 ? "是" : "否") << std::endl;
    std::cout << "    (r-1) 可被 2*t 整除: " << (((r-1) % (2*t)) == 0 ? "是" : "否") << std::endl;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "========================================\n";
    std::cout << "      生成强素数算法（Strong Prime）\n";
    std::cout << "========================================\n";

    int l;
    std::cout << "请输入强素数的比特长度 l (1 <= 长度 <= 32): ";
    std::cin >> l;

    generate_strong_prime(l);

    return 0;
}