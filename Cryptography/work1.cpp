//RSA 加解密实验程序（实验1

#include <iostream>
#include <cmath>
#include <cstdint> // 用于 uint64_t 等类型
#include <string>  // 用于 std::stoull

// 使用 unsigned long long 处理大数计算
using ull = unsigned long long;

// 判断一个数是否为素数（简单试除法，针对实验给定的数据规模足够）
bool is_prime(ull n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (ull i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// 最大公约数（GCD）
ull gcd(ull a, ull b) {
    while (b != 0) {
        ull temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// 扩展欧几里得算法求模逆
ull mod_inverse(ull a, ull m) {
    ull m0 = m;
    ull y = 0, x = 1;
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
    if (x < 0) x += m0;
    return x;
}

// 模幂运算 a^b mod m，使用 __int128 防止溢出
ull mod_pow(ull a, ull b, ull m) {
    ull res = 1;
    a %= m;
    while (b > 0) {
        if (b & 1) {
            __int128 temp = (__int128)res * a % m;
            res = (ull)temp;
        }
        __int128 temp = (__int128)a * a % m;
        a = (ull)temp;
        b >>= 1;
    }
    return res;
}

// RSA 加密函数
void rsa_encrypt(ull p, ull q, ull e, ull m, ull& c, ull& d) {
    if (!is_prime(p) || !is_prime(q)) {
        std::cerr << "【错误】p 或 q 不是素数，请检查输入！" << std::endl;
        c = d = 0;
        return;
    }
    ull n = p * q;
    ull phi = (p - 1) * (q - 1);
    if (gcd(e, phi) != 1) {
        std::cerr << "【错误】e 与 φ(n) 不互素（gcd(e, φ(n)) ≠ 1），无法生成有效密钥！" << std::endl;
        c = d = 0;
        return;
    }
    d = mod_inverse(e, phi);
    c = mod_pow(m, e, n);

    // 输出中间结果帮助理解
    std::cout << "【RSA 参数计算】" << std::endl;
    std::cout << "    n  = p × q     = " << n << std::endl;
    std::cout << "    φ(n) = (p-1)(q-1) = " << phi << std::endl;
    std::cout << "    公钥：(n, e) = (" << n << ", " << e << ")" << std::endl;
    std::cout << "    私钥：d = " << d << std::endl;
    std::cout << "【加密结果】" << std::endl;
    std::cout << "    密文 c = m^e mod n = " << c << std::endl;
    std::cout << "    私钥 d = " << d << std::endl;
}

// RSA 解密函数
void rsa_decrypt(ull p, ull q, ull d, ull c, ull& m, ull& e) {
    if (!is_prime(p) || !is_prime(q)) {
        std::cerr << "【错误】p 或 q 不是素数，请检查输入！" << std::endl;
        m = e = 0;
        return;
    }
    ull n = p * q;
    ull phi = (p - 1) * (q - 1);
    if (gcd(d, phi) != 1) {
        std::cerr << "【错误】d 与 φ(n) 不互素，无法进行解密！" << std::endl;
        m = e = 0;
        return;
    }
    e = mod_inverse(d, phi);
    m = mod_pow(c, d, n);

    std::cout << "【RSA 参数计算】" << std::endl;
    std::cout << "    n  = p × q     = " << n << std::endl;
    std::cout << "    φ(n) = (p-1)(q-1) = " << phi << std::endl;
    std::cout << "    私钥：d = " << d << std::endl;
    std::cout << "    公钥：(n, e) = (" << n << ", " << e << ")" << std::endl;
    std::cout << "【解密结果】" << std::endl;
    std::cout << "    明文 m = c^d mod n = " << m << std::endl;
    std::cout << "    公钥指数 e = " << e << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "     RSA 加解密实验程序（实验1）" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "用法：" << std::endl;
    std::cout << "  加密： " << argv[0] << " encrypt p q e m" << std::endl;
    std::cout << "  解密： " << argv[0] << " decrypt p q d c" << std::endl;
    std::cout << "示例：" << std::endl;
    std::cout << "  加密： " << argv[0] << " encrypt 2357 2551 3674911 5234673" << std::endl;
    std::cout << "  解密： " << argv[0] << " decrypt 885320963 238855417 116402471153538991 113535859035722866" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    if (argc < 2) {
        std::cerr << "【错误】参数不足！请参考上方用法。" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "encrypt") {
        if (argc != 6) {
            std::cerr << "【错误】加密模式需要 4 个参数：p q e m" << std::endl;
            return 1;
        }
        ull p = std::stoull(argv[2]);
        ull q = std::stoull(argv[3]);
        ull e = std::stoull(argv[4]);
        ull m = std::stoull(argv[5]);
        ull c, d;
        std::cout << "【正在执行加密】 m = " << m << " → c" << std::endl;
        rsa_encrypt(p, q, e, m, c, d);
    } 
    else if (mode == "decrypt") {
        if (argc != 6) {
            std::cerr << "【错误】解密模式需要 4 个参数：p q d c" << std::endl;
            return 1;
        }
        ull p = std::stoull(argv[2]);
        ull q = std::stoull(argv[3]);
        ull d = std::stoull(argv[4]);
        ull c = std::stoull(argv[5]);
        ull m, e_val;
        std::cout << "【正在执行解密】 c = " << c << " → m" << std::endl;
        rsa_decrypt(p, q, d, c, m, e_val);
    } 
    else {
        std::cerr << "【错误】无效模式 '" << mode << "'，请使用 'encrypt' 或 'decrypt'。" << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "程序执行结束。" << std::endl;
    return 0;
}