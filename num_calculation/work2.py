import math
import matplotlib.pyplot as plt

def linspace(start, end, num):
    """生成等间距的点"""
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]

def legendre_polynomial(k, x):
    """计算勒让德多项式 P_k(x)"""
    if k == 0:
        return 1
    elif k == 1:
        return x
    else:
        return ((2 * k - 1) * x * legendre_polynomial(k - 1, x) - (k - 1) * legendre_polynomial(k - 2, x)) / k

def trapezoidal_integral(func, a, b, num_points=1000):
    """使用梯形法进行数值积分"""
    x_points = linspace(a, b, num_points)
    y_points = [func(x) for x in x_points]
    h = (b - a) / (num_points - 1)
    integral = 0.5 * h * (y_points[0] + y_points[-1]) + h * sum(y_points[1:-1])
    return integral

def best_square_approximation(a, b, c, k, n_points):
    """勒让德正交多项式的最佳平方逼近"""
    x_points = linspace(a, b, n_points)
    y_points = [1 / (1 + c * x**2) for x in x_points]
    x_mapped = [2 * (x - a) / (b - a) - 1 for x in x_points]

    coeffs = []
    for i in range(k + 1):
        def L_i(x):
            t = 2 * (x - a) / (b - a) - 1
            return legendre_polynomial(i, t)

        numerator = trapezoidal_integral(lambda x: (1 / (1 + c * x**2)) * L_i(x), a, b)
        denominator = trapezoidal_integral(lambda x: L_i(x)**2, a, b)
        coeffs.append(numerator / denominator)

    def approximant(x):
        if isinstance(x, (int, float)):
            x_mapped = 2 * (x - a) / (b - a) - 1
            result = sum(coeffs[i] * legendre_polynomial(i, x_mapped) for i in range(k + 1))
            return result
        else:
            results = []
            for xi in x:
                x_mapped = 2 * (xi - a) / (b - a) - 1
                result = sum(coeffs[i] * legendre_polynomial(i, x_mapped) for i in range(k + 1))
                results.append(result)
            return results

    return approximant

def least_squares_fit(a, b, c, k, n_points):
    """最小二乘拟合"""
    x_points = linspace(a, b, n_points)
    y_points = [1 / (1 + c * x**2) for x in x_points]

    V = [[x**i for i in range(k + 1)] for x in x_points]
    VT = [[V[j][i] for j in range(len(V))] for i in range(len(V[0]))]

    VT_V = [[sum(VT[i][l] * V[l][j] for l in range(len(VT[0]))) for j in range(len(V[0]))] for i in range(len(VT))]
    VT_y = [sum(VT[i][j] * y_points[j] for j in range(len(VT[0]))) for i in range(len(VT))]

    n = len(VT_V)
    aug = [VT_V[i] + [VT_y[i]] for i in range(n)]

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]

        for k in range(i + 1, n):
            if aug[i][i] != 0:
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]

    coeffs = [0] * n
    for i in range(n - 1, -1, -1):
        coeffs[i] = aug[i][n]
        for j in range(i + 1, n):
            coeffs[i] -= aug[i][j] * coeffs[j]
        if aug[i][i] != 0:
            coeffs[i] /= aug[i][i]

    def approximant(x):
        if isinstance(x, (int, float)):
            result = sum(coeffs[i] * (x**i) for i in range(k + 1))
            return result
        else:
            results = []
            for xi in x:
                result = sum(coeffs[i] * (xi**i) for i in range(k + 1))
                results.append(result)
            return results

    return approximant

def plot_comparison(x_test, y_test_true, y_best_square, y_least_squares):
    """绘制函数对比图"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, label="目标函数", color="black")
    plt.plot(x_test, y_best_square, label="最佳平方逼近", linestyle="--")
    plt.plot(x_test, y_least_squares, label="最小二乘拟合", linestyle=":")
    plt.xlabel("x 值")
    plt.ylabel("函数值")
    plt.title("函数对比图")
    plt.legend()
    plt.grid()
    plt.show()

def plot_error_comparison(x_test, y_test_true, y_best_square, y_least_squares):
    """绘制误差对比图"""
    error_best_square = [abs(y_best_square[i] - y_test_true[i]) for i in range(len(y_test_true))]
    error_least_squares = [abs(y_least_squares[i] - y_test_true[i]) for i in range(len(y_test_true))]

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, error_best_square, label="最佳平方逼近误差", marker='o')
    plt.plot(x_test, error_least_squares, label="最小二乘拟合误差", marker='x')
    plt.xlabel("x 值")
    plt.ylabel("误差")
    plt.title("误差对比图")
    plt.legend()
    plt.grid()
    plt.show()

def compare_approximations(a, b, c, k, n_points, m_points):
    x_points = linspace(a, b, n_points)
    y_points = [1 / (1 + c * x**2) for x in x_points]
    x_test = linspace(a, b, m_points)
    y_test_true = [1 / (1 + c * x**2) for x in x_test]

    best_square = best_square_approximation(a, b, c, k, n_points)
    least_squares = least_squares_fit(a, b, c, k, n_points)

    y_best_square = best_square(x_test)
    y_least_squares = least_squares(x_test)

    error_best_square = sum(abs(y_best_square[i] - y_test_true[i]) for i in range(len(y_test_true))) / len(y_test_true)
    error_least_squares = sum(abs(y_least_squares[i] - y_test_true[i]) for i in range(len(y_test_true))) / len(y_test_true)

    print("x值\t\t真实值\t\t最佳平方逼近\t最小二乘拟合")
    print("-" * 60)
    for i in range(0, len(x_test), max(1, len(x_test)//10)):
        print(f"{x_test[i]:.3f}\t\t{y_test_true[i]:.6f}\t{y_best_square[i]:.6f}\t\t{y_least_squares[i]:.6f}")

    print(f"\n最佳平方逼近的平均误差: {error_best_square:.7f}")
    print(f"最小二乘拟合的平均误差: {error_least_squares:.7f}")

    # 绘制对比图
    plot_comparison(x_test, y_test_true, y_best_square, y_least_squares)
    plot_error_comparison(x_test, y_test_true, y_best_square, y_least_squares)

def main():
    print("最佳平方逼近与最小二乘拟合对比实验")
    print("目标函数: f(x) = 1 / (1 + c*x^2)")

    a = float(input("请输入区间左端点 a: "))
    b = float(input("请输入区间右端点 b: "))
    c = float(input("请输入目标函数参数 c: "))
    k = int(input("请输入逼近多项式次数 k (建议1, 2, 3): "))
    n = int(input("请输入采样点数 n+1 (建议11): "))
    m = int(input("请输入实验点数 m (建议100): "))

    print("\n开始计算...")
    compare_approximations(a, b, c, k, n, m)
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()