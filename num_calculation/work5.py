"""
高斯-塞德尔迭代和SOR迭代的通用程序
实现要求：
1. 输入：矩阵A、向量b、迭代初值x⁰、迭代最大步数K、误差控制ε
2. 对于SOR迭代，还需输入松弛因子ω
3. 输出：迭代步数及方程Ax=b的根值x*
"""

from typing import List, Tuple

# 类型定义
Matrix = List[List[float]]
Vector = List[float]


def copy_vector(v: Vector) -> Vector:
    """复制向量"""
    return [x for x in v]


def vector_norm(v: Vector) -> float:
    """计算向量的无穷范数（最大绝对值）"""
    return max(abs(x) for x in v)


def matrix_vector_multiply(A: Matrix, x: Vector) -> Vector:
    """计算矩阵A与向量x的乘积"""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        result[i] = sum(A[i][j] * x[j] for j in range(n))
    return result


def vector_subtract(v1: Vector, v2: Vector) -> Vector:
    """计算向量v1 - v2"""
    return [v1[i] - v2[i] for i in range(len(v1))]


def gauss_seidel_iteration(A: Matrix, b: Vector, x0: Vector, K: int, epsilon: float) -> Tuple[int, Vector]:
    """
    高斯-塞德尔迭代法求解Ax=b
    
    参数:
        A: 系数矩阵
        b: 右端向量
        x0: 迭代初值
        K: 最大迭代步数
        epsilon: 误差控制
    
    返回:
        (迭代步数, 解向量x*)
    """
    n = len(A)
    x = copy_vector(x0)
    x_new = copy_vector(x0)
    
    for k in range(1, K + 1):
        # 高斯-塞德尔迭代公式
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))  # 使用新值
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # 使用旧值
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # 计算误差（使用无穷范数）
        error = vector_norm(vector_subtract(x_new, x))
        
        # 更新x
        x = copy_vector(x_new)
        
        # 检查收敛
        if error < epsilon:
            return k, x
    
    return K, x


def sor_iteration(A: Matrix, b: Vector, x0: Vector, K: int, epsilon: float, omega: float) -> Tuple[int, Vector]:
    """
    超松弛迭代法（SOR）求解Ax=b
    
    参数:
        A: 系数矩阵
        b: 右端向量
        x0: 迭代初值
        K: 最大迭代步数
        epsilon: 误差控制
        omega: 松弛因子
    
    返回:
        (迭代步数, 解向量x*)
    """
    n = len(A)
    x = copy_vector(x0)
    x_new = copy_vector(x0)
    
    for k in range(1, K + 1):
        # SOR迭代公式
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))  # 使用新值
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # 使用旧值
            x_gs = (b[i] - sum1 - sum2) / A[i][i]  # 高斯-塞德尔迭代值
            x_new[i] = (1 - omega) * x[i] + omega * x_gs  # SOR公式
        
        # 计算误差（使用无穷范数）
        error = vector_norm(vector_subtract(x_new, x))
        
        # 更新x
        x = copy_vector(x_new)
        
        # 检查收敛
        if error < epsilon:
            return k, x
    
    return K, x


def format_vector(v: Vector, precision: int = 8, width: int = 12) -> str:
    """格式化向量输出"""
    fmt = f"{{:>{width}.{precision}f}}"
    return "[" + " ".join(fmt.format(val) for val in v) + "]"


def test_case_1():
    """测试用例1：使用给定的矩阵A和向量b进行测试"""
    # 矩阵A
    A = [
        [31, -13, 0, 0, 0, -10, 0, 0, 0],
        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
        [0, -9, 31, -10, 0, 0, 0, 0, 0],
        [0, 0, -10, 79, -30, 0, 0, 0, -9],
        [0, 0, 0, -30, 57, -7, 0, -5, 0],
        [0, 0, 0, 0, -7, 47, -30, 0, 0],
        [0, 0, 0, 0, 0, -30, 41, 0, 0],
        [0, 0, 0, 0, -5, 0, 0, 27, -2],
        [0, 0, 0, -9, 0, 0, 0, -2, 29],
    ]
    
    # 向量b
    b = [-15, 27, -23, 0, -20, 12, -7, 7, 10]
    
    # 迭代初值x⁰ = 0
    n = len(A)
    x0 = [0.0] * n
    
    # 参数设置
    K = 10000  # 最大迭代步数
    epsilon = 1e-8  # 误差控制
    
    print("=" * 70)
    print("测试用例1：使用给定的矩阵A和向量b")
    print("=" * 70)
    print(f"迭代初值 x⁰ = {format_vector(x0)}")
    print(f"误差控制 ε = {epsilon}")
    print()
    
    # 高斯-塞德尔迭代
    print("1. 高斯-塞德尔迭代法：")
    iterations_gs, x_gs = gauss_seidel_iteration(A, b, x0, K, epsilon)
    print(f"   迭代步数: {iterations_gs}")
    print(f"   方程 Ax = b 的根值 x*:")
    print(f"   {format_vector(x_gs)}")
    print()
    
    # SOR迭代（使用默认松弛因子ω=1.0，即高斯-塞德尔迭代）
    print("2. 超松弛迭代法（SOR）：")
    omega = 1.0
    iterations_sor, x_sor = sor_iteration(A, b, x0, K, epsilon, omega)
    print(f"   松弛因子 ω = {omega}")
    print(f"   迭代步数: {iterations_sor}")
    print(f"   方程 Ax = b 的根值 x*:")
    print(f"   {format_vector(x_sor)}")
    print()
    
    # 验证解的正确性
    print("3. 验证解的正确性：")
    residual_gs = vector_subtract(matrix_vector_multiply(A, x_gs), b)
    residual_sor = vector_subtract(matrix_vector_multiply(A, x_sor), b)
    print(f"   高斯-塞德尔迭代的残差范数: {vector_norm(residual_gs):.2e}")
    print(f"   SOR迭代的残差范数: {vector_norm(residual_sor):.2e}")
    print()


def test_case_2():
    """测试用例2：测试不同松弛因子ω，找出最佳值"""
    # 矩阵A
    A = [
        [31, -13, 0, 0, 0, -10, 0, 0, 0],
        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
        [0, -9, 31, -10, 0, 0, 0, 0, 0],
        [0, 0, -10, 79, -30, 0, 0, 0, -9],
        [0, 0, 0, -30, 57, -7, 0, -5, 0],
        [0, 0, 0, 0, -7, 47, -30, 0, 0],
        [0, 0, 0, 0, 0, -30, 41, 0, 0],
        [0, 0, 0, 0, -5, 0, 0, 27, -2],
        [0, 0, 0, -9, 0, 0, 0, -2, 29],
    ]
    
    # 向量b
    b = [-15, 27, -23, 0, -20, 12, -7, 7, 10]
    
    # 迭代初值x⁰ = 0
    n = len(A)
    x0 = [0.0] * n
    
    # 参数设置
    K = 10000  # 最大迭代步数
    epsilon = 1e-8  # 误差控制
    
    print("=" * 70)
    print("测试用例2：测试不同松弛因子ω，找出最佳值")
    print("=" * 70)
    print(f"松弛因子范围: ω = i/50, i = 1, 2, ..., 99")
    print(f"误差控制 ε = {epsilon}")
    print()
    
    # 存储每个ω对应的迭代步数
    omega_iterations = []
    best_omega = 1.0
    best_iterations = K + 1
    
    # 测试ω = i/50, i = 1, 2, ..., 99
    print("正在测试不同松弛因子...")
    for i in range(1, 100):
        omega = i / 50.0
        iterations, _ = sor_iteration(A, b, x0, K, epsilon, omega)
        omega_iterations.append((omega, iterations))
        
        # 更新最佳值
        if iterations < best_iterations:
            best_iterations = iterations
            best_omega = omega
    
    # 打印结果
    print("\n松弛因子ω与迭代步数对应表（部分显示）：")
    print("-" * 70)
    print(f"{'ω':>10} | {'迭代步数':>10}")
    print("-" * 70)
    
    # 显示前10个、中间10个和后10个
    for idx in range(10):
        omega, iters = omega_iterations[idx]
        print(f"{omega:>10.2f} | {iters:>10}")
    
    print("   ...")
    
    for idx in range(44, 54):
        omega, iters = omega_iterations[idx]
        print(f"{omega:>10.2f} | {iters:>10}")
    
    print("   ...")
    
    for idx in range(89, 99):
        omega, iters = omega_iterations[idx]
        print(f"{omega:>10.2f} | {iters:>10}")
    
    print("-" * 70)
    print(f"\n最佳松弛因子: ω = {best_omega:.2f}")
    print(f"最佳松弛因子对应的迭代步数: {best_iterations}")
    print()
    
    # 使用最佳松弛因子重新计算并显示解
    print(f"使用最佳松弛因子 ω = {best_omega:.2f} 计算得到的解:")
    _, x_best = sor_iteration(A, b, x0, K, epsilon, best_omega)
    print(f"{format_vector(x_best)}")
    print()


def main():
    """主函数"""
    # 测试用例1
    test_case_1()
    
    # 测试用例2
    test_case_2()


if __name__ == "__main__":
    main()

