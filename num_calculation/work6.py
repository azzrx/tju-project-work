"""
第7章 非线性方程（组）的数值解法
实现要求：
1. 不动点迭代
2. Steffensen加速方法
3. Newton迭代

所有代码独立完成，不使用任何数值计算算法库
"""

import math
from typing import Callable, List, Tuple


def fixed_point_iteration(phi: Callable[[float], float], x0: float, 
                          max_iter: int, epsilon: float) -> Tuple[int, List[float], float]:
    """
    不动点迭代法
    
    参数:
        phi: 不动点迭代函数 x = phi(x)
        x0: 初始值
        max_iter: 最大迭代次数
        epsilon: 精度要求
    
    返回:
        (迭代次数, 迭代序列, 最终结果)
    """
    x = x0
    iterations = [x0]
    
    for k in range(1, max_iter + 1):
        x_new = phi(x)
        iterations.append(x_new)
        
        # 检查收敛条件 |x_k - x_{k-1}| < epsilon
        if abs(x_new - x) < epsilon:
            return k, iterations, x_new
        
        x = x_new
    
    return max_iter, iterations, x


def steffensen_acceleration(phi: Callable[[float], float], x0: float,
                           max_iter: int, epsilon: float) -> Tuple[int, List[float], float]:
    """
    Steffensen加速方法
    
    参数:
        phi: 不动点迭代函数 x = phi(x)
        x0: 初始值
        max_iter: 最大迭代次数
        epsilon: 精度要求
    
    返回:
        (迭代次数, 迭代序列, 最终结果)
    """
    x = x0
    iterations = [x0]
    
    for k in range(1, max_iter + 1):
        # Steffensen加速公式
        y = phi(x)
        z = phi(y)
        
        # 避免分母为零
        denominator = z - 2 * y + x
        if abs(denominator) < 1e-15:
            # 如果分母太小，使用普通不动点迭代
            x_new = phi(x)
        else:
            # Steffensen加速公式: x_{k+1} = x_k - (y - x_k)^2 / (z - 2y + x_k)
            x_new = x - (y - x) ** 2 / denominator
        
        iterations.append(x_new)
        
        # 检查收敛条件 |x_k - x_{k-1}| < epsilon
        if abs(x_new - x) < epsilon:
            return k, iterations, x_new
        
        x = x_new
    
    return max_iter, iterations, x


def newton_iteration(f: Callable[[float], float], df: Callable[[float], float],
                     x0: float, max_iter: int, epsilon: float) -> Tuple[int, List[float], float]:
    """
    Newton迭代法
    
    参数:
        f: 函数 f(x)
        df: 函数的导数 f'(x)
        x0: 初始值
        max_iter: 最大迭代次数
        epsilon: 精度要求
    
    返回:
        (迭代次数, 迭代序列, 最终结果)
    """
    x = x0
    iterations = [x0]
    
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        
        # 避免除零
        if abs(dfx) < 1e-15:
            raise ValueError(f"导数为零，Newton迭代无法继续 (x = {x})")
        
        # Newton迭代公式: x_{k+1} = x_k - f(x_k) / f'(x_k)
        x_new = x - fx / dfx
        iterations.append(x_new)
        
        # 检查收敛条件 |x_k - x_{k-1}| < epsilon
        if abs(x_new - x) < epsilon:
            return k, iterations, x_new
        
        x = x_new
    
    return max_iter, iterations, x


# ==================== 函数定义 ====================

# 函数1: f(x) = x² - 3x + 2 - e^x
def f1(x: float) -> float:
    """函数 f(x) = x² - 3x + 2 - e^x"""
    return x * x - 3 * x + 2 - math.exp(x)


def df1(x: float) -> float:
    """函数 f(x) 的导数 f'(x) = 2x - 3 - e^x"""
    return 2 * x - 3 - math.exp(x)


def phi1(x: float) -> float:
    """
    不动点迭代格式: x = (x² + 2 - e^x) / 3
    将 f(x) = x² - 3x + 2 - e^x = 0 改写为 x = (x² + 2 - e^x) / 3
    """
    return (x * x + 2 - math.exp(x)) / 3.0


# 函数2: g(x) = x³ + 2x² + 10x - 20
def g1(x: float) -> float:
    """函数 g(x) = x³ + 2x² + 10x - 20"""
    return x * x * x + 2 * x * x + 10 * x - 20


def dg1(x: float) -> float:
    """函数 g(x) 的导数 g'(x) = 3x² + 4x + 10"""
    return 3 * x * x + 4 * x + 10


def phi2(x: float) -> float:
    """
    不动点迭代格式: x = 20 / (x² + 2x + 10)
    将 g(x) = x³ + 2x² + 10x - 20 = 0 改写为 x = 20 / (x² + 2x + 10)
    这个格式在根附近更稳定
    """
    denominator = x * x + 2 * x + 10
    if abs(denominator) < 1e-15:
        raise ValueError("分母接近零")
    return 20.0 / denominator


# ==================== 测试函数 ====================

def test_function1():
    """测试函数 f(x) = x² - 3x + 2 - e^x"""
    print("=" * 80)
    print("函数1: f(x) = x² - 3x + 2 - e^x")
    print("=" * 80)
    
    epsilon = 1e-8
    max_iter = 1000
    
    # 选择合适的初始值（通过观察函数性质）
    # f(0) = 2 - 1 = 1 > 0, f(1) = 1 - 3 + 2 - e = -e < 0
    # 所以根在 [0, 1] 之间
    x0 = 0.5
    
    print(f"\n初始迭代值: x0 = {x0}")
    print(f"精度要求: ε = {epsilon}")
    print()
    
    # 1. 不动点迭代
    print("-" * 80)
    print("1. 不动点迭代法 (x = (x² + 2 - e^x) / 3):")
    print("-" * 80)
    try:
        iter_count, iterations, result = fixed_point_iteration(phi1, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        f_value = f1(result)
        print(f"验证: f(x*) = {f_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, f(x_{i}) = {f1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
    except Exception as e:
        print(f"不动点迭代失败: {e}")
    
    print()
    
    # 2. Steffensen加速
    print("-" * 80)
    print("2. Steffensen加速方法:")
    print("-" * 80)
    try:
        iter_count, iterations, result = steffensen_acceleration(phi1, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        f_value = f1(result)
        print(f"验证: f(x*) = {f_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, f(x_{i}) = {f1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
    except Exception as e:
        print(f"Steffensen加速失败: {e}")
    
    print()
    
    # 3. Newton迭代
    print("-" * 80)
    print("3. Newton迭代法:")
    print("-" * 80)
    try:
        iter_count, iterations, result = newton_iteration(f1, df1, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        f_value = f1(result)
        print(f"验证: f(x*) = {f_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, f(x_{i}) = {f1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, f(x_{i}) = {f1(iterations[i]):.15f}")
    except Exception as e:
        print(f"Newton迭代失败: {e}")
    
    print()


def test_function2():
    """测试函数 g(x) = x³ + 2x² + 10x - 20"""
    print("=" * 80)
    print("函数2: g(x) = x³ + 2x² + 10x - 20")
    print("=" * 80)
    
    epsilon = 1e-8
    max_iter = 1000
    
    # 选择合适的初始值
    # g(0) = -20 < 0, g(2) = 8 + 8 + 20 - 20 = 16 > 0
    # 所以根在 [0, 2] 之间
    x0 = 1.0
    
    print(f"\n初始迭代值: x0 = {x0}")
    print(f"精度要求: ε = {epsilon}")
    print()
    
    # 1. 不动点迭代
    print("-" * 80)
    print("1. 不动点迭代法 (x = 20 / (x² + 2x + 10)):")
    print("-" * 80)
    try:
        iter_count, iterations, result = fixed_point_iteration(phi2, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        g_value = g1(result)
        print(f"验证: g(x*) = {g_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, g(x_{i}) = {g1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
    except Exception as e:
        print(f"不动点迭代失败: {e}")
    
    print()
    
    # 2. Steffensen加速
    print("-" * 80)
    print("2. Steffensen加速方法:")
    print("-" * 80)
    try:
        iter_count, iterations, result = steffensen_acceleration(phi2, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        g_value = g1(result)
        print(f"验证: g(x*) = {g_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, g(x_{i}) = {g1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
    except Exception as e:
        print(f"Steffensen加速失败: {e}")
    
    print()
    
    # 3. Newton迭代
    print("-" * 80)
    print("3. Newton迭代法:")
    print("-" * 80)
    try:
        iter_count, iterations, result = newton_iteration(g1, dg1, x0, max_iter, epsilon)
        print(f"迭代次数: {iter_count}")
        print(f"最终结果: x* = {result:.10f}")
        g_value = g1(result)
        print(f"验证: g(x*) = {g_value:.15f}")
        print("\n迭代过程（前10次和后10次）:")
        if len(iterations) <= 20:
            for i, x in enumerate(iterations):
                print(f"  k={i:3d}: x_{i} = {x:.10f}, g(x_{i}) = {g1(x):.15f}")
        else:
            for i in range(10):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
            print("  ...")
            for i in range(len(iterations) - 10, len(iterations)):
                print(f"  k={i:3d}: x_{i} = {iterations[i]:.10f}, g(x_{i}) = {g1(iterations[i]):.15f}")
    except Exception as e:
        print(f"Newton迭代失败: {e}")
    
    print()


def bisection_method(f: Callable[[float], float], a: float, b: float,
                     epsilon: float) -> Tuple[int, int, List[float], float]:
    """
    二分法求根
    
    参数:
        f: 函数 f(x)
        a: 区间左端点
        b: 区间右端点
        epsilon: 精度要求
    
    返回:
        (迭代次数, 函数计算次数, 迭代序列, 最终结果)
    """
    fa = f(a)
    fb = f(b)
    function_count = 2  # 初始计算 f(a) 和 f(b)
    
    if fa * fb > 0:
        raise ValueError(f"区间 [{a}, {b}] 内可能没有根（端点函数值同号）")
    
    iterations = []
    iter_count = 0
    
    while (b - a) / 2 > epsilon:
        c = (a + b) / 2
        fc = f(c)
        function_count += 1
        iterations.append(c)
        iter_count += 1
        
        if abs(fc) < 1e-15:  # 精确根
            return iter_count, function_count, iterations, c
        
        if fa * fc < 0:
            b = c
            fb = fc  # 重用函数值
        else:
            a = c
            fa = fc  # 重用函数值
    
    # 最终结果取区间中点
    result = (a + b) / 2
    iterations.append(result)
    return iter_count, function_count, iterations, result


# ==================== 问题3的函数定义 ====================

def f_problem3(x: float) -> float:
    """问题3的函数: f(x) = e^x + 10x - 2"""
    return math.exp(x) + 10 * x - 2


def phi_problem3(x: float) -> float:
    """问题3的迭代函数: x_{k+1} = (2 - e^{x_k}) / 10"""
    return (2 - math.exp(x)) / 10.0


def compare_problem3():
    """
    问题3: 比较求 e^x + 10x - 2 = 0 的根到三位小数所需的计算量
    (1) 在区间[0,1]内用二分法
    (2) 用迭代法 x_{k+1} = (2 - e^{x_k})/10, 取初值 x_0 = 0
    """
    print("=" * 80)
    print("问题3: 比较求 e^x + 10x - 2 = 0 的根到三位小数所需的计算量")
    print("=" * 80)
    
    # 三位小数精度，需要误差 < 0.0005
    epsilon = 0.0005
    
    print(f"\n精度要求: 三位小数 (ε = {epsilon})")
    print(f"函数: f(x) = e^x + 10x - 2")
    print()
    
    # 验证区间[0,1]内有根
    f0 = f_problem3(0)
    f1 = f_problem3(1)
    print(f"验证区间[0,1]: f(0) = {f0:.6f}, f(1) = {f1:.6f}")
    if f0 * f1 > 0:
        print("警告: 区间端点函数值同号，可能没有根")
    else:
        print("区间[0,1]内有根")
    print()
    
    # 方法1: 二分法
    print("-" * 80)
    print("方法1: 二分法 (在区间[0,1]内)")
    print("-" * 80)
    try:
        iter_count_bisection, function_evaluations_bisection, iterations_bisection, result_bisection = bisection_method(
            f_problem3, 0.0, 1.0, epsilon
        )
        
        print(f"迭代次数: {iter_count_bisection}")
        print(f"函数值计算次数: {function_evaluations_bisection}")
        print(f"最终结果: x* = {result_bisection:.6f}")
        print(f"验证: f(x*) = {f_problem3(result_bisection):.10e}")
        print(f"\n迭代过程:")
        for i, x in enumerate(iterations_bisection):
            print(f"  k={i+1:3d}: x = {x:.6f}, f(x) = {f_problem3(x):.10e}")
    except Exception as e:
        print(f"二分法失败: {e}")
        iter_count_bisection = 0
        function_evaluations_bisection = 0
        result_bisection = None
    
    print()
    
    # 方法2: 迭代法
    print("-" * 80)
    print("方法2: 迭代法 x_{k+1} = (2 - e^{x_k})/10, 初值 x_0 = 0")
    print("-" * 80)
    try:
        x0 = 0.0
        iter_count_iteration, iterations_iteration, result_iteration = fixed_point_iteration(
            phi_problem3, x0, 1000, epsilon
        )
        
        # 迭代法每次迭代需要1次函数值计算（计算e^x）
        function_evaluations_iteration = iter_count_iteration
        
        print(f"迭代次数: {iter_count_iteration}")
        print(f"函数值计算次数: {function_evaluations_iteration}")
        print(f"最终结果: x* = {result_iteration:.6f}")
        print(f"验证: f(x*) = {f_problem3(result_iteration):.10e}")
        print(f"\n迭代过程:")
        for i, x in enumerate(iterations_iteration):
            print(f"  k={i:3d}: x_{i} = {x:.6f}, f(x_{i}) = {f_problem3(x):.10e}")
    except Exception as e:
        print(f"迭代法失败: {e}")
        iter_count_iteration = 0
        function_evaluations_iteration = 0
        result_iteration = None
    
    print()
    
    # 比较结果
    print("=" * 80)
    print("计算量比较总结")
    print("=" * 80)
    if result_bisection is not None and result_iteration is not None:
        print(f"\n二分法:")
        print(f"  - 迭代次数: {iter_count_bisection}")
        print(f"  - 函数值计算次数: {function_evaluations_bisection}")
        print(f"  - 最终结果: {result_bisection:.6f}")
        
        print(f"\n迭代法:")
        print(f"  - 迭代次数: {iter_count_iteration}")
        print(f"  - 函数值计算次数: {function_evaluations_iteration}")
        print(f"  - 最终结果: {result_iteration:.6f}")
        
        print(f"\n比较:")
        if function_evaluations_bisection < function_evaluations_iteration:
            print(f"  二分法计算量更少（少 {function_evaluations_iteration - function_evaluations_bisection} 次函数计算）")
        elif function_evaluations_iteration < function_evaluations_bisection:
            print(f"  迭代法计算量更少（少 {function_evaluations_bisection - function_evaluations_iteration} 次函数计算）")
        else:
            print(f"  两种方法计算量相同")
        
        print(f"\n理论分析:")
        print(f"  二分法: 每次迭代将区间缩小一半，需要约 log2((b-a)/ε) 次迭代")
        print(f"          理论迭代次数: {math.ceil(math.log2((1.0 - 0.0) / epsilon)):.0f}")
        print(f"  迭代法: 收敛速度取决于迭代函数的收敛因子")
        print(f"          如果 |φ'(x*)| < 1，则线性收敛")


def compare_methods():
    """比较三种方法的优缺点"""
    print("=" * 80)
    print("方法比较与优缺点分析")
    print("=" * 80)
    print("""
1. 不动点迭代法:
   优点:
   - 实现简单，只需要一个迭代函数
   - 不需要计算导数
   - 对某些问题收敛稳定
   
   缺点:
   - 收敛速度较慢（线性收敛）
   - 需要选择合适的迭代格式才能保证收敛
   - 收敛性依赖于迭代函数的性质

2. Steffensen加速方法:
   优点:
   - 不需要计算导数
   - 收敛速度比不动点迭代快（平方收敛）
   - 可以加速慢收敛的不动点迭代
   
   缺点:
   - 需要计算两次迭代函数值
   - 当分母接近零时可能不稳定
   - 对初始值的选择有一定要求

3. Newton迭代法:
   优点:
   - 收敛速度最快（平方收敛）
   - 理论成熟，应用广泛
   - 对于单根收敛性很好
   
   缺点:
   - 需要计算函数的导数
   - 当导数接近零时可能失败
   - 对初始值的选择要求较高
   - 可能发散或收敛到错误的根

总结:
   - 如果函数导数容易计算，Newton迭代法通常是最佳选择
   - 如果无法计算导数，Steffensen加速方法是不错的选择
   - 不动点迭代法虽然简单，但收敛速度较慢，适合作为基础方法
    """)


def main():
    """主函数"""
    # 问题3: 比较二分法和迭代法的计算量
    compare_problem3()
    
    print("\n" + "=" * 80 + "\n")
    
    # 测试函数1
    test_function1()
    
    # 测试函数2
    test_function2()
    
    # 方法比较
    compare_methods()


if __name__ == "__main__":
    main()

