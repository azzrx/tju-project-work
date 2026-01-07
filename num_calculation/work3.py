import math

def f(x):
    """被积函数 f(x) = sqrt(x) * ln(x)"""
    if x <= 0:
        return 0
    return math.sqrt(x) * math.log(x)

def composite_trapezoidal_rule(a, b, h):
    """复化梯形公式计算积分"""
    n = int((b - a) / h)
    if n <= 0:
        n = 1
        h = b - a
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    
    # 复化梯形公式: h/2 * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(xn-1) + f(xn)]
    result = h / 2 * (y[0] + y[-1] + 2 * sum(y[1:-1]))
    return result, n

def romberg_integration(a, b, epsilon):
    """龙贝格算法计算积分"""
    # 初始化龙贝格表
    R = []
    
    # 计算第一列（复化梯形公式）
    h = b - a
    # R[0][0] = h/2 * (f(a) + f(b))
    R.append([h / 2 * (f(a) + f(b))])
    
    k = 1
    
    while True:
        h = h / 2
        # 复化梯形公式递推
        sum_term = 0
        for i in range(1, 2**k, 2):
            sum_term += f(a + i * h)
        
        # 添加新行
        R.append([R[k-1][0] / 2 + h * sum_term])
        
        # Richardson外推
        for j in range(1, k + 1):
            R[k].append((4**j * R[k][j-1] - R[k-1][j-1]) / (4**j - 1))
        
        # 检查精度
        if k > 0 and abs(R[k][k] - R[k-1][k-1]) < epsilon:
            return R[k][k], k, h
        
        k += 1
        if k > 15:  # 防止无限循环
            break
    
    return R[-1][-1], k-1, h

def compare_methods(a, b, initial_h, epsilon):
    """比较复化梯形公式和龙贝格算法"""
    print("数值积分计算对比")
    print(f"被积函数: f(x) = √x × ln(x)")
    print(f"积分区间: [{a}, {b}]")
    print(f"精度要求: ε = {epsilon}")
    print("-" * 50)
    
    # 复化梯形公式
    print("1. 复化梯形公式:")
    h = initial_h
    iterations_trap = 0
    
    prev_result = 0
    final_trap_result = 0
    while True:
        result_trap, n = composite_trapezoidal_rule(a, b, h)
        error_estimate = abs(result_trap - prev_result) / 3  # 误差估计
        
        print(f"   步长 h = {h:.6f}, 划分次数 = {n}, 积分值 = {result_trap:.8f}", end="")
        if iterations_trap > 0:
            print(f", 估计误差 = {error_estimate:.2e}")
        else:
            print()
        
        if iterations_trap > 0 and error_estimate < epsilon:
            print(f"   达到精度要求，最终步长 h = {h:.6f}，划分 {n} 次")
            final_trap_result = result_trap
            break
            
        prev_result = result_trap
        h = h / 2
        iterations_trap += 1
        
        if iterations_trap > 20:  # 防止无限循环
            print("   超过最大迭代次数")
            final_trap_result = result_trap
            break
    
    print()
    
    # 龙贝格算法
    print("2. 龙贝格算法:")
    result_romberg, divisions, final_h = romberg_integration(a, b, epsilon)
    print(f"   达到精度要求，最终步长 h = {final_h:.6f}，划分 {2**divisions} 次")
    print(f"   积分值 = {result_romberg:.8f}")
    
    print()
    print("3. 方法对比:")
    print(f"   复化梯形公式最终计算值: {final_trap_result:.8f}")
    print(f"   复化梯形公式需要 {n} 次划分，步长 h = {h:.6f}")
    print(f"   龙贝格算法最终计算值: {result_romberg:.8f}")
    print(f"   龙贝格算法需要 {2**divisions} 次划分，步长 h = {final_h:.6f}")
    
    if n > 2**divisions:
        print("   龙贝格算法在效率上优于复化梯形公式")
    else:
        print("   复化梯形公式在效率上优于龙贝格算法")
    return final_trap_result, result_romberg

def main():
    print("第四章 数值积分")
    print("使用复化梯形公式和龙贝格算法计算积分")
    print("=" * 50)
    
    # 输入参数
    a = float(input("请输入积分下限 a: "))
    b = float(input("请输入积分上限 b: "))
    initial_h = float(input("请输入初始步长 h: "))
    epsilon = float(input("请输入精度要求 ε: "))
    
    print()
    compare_methods(a, b, initial_h, epsilon)


if __name__ == "__main__":
    
        main()