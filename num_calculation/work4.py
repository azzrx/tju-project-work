from __future__ import annotations

import random
from typing import List, Tuple

Number = float
Matrix = List[List[Number]]
Vector = List[Number]
EPS = 1e-12


def copy_matrix(a: Matrix) -> Matrix:
    return [row[:] for row in a]


def copy_vector(v: Vector) -> Vector:
    return v[:]


def gaussian_elimination_with_pivot(a: Matrix, b: Vector) -> Tuple[Matrix, Vector, Vector]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Returns:
        U: Upper triangular matrix after elimination (in-place modified copy).
        modified_b: Right-hand side after elimination.
        x: Solution vector.
    """
    n = len(a)
    u = copy_matrix(a)
    rhs = copy_vector(b)

    for k in range(n - 1):
        pivot_row = max(range(k, n), key=lambda idx: abs(u[idx][k]))
        if abs(u[pivot_row][k]) < EPS:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot_row != k:
            u[k], u[pivot_row] = u[pivot_row], u[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        for i in range(k + 1, n):
            factor = u[i][k] / u[k][k]
            u[i][k] = 0.0
            for j in range(k + 1, n):
                u[i][j] -= factor * u[k][j]
            rhs[i] -= factor * rhs[k]

    if abs(u[-1][-1]) < EPS:
        raise ValueError("Matrix is singular or nearly singular.")

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ax = sum(u[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (rhs[i] - sum_ax) / u[i][i]

    return u, rhs, x


def lu_decomposition_doolittle(a: Matrix) -> Tuple[Matrix, Matrix]:
    """Compute LU decomposition A = LU using Doolittle's method without pivoting."""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]

    for i in range(n):
        l[i][i] = 1.0

    for k in range(n):
        for j in range(k, n):
            u[k][j] = a[k][j] - sum(l[k][p] * u[p][j] for p in range(k))

        if abs(u[k][k]) < EPS:
            raise ValueError("Zero pivot encountered during LU decomposition.")

        for i in range(k + 1, n):
            numerator = a[i][k] - sum(l[i][p] * u[p][k] for p in range(k))
            l[i][k] = numerator / u[k][k]

    return l, u


def lu_solve(l: Matrix, u: Matrix, b: Vector) -> Vector:
    """Solve LUx = b via forward/back substitution."""
    n = len(l)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i] - sum(l[i][j] * y[j] for j in range(i))

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ux = sum(u[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_ux) / u[i][i]

    return x


def format_matrix(mat: Matrix, precision: int = 2, width: int = 8) -> str:
    lines = []
    fmt = f"{{:>{width}.{precision}f}}"
    for row in mat:
        line = "[" + " ".join(fmt.format(val) for val in row) + "]"
        lines.append(line)
    return "\n".join(lines)


def format_vector(vec: Vector, precision: int = 6, width: int = 10) -> str:
    fmt = f"{{:>{width}.{precision}f}}"
    return "[" + " ".join(fmt.format(val) for val in vec) + "]"


def print_augmented_matrix(u: Matrix, b: Vector) -> None:
    """Pretty-print the augmented matrix [U | b]."""
    precision = 2
    width = 8
    fmt_val = f"{{:>{width}.{precision}f}}"
    lines = []
    for row, value in zip(u, b):
        left = " ".join(fmt_val.format(val) for val in row)
        lines.append(f"[{left} | {fmt_val.format(value)}]")
    print("增广矩阵 [U | b]:")
    print("\n".join(lines))
    print()


def run_gaussian_examples() -> None:
    a = [
        [31, -13, 0, 0, 0, -10, 0, 0, 0],
        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
        [0, -9, 31, -10, 0, 0, 0, 0, 0],
        [0, 0, -10, 79, -30, 0, 0, 0, -9],
        [0, 0, 0, -30, 57, -7, 0, -5, 0],
        [0, 0, 0, 0, -7, 47, -30, 0, 0],
        [0, 0, 0, 0, 0, -30, 41, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 27, -2],
        [0, 0, 0, 0, 0, 0, 0, -2, 29],
    ]
    b = [-15, 27, -23, 0, -20, 12, -7, 7, 10]

    u, modified_b, x = gaussian_elimination_with_pivot(a, b)
    print("====== 高斯消元（列主元）示例 ======")
    print_augmented_matrix(u, modified_b)
    print("方程组解 x*:")
    print(format_vector(x))
    print()


def run_lu_examples() -> None:
    a = [
        [31, -13, 0, 0, 0, -10, 0, 0, 0],
        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
        [0, -9, 31, -10, 0, 0, 0, 0, 0],
        [0, 0, -10, 79, -30, 0, 0, 0, -9],
        [0, 0, 0, -30, 57, -7, 0, -5, 0],
        [0, 0, 0, 0, -7, 47, -30, 0, 0],
        [0, 0, 0, 0, 0, -30, 41, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 27, -2],
        [0, 0, 0, 0, 0, 0, 0, -2, 29],
    ]
    b =[-15, 27, -23, 0, -20, 12, -7, 7, 10]

    l, u = lu_decomposition_doolittle(a)
    x = lu_solve(l, u, b)

    print("====== LU 分解示例 ======")
    print("L 矩阵:")
    print(format_matrix(l, precision=6, width=10))
    print()
    print("U 矩阵:")
    print(format_matrix(u, precision=6, width=10))
    print()
    print("方程组解 x*:")
    print(format_vector(x, width=12))
    print()


def random_invertible_matrix(n: int, low: int = -10, high: int = 10) -> Matrix:
    """Generate a random invertible matrix by resampling until Gaussian elimination succeeds."""
    while True:
        mat = [[float(random.randint(low, high)) for _ in range(n)] for _ in range(n)]
        test_b = [float(random.randint(low, high)) for _ in range(n)]
        if all(abs(val) < EPS for val in test_b):
            continue
        try:
            gaussian_elimination_with_pivot(mat, test_b)
            return mat
        except ValueError:
            continue


def random_nonzero_vector(n: int, low: int = -10, high: int = 10) -> Vector:
    while True:
        vec = [float(random.randint(low, high)) for _ in range(n)]
        if any(abs(val) > EPS for val in vec):
            return vec


def run_random_tests() -> None:
    n = 20
    a = random_invertible_matrix(n)
    b = random_nonzero_vector(n)

    u_gauss, modified_b, x_gauss = gaussian_elimination_with_pivot(a, b)
    l, u = lu_decomposition_doolittle(a)
    x_lu = lu_solve(l, u, b)

    max_diff = max(abs(x_gauss[i] - x_lu[i]) for i in range(n))
    print("====== 随机矩阵测试 (n = 20) ======")
    print("随机矩阵 A 经过高斯消元后的增广矩阵 [U | b]:")
    print_augmented_matrix(u_gauss, modified_b)
    print("高斯消元得到的解 x:")
    print(format_vector(x_gauss, width=12))
    print()
    print("随机矩阵 A 的 LU 分解结果:")
    print("L 矩阵:")
    print(format_matrix(l, precision=4, width=9))
    print()
    print("U 矩阵:")
    print(format_matrix(u, precision=4, width=9))
    print()
    print("LU 分解求解得到的解 x:")
    print(format_vector(x_lu, width=12))
    print()
    print(f"两种方法解向量的最大差值: {max_diff:.6e}")
    print()


def main() -> None:
    random.seed(0)
    run_gaussian_examples()
    run_lu_examples()
    run_random_tests()


if __name__ == "__main__":
    main()


