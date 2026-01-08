# 人工智能实验代码库

本文件夹包含三个不同的人工智能领域经典算法与应用的 Python 实现。这些代码涵盖了深度学习、启发式搜索和专家系统三个方向。

## 文件列表与详细介绍

### 1. LeNet-5.py：手写数字识别 (CNN)

基于 **PyTorch** 框架实现的经典 **LeNet-5** 卷积神经网络，用于 **MNIST** 手写数字数据集的分类任务。

*   **功能**：
    *   自动下载并预处理 MNIST 数据集（调整尺寸为 32x32）。
    *   构建包含卷积层、池化层和全连接层的 LeNet-5 模型。
    *   在 CPU 或 GPU 上进行 5 个轮次的训练。
    *   评估模型在测试集上的准确率，并保存模型权重。
*   **如何运行**：
    ```bash
    python LeNet-5.py
    ```
*   **关键技术**：PyTorch, CNN, MNIST, Adam 优化器。

### 2. A- Algorithm.py：八数码问题求解 (A 算法)

使用 **A 算法**（在本代码的具体配置中，演示了不使用启发函数 $h(n)=0$ 的情况，即退化为广度优先搜索/Dijkstra）来解决经典的 **八数码问题 (8-Puzzle)**。

*   **功能**：
    *   定义了八数码问题的初始状态和目标状态。
    *   实现了状态的后继生成（上下左右移动空格）。
    *   实现了通用的 A* 搜索框架。
    *   **注意**：代码默认演示 `use_heuristic=False` 的情况，用于展示基础搜索过程。
    *   输出从初始状态到目标状态的完整路径及步数。
*   **如何运行**：
    ```bash
    python "A- Algorithm.py"
    ```
    *(注意文件名中有空格，需加引号)*
*   **关键技术**：A* 搜索框架, 状态空间搜索, 优先队列 (Heapq)。

### 3. system_animals.py：产生式动物识别系统

一个基于 **产生式规则 (Production System)** 的小型专家系统，用于通过特征识别动物。

*   **功能**：
    *   **正向推理**：从已知事实出发，不断匹配规则库，推导出新的结论。
    *   **规则库**：包含 15 条规则，可识别 7 种动物（虎、金钱豹、斑马、长颈鹿、鸵鸟、企鹅、信天翁）。
    *   **交互模式**：支持用户交互式输入特征（如“有毛发”、“吃肉”等）进行识别。
    *   **测试模式**：内置了标准测试用例，可一键验证推理逻辑。
*   **如何运行**：
    *   **交互模式**（默认）：
        ```bash
        python system_animals.py
        ```
    *   **运行测试用例**：
        ```bash
        python system_animals.py --run-tests
        ```
*   **关键技术**：正向推理 (Forward Chaining), 知识表示, 专家系统。

## 环境要求

要运行上述所有脚本，请确保安装了 Python 3.x 以及以下依赖库：

| 文件 | 依赖库 | 安装命令 |
| :--- | :--- | :--- |
| **LeNet-5.py** | `torch`, `torchvision` | `pip install torch torchvision` |
| **A- Algorithm.py** | 标准库 (`heapq`, `collections`) | 无需安装 |
| **system_animals.py** | 标准库 (`argparse`, `sys`, `typing`) | 无需安装 |

## 快速开始

建议首先创建一个虚拟环境，然后安装 PyTorch，即可运行所有代码：

```bash
# 1. 安装 PyTorch (根据你的环境选择合适的版本)
pip install torch torchvision

# 2. 运行 LeNet-5
python LeNet-5.py

# 3. 运行八数码问题求解
python "A- Algorithm.py"

# 4. 运行动物识别系统
python system_animals.py
```
