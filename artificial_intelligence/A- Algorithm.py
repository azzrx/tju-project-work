import heapq 
import collections

# 定义初始状态和目标状态 (以元组表示，0 代表空格)
# 初始状态 : 2 8 3 / 1 _ 4 / 7 6 5 [cite: 3]
INITIAL_STATE = (2, 8, 3, 1, 0, 4, 7, 6, 5) 
# 目标状态 : 1 2 3 / 8 _ 4 / 7 6 5 [cite: 3]
GOAL_STATE = (1, 2, 3, 8, 0, 4, 7, 6, 5)

# 启发函数 h(n): 位置不符的数码数目 
def heuristic_misplaced_tiles(state):
    """
    计算 h(n): 棋局与目标棋局相比，位置不符的数码数目 
    """
    count = 0
    for i in range(9):
        # 忽略空格 0
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            count += 1
    return count

# 查找空格位置
def find_blank(state):
    return state.index(0)

# 生成后继状态 (操作符)
def generate_successors(state):
    """
    生成给定状态的所有可能的后继状态（上下左右移动 [cite: 7]）
    """
    successors = []
    blank_index = find_blank(state)
    r, c = blank_index // 3, blank_index % 3 # 空格的行和列

    # 移动方向: (行变化, 列变化)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上, 下, 左, 右 [cite: 7]

    for dr, dc in moves:
        new_r, new_c = r + dr, c + dc
        new_index = new_r * 3 + new_c

        # 检查是否越界
        if 0 <= new_r < 3 and 0 <= new_c < 3:
            # 交换空格和目标位置的棋子
            new_state_list = list(state)
            new_state_list[blank_index], new_state_list[new_index] = new_state_list[new_index], new_state_list[blank_index]
            successors.append(tuple(new_state_list))
    
    return successors

# A 算法 (A* 框架) 实现 [cite: 49, 50, 51, 52, 53, 54, 55, 56, 57]
def solve_8puzzle_A_algorithm(initial_state, goal_state, use_heuristic=True):
    # OPEN 表: 存储已生成而未考察的节点 (使用优先队列，按 f 值排序)
    # 存储格式: (f值, g值, 状态, 父节点状态)
    # 当 use_heuristic=False 时，h(n) 被视为 0，此时算法退化为只依据 g(n) 的 A 算法（即不使用启发）
    h_init = heuristic_misplaced_tiles(initial_state) if use_heuristic else 0
    open_list = [(h_init, 0, initial_state, None)] # f(n) = g(n) + h(n). 初始 g(n)=0
    
    # CLOSED 表: 记录已访问过的节点 (使用字典存储状态 -> (g值, 父节点)) [cite: 48]
    closed_list = {initial_state: (0, None)}
    
    # 路径找到后的最终状态和 g 值
    final_g = -1
    
    while open_list: # 当 open 表不为空时 [cite: 50]
        # 寻找 open 表中 f 值最小的点 current 
        f, g, current_state, parent_state = heapq.heappop(open_list)
        
        # 如果 current 是终止点，则找到结果，程序结束 [cite: 52]
        if current_state == goal_state:
            final_g = g
            print("🎉 找到目标状态!")
            break
        
        # open 表移出 current (已由 heappop 完成) [cite: 53]
        
        # 扩展 current，对每一个邻近点 (后继状态)
        for neighbor_state in generate_successors(current_state):
            new_g = g + 1 # 路径费用每一步为 1 [cite: 9]
            
            # 若它不可走或在 closed 表中，略过 [cite: 54]
            if neighbor_state in closed_list and new_g >= closed_list[neighbor_state][0]:
                continue
            
            # 若它不在 closed 表中，或找到更短路径
            if neighbor_state not in closed_list or new_g < closed_list[neighbor_state][0]:
                h = heuristic_misplaced_tiles(neighbor_state) if use_heuristic else 0
                f_new = new_g + h # f(n) = g(n) + h(n) 
                
                # 更新 closed 表: 存储更小的 g 值和父节点 [cite: 56]
                closed_list[neighbor_state] = (new_g, current_state) 
                
                # 若它不在 open 表中，加入。若在 open 表中，更新 g 值（通过加入新值，旧值 f 值更高会被忽略） [cite: 55, 56]
                heapq.heappush(open_list, (f_new, new_g, neighbor_state, current_state))
                
    # 若 open 表为空，则路径不存在 [cite: 57]
    if final_g == -1:
        return None 

    # 路径重构函数
    path = []
    state = goal_state
    while state is not None:
        path.append(state)
        # 从 closed_list 中获取父节点
        state = closed_list[state][1] 
    
    path.reverse() # 反转列表得到从初始状态到目标状态的路径
    return path

# 格式化输出状态
def print_state(state):
    for i in range(0, 9, 3):
        print(f"| {state[i] if state[i]!=0 else '_'} {state[i+1] if state[i+1]!=0 else '_'} {state[i+2] if state[i+2]!=0 else '_'} |")

# 主执行
if __name__ == "__main__":
    print("--- 实验二：A 算法求解八数码问题（不使用启发函数）--- [cite: 1, 2]")
    print("\n[初始状态 S0]")
    print_state(INITIAL_STATE)
    
    h_s0 = heuristic_misplaced_tiles(INITIAL_STATE)
    print(f"启发函数 h(S0) (位置不符数): {h_s0} ")
    
    print("\n[目标状态]")
    print_state(GOAL_STATE)
    
    print("\n--- 开始 A 算法搜索 ---")
    
    # 将 use_heuristic=False 来执行不使用启发函数的 A 算法
    result_path = solve_8puzzle_A_algorithm(INITIAL_STATE, GOAL_STATE, use_heuristic=False)
    
    if result_path:
        # 路径长度是路径中的步数 [cite: 9]
        path_length = len(result_path) - 1 
        print(f" 达到目标状态时走的路径长度 (步数): {path_length} [cite: 60]")
        print("\n--- 路径展示 --- (中间结果展示 [cite: 60])")
        for i, state in enumerate(result_path):
            g_val = i # g(n) 为实际代价，等于步数
            h_val = heuristic_misplaced_tiles(state)
            f_val = g_val + h_val
            
            print(f"\n步骤 {i} (g={g_val}, h={h_val}, f={f_val}):")
            print_state(state)
    else:
        print(" 搜索失败，未找到目标状态的路径。")