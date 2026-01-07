"""
产生式动物识别系统 (Production System for Animal Recognition)

实现说明:
- 使用正向推理 (Forward Chaining)
- 规则库包含 15 条规则 (r1..r15)，用于从输入事实推导中间概念并最终识别动物
- 支持交互问答、和自动运行示例测试 (`--run-tests`)

"""


from typing import Set, List
import argparse
import sys


# 代表一条产生式规则的简单类：包含规则 id、前件条件集合和结论
class Rule:
    def __init__(self, rid: str, conditions: Set[str], conclusion: str):
        self.rid = rid
        self.conditions = set(conditions)
        self.conclusion = conclusion

    def is_triggered(self, facts: Set[str]) -> bool:
        """判断当前事实集合是否满足规则的所有前件（子集关系）。"""
        return self.conditions.issubset(facts)


def build_rule_base() -> List[Rule]:
    """构造并返回规则库（规则按列表顺序存放，顺序即为当前的简单优先级）。

    说明：条件和结论均使用中文短语，便于交互与报告。
    """
    R = [
        Rule('r1', {'有毛发'}, '哺乳动物'),
        Rule('r2', {'有奶'}, '哺乳动物'),
        Rule('r3', {'有羽毛'}, '鸟'),
        Rule('r4', {'会飞', '会下蛋'}, '鸟'),
        Rule('r5', {'吃肉'}, '食肉动物'),
        Rule('r6', {'有犬齿', '有爪', '眼盯前方'}, '食肉动物'),
        Rule('r7', {'哺乳动物', '有蹄'}, '有蹄类动物'),
        Rule('r8', {'哺乳动物', '反刍动物'}, '有蹄类动物'),
        Rule('r9', {'哺乳动物', '食肉动物', '黄褐色', '身上有暗斑点'}, '金钱豹'),
        Rule('r10', {'哺乳动物', '食肉动物', '黄褐色', '身上有黑色条纹'}, '虎'),
        Rule('r11', {'有蹄类动物', '有长脖子', '有长腿', '身上有暗斑点'}, '长颈鹿'),
        Rule('r12', {'有蹄类动物', '身上有黑色条纹'}, '斑马'),
        Rule('r13', {'鸟', '有长脖子', '有长腿', '不会飞', '有黑白二色'}, '鸵鸟'),
        Rule('r14', {'鸟', '会游泳', '不会飞', '有黑白二色'}, '企鹅'),
        Rule('r15', {'鸟', '善飞'}, '信天翁'),
    ]
    return R
def forward_chain(initial_facts: Set[str], rules: List[Rule], verbose: bool = True) -> Set[str]:
    """正向推理引擎（循环匹配-推导直到收敛）。

    参数:
    - initial_facts: 初始事实集合（字符串集合）
    - rules: 规则列表
    - verbose: 若为 True，则打印每次匹配和推导的日志

    返回:
    - 收敛后的事实集合（包含初始事实与所有推导出的事实）
    """
    facts = set(initial_facts)  # 当前事实集合
    fired = set()  # 记录已触发过的规则 id，防止重复触发和重复打印
    changed = True  # 控制迭代：当一轮没有新增事实时停止

    while changed:
        changed = False
        # 遍历规则，按列表顺序检查是否可触发
        for rule in rules:
            # 如果规则已被触发过则跳过（避免重复执行同一规则）
            if rule.rid in fired:
                continue
            if rule.is_triggered(facts):
                # 规则前件满足
                if rule.conclusion not in facts:
                    # 将结论加入事实集合并记录发生变化
                    facts.add(rule.conclusion)
                    changed = True
                    if verbose:
                        # 打印清晰的匹配/推导信息，便于观察推理链
                        conds = ', '.join(sorted(rule.conditions))
                        print(f"匹配规则 {rule.rid}：如果 {conds} → 推导 ‘{rule.conclusion}’")
                else:
                    # 如果结论已存在，仍把规则标记为触发以避免重复判断
                    if verbose:
                        conds = ', '.join(sorted(rule.conditions))
                        print(f"规则 {rule.rid} 可触发但结论 ‘{rule.conclusion}’ 已存在，跳过添加")
                fired.add(rule.rid)

    return facts


def recognize_animal_from_facts(facts: Set[str], rules: List[Rule], verbose: bool = True) -> List[str]:
    """对给定初始事实运行推理，并返回被识别出的目标动物（如果有）。

    目标动物集合在此处硬编码为题目要求的 7 种。
    """
    final_facts = forward_chain(facts, rules, verbose=verbose)
    # 7 种目标动物（按需求固定顺序）
    species = ['虎', '金钱豹', '斑马', '长颈鹿', '鸵鸟', '企鹅', '信天翁']
    found = [s for s in species if s in final_facts]
    return found


# 内置示例用例，用于快速验证推理链是否按预期工作
SAMPLE_CASES = {
    '长颈鹿': {'身上有暗斑点', '有长脖子', '有长腿', '有奶', '有蹄'},
    '金钱豹': {'有奶', '吃肉', '黄褐色', '身上有暗斑点', '有爪'},
    '虎': {'有奶', '吃肉', '黄褐色', '身上有黑色条纹', '有犬齿'},
    '斑马': {'有奶', '有蹄', '身上有黑色条纹'},
    '鸵鸟': {'有羽毛', '有长脖子', '有长腿', '不会飞', '有黑白二色'},
    '企鹅': {'有羽毛', '会游泳', '不会飞', '有黑白二色'},
    '信天翁': {'有羽毛', '善飞'},
}


def interactive_input(possible_features: List[str]) -> Set[str]:
    """逐条询问用户对每个可能特征的有无，返回事实集合。

    输入使用 Y/N（回车视为 N）。返回的事实与规则中使用的短语一致。
    """
    print("请对下列特征逐一输入 Y/N（回车表示 N）：")
    facts = set()
    for feat in possible_features:
        ans = input(f"该动物是否有特征 ‘{feat}’? [Y/N] ").strip().lower()
        if ans.startswith('y'):
            facts.add(feat)
    return facts


def run_sample_tests(rules: List[Rule]):
    """运行内置示例并打印每个用例的推理过程与结果，便于快速验证。"""
    print("运行示例测试：")
    for name, feats in SAMPLE_CASES.items():
        print('\n' + '='*40)
        print(f"测试用例: {name}")
        print(f"初始事实: {sorted(feats)}")
        found = recognize_animal_from_facts(set(feats), rules, verbose=True)
        if found:
            print(f"最终识别结果: {', '.join(found)}")
        else:
            print("最终识别结果: 无法识别")


def main(argv=None):
    parser = argparse.ArgumentParser(description='产生式动物识别系统')
    parser.add_argument('--run-tests', action='store_true', help='运行内置示例测试并退出')
    args = parser.parse_args(argv)

    rules = build_rule_base()

    # 列出所有可能会在问答中使用到的特征（用于交互问答时逐项询问）
    all_possible_features = sorted({
        '有毛发', '有奶', '有羽毛', '会飞', '会下蛋', '吃肉', '有犬齿', '有爪', '眼盯前方',
        '有蹄', '反刍动物', '黄褐色', '身上有暗斑点', '身上有黑色条纹', '有长脖子', '有长腿',
        '不会飞', '有黑白二色', '会游泳', '善飞'
    })

    if args.run_tests:
        run_sample_tests(rules)
        return

    print("=== 产生式动物识别系统 ===")
    print("选项: 1) 交互问答  2)  运行示例测试 ")
    while True:
        choice = input("请输入选项编号: ").strip()
        if choice == '1':
            # 交互问答：逐条询问并打印推理过程与最终结果
            facts = interactive_input(all_possible_features)
            print(f"您输入的初始事实: {sorted(facts)}")
            found = recognize_animal_from_facts(facts, rules, verbose=True)
            if found:
                print(f"最终识别结果: {', '.join(found)}")
            else:
                print("最终识别结果: 无法识别")
        elif choice == '2':
            # 运行内置示例并打印详细推理过程
            run_sample_tests(rules)
        else:
            print('无效选项，请重试。')


if __name__ == '__main__':
    main()