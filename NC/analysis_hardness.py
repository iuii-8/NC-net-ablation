import pandas as pd

def calc_hardness(query):
    """根据 Vega-Zero 语法的复杂度动态推导难度标签"""
    query = str(query).lower()
    # 统计复杂组件的数量
    complexity = sum(1 for kw in ['aggregate ', 'filter ', 'group ', 'bin ', 'sort ', 'topk '] if kw in query)
    
    if complexity == 0: return 'Easy'
    elif complexity == 1: return 'Medium'
    elif complexity == 2: return 'Hard'
    else: return 'Extra Hard'

try:
    print("正在解析数据并动态计算难度，请稍候...")
    
    # 1. 读取原始测试集，获取 w/o template 样本（2460条）的总难度分布
    df_all = pd.read_csv('dataset/dataset_final/test.csv')
    df_wo = df_all[df_all['source'].str.lower().str.contains(r'\[t\]', na=False)].copy()
    df_wo['hardness'] = df_wo['labels'].apply(calc_hardness)
    totals = df_wo['hardness'].value_counts().to_dict()

    # 2. 读取两份错误分析文件
    err_base = pd.read_csv('dataset/dataset_final/baseline_errors.csv')
    err_auto = pd.read_csv('dataset/dataset_autotemplate/autotemplate_errors.csv')

    # 为错误样本打上难度标签
    err_base['hardness'] = err_base['gold'].apply(calc_hardness)
    err_auto['hardness'] = err_auto['gold'].apply(calc_hardness)

    # 统计各个难度下的错误数量
    errors_base_count = err_base['hardness'].value_counts().to_dict()
    errors_auto_count = err_auto['hardness'].value_counts().to_dict()

    print("\n| 难度等级 (Hardness) | 样本数量 | Baseline 准确率 | AutoTemplate 准确率 | 提升幅度 (Gap) |")
    print("|---|---:|---:|---:|---:|")

    # 3. 循环计算最终准确率并打印表格
    for level in ['Easy', 'Medium', 'Hard', 'Extra Hard']:
        total = totals.get(level, 0)
        if total == 0: continue
        
        err_b = errors_base_count.get(level, 0)
        err_a = errors_auto_count.get(level, 0)

        # 准确率 = (总数 - 错误数) / 总数
        acc_b = ((total - err_b) / total) * 100
        acc_a = ((total - err_a) / total) * 100
        gap = acc_a - acc_b
        
        print(f"| {level} | {total} | {acc_b:.2f}% | {acc_a:.2f}% | +{gap:.2f} pp |")

except Exception as e:
    print(f"脚本执行出错: {e}")