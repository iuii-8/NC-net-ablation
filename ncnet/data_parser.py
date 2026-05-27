import pandas as pd
import json
import re
import os


def extract_labels_from_vega_zero(vega_zero_str):
    if pd.isna(vega_zero_str):
        return {
            "mark": "none", "aggregate": "none", "filter": "no",
            "group": "no", "sort": "none", "bin": "no"
        }

    s = str(vega_zero_str).lower().strip()

    # 1. 抽取 mark (bar/line/scatter/pie)
    mark_match = re.search(r'\bmark\s+(\w+)\b', s)
    mark = mark_match.group(1) if mark_match else "none"

    # 2. 抽取 aggregate (count/mean/sum/min/max)
    agg_match = re.search(r'\baggregate\s+(\w+)\b', s)
    aggregate = agg_match.group(1) if agg_match else "none"

    # 3. 抽取 filter (通过关键词判断是否存在)
    filter_val = "yes" if "filter" in s else "no"

    # 4. 抽取 group (通过关键词判断是否存在)
    group_val = "yes" if "group" in s else "no"

    # 5. 抽取 sort (asc/desc)
    sort_match = re.search(r'\bsort\s+\w+\s+(desc|asc)\b', s)
    sort_val = sort_match.group(1) if sort_match else "none"

    # 6. 抽取 bin (通过关键词判断是否存在)
    bin_val = "yes" if "bin" in s else "no"

    return {
        "mark": mark,
        "aggregate": aggregate,
        "filter": filter_val,
        "group": group_val,
        "sort": sort_val,
        "bin": bin_val
    }


def process_all_files(file_paths, output_dir="processed_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"未找到文件: {path}，跳过该文件的处理。")
            continue

        print(f"正在处理 {name} 数据集 ({path})...")
        df = pd.read_csv(path)

        processed_list = []
        for _, row in df.iterrows():
            labels = extract_labels_from_vega_zero(row.get('vega_zero', ''))

            # 构建输出结构
            item = {
                "tvBench_id": str(row.get('tvBench_id', '')),
                "db_id": str(row.get('db_id', '')),
                "query": str(row.get('question', '')),  # 模型的输入端：自然语言问题
                "schema": str(row.get('mentioned_columns', '')),  # 辅助的 schema 提示
                "gold_template": str(row.get('vega_zero', '')),  # 原始的完整语法
                "labels": labels  # 分类头需要的标签
            }
            processed_list.append(item)

        output_path = os.path.join(output_dir, f"{name}_labeled.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_list, f, ensure_ascii=False, indent=2)

        print(f"共生成 {len(processed_list)} 条数据 -> 保存在 {output_path}")


if __name__ == "__main__":
    files = {
        "train": "train.csv",
        "dev": "dev.csv",
        "test": "test.csv"
    }
    process_all_files(files)