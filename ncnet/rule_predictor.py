import pandas as pd
import re
import os


class RuleBasedTemplatePredictor:
    def __init__(self):
        self.rules = {
            "mark": {
                "bar": [r'\b(bar|column)\b'],
                "line": [r'\b(line|trend|over time|curve)\b'],
                "pie": [r'\b(pie|proportion|share|arc)\b'],
                "scatter": [r'\b(scatter|correlation|point)\b']
            },
            "aggregate": {
                "mean": [r'\b(average|mean|avg)\b'],
                "sum": [r'\b(sum|total)\b'],
                "count": [r'\b(how many|number of|count)\b'],
                "max": [r'\b(maximum|highest|most)\b'],
                "min": [r'\b(minimum|lowest|least)\b']
            },
            "sort": {
                "desc": [r'\b(descending|desc|highest|top)\b'],
                "asc": [r'\b(ascending|asc|lowest|bottom)\b']
            }
        }

    def predict_single_query(self, text):
        q = str(text).lower()
        preds = {
            "mark": "none", "aggregate": "none", "filter": "no",
            "group": "no", "sort": "none", "bin": "no"
        }

        # 1. 匹配图表 mark
        for label, patterns in self.rules["mark"].items():
            if any(re.search(p, q) for p in patterns):
                preds["mark"] = label
                break

        # 2. 匹配聚合方式
        for label, patterns in self.rules["aggregate"].items():
            if any(re.search(p, q) for p in patterns):
                preds["aggregate"] = label
                break

        # 3. 匹配过滤条件 filter
        if any(w in q for w in ["where", "filter", "greater than", "less than", "equal"]):
            preds["filter"] = "yes"

        # 4. 匹配分组 group
        if any(w in q for w in ["by", "each", "per", "grouped"]):
            preds["group"] = "yes"

        # 5. 匹配排序 sort
        for label, patterns in self.rules["sort"].items():
            if any(re.search(p, q) for p in patterns):
                preds["sort"] = label
                break

        # 6. 匹配分箱 bin
        if any(w in q for w in ["bin", "binned", "range", "interval", "weekday"]):
            preds["bin"] = "yes"

        return preds

    def to_pt_token_string(self, preds):
        return (f"<PT> mark {preds['mark']} aggregate {preds['aggregate']} "
                f"group {preds['group']} filter {preds['filter']} "
                f"sort {preds['sort']} bin {preds['bin']} </PT>")


def generate_baseline_test_csv(input_csv, output_csv="test_with_rule_template.csv"):
    """
    读取测试集 CSV，追加规则预测的模板串
    """
    if not os.path.exists(input_csv):
        print(f"无法生成 Baseline，找不到输入文件: {input_csv}")
        return

    print(f"正在通过规则预测器为 {input_csv} 注入预测模板...")
    df = pd.read_csv(input_csv)
    predictor = RuleBasedTemplatePredictor()

    pt_templates = []
    hybrid_inputs = []

    for _, row in df.iterrows():
        question = row.get('question', '')
        # 预测并转化为 token 串
        preds = predictor.predict_single_query(question)
        pt_str = predictor.to_pt_token_string(preds)
        pt_templates.append(pt_str)

        # 拼接输入: query + <PT>...</PT> + schema
        # 如果原始格式里带了标签对，也可以在这直接用文本拼接
        schema = str(row.get('mentioned_columns', ''))
        hybrid_input = f"{question} {pt_str} {schema}"
        hybrid_inputs.append(hybrid_input)

    df['predicted_pt_template'] = pt_templates
    df['ncnet_hybrid_input'] = hybrid_inputs

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"注入完毕，成功导出基线测试集 -> {output_csv}")


if __name__ == "__main__":
    # 可更换输入文件
    generate_baseline_test_csv("test.csv", "test_with_rule_template.csv")