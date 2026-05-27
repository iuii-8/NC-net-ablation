"""Build official ncNet dataset files using AutoTemplate predictions.

This script converts the AutoTemplate predictor outputs into the original ncNet
CSV format. It reads official ncNet dataset_final/*.csv and replaces the
no-template rows' <C>...</C> template segment with a predicted chart template.

Input:
    - NC/dataset/dataset_final/{train,dev,test}.csv
    - ncnet/processed_data/{train,dev,test}_with_predicted_template.json

Output:
    - NC/dataset/dataset_autotemplate/{train,dev,test}.csv

The output preserves ncNet's required columns:
    tvBench_id, db_id, chart, hardness, query, question, vega_zero,
    mentioned_columns, mentioned_values, query_template, source, labels,
    token_types
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


BASE_TEMPLATE = (
    "mark {mark} data {table} encoding x [X] y aggregate [AggFunction] [Y] "
    "color [Z] transform filter [F] group [G] bin [B] sort {sort_slot} topk [K]"
)


def normalize_mark(mark):
    mark = str(mark).lower().strip()
    aliases = {
        "pie": "arc",
        "scatter": "point",
    }
    return aliases.get(mark, mark)


def extract_table_name(vega_zero):
    tokens = str(vega_zero).lower().split()
    if "data" not in tokens:
        return "[D]"
    data_idx = tokens.index("data")
    if data_idx + 1 >= len(tokens):
        return "[D]"
    return tokens[data_idx + 1]


def infer_sort_slot(gold_template, predicted_sort):
    """Infer official ncNet sort slot from gold syntax if possible.

    The predicted template only tells asc/desc/none, while official ncNet's
    chart template uses a slot such as [X] asc or [Y] desc. We use the gold
    target syntax only to infer whether the sort axis is x/y/o. The sort
    direction still comes from AutoTemplate's predicted label.
    """
    predicted_sort = str(predicted_sort).lower().strip()
    if predicted_sort == "none":
        return "[S]"

    tokens = str(gold_template).lower().split()
    axis = "[O]"
    if "sort" in tokens:
        sort_idx = tokens.index("sort")
        if sort_idx + 1 < len(tokens):
            sort_axis = tokens[sort_idx + 1]
            if sort_axis == "x":
                axis = "[X]"
            elif sort_axis == "y":
                axis = "[Y]"

    return f"{axis} {predicted_sort}"


def predicted_labels_to_official_template(predicted_labels, table_name, gold_template):
    mark = normalize_mark(predicted_labels.get("mark", "[T]"))
    sort_slot = infer_sort_slot(gold_template, predicted_labels.get("sort", "none"))

    return BASE_TEMPLATE.format(
        mark=mark,
        table=table_name,
        sort_slot=sort_slot,
    )


def replace_chart_template_in_source(source, new_template):
    pattern = r"<C>.*?</C>"
    replacement = f"<C> {new_template} </C>"
    return re.sub(pattern, replacement, str(source), count=1)


def get_token_types(input_source):
    """Reproduce ncNet preprocessing token type generation."""
    token_types = ""

    n_match = re.findall(r"<N>.*?</N>", input_source)
    c_match = re.findall(r"<C>.*?</C>", input_source)
    col_match = re.findall(r"<COL>.*?</COL>", input_source)
    val_match = re.findall(r"<VAL>.*?</VAL>", input_source)

    if not (n_match and c_match and col_match and val_match):
        raise ValueError(f"Invalid ncNet source format: {input_source}")

    for _ in n_match[0].split(" "):
        token_types += " nl"

    for _ in c_match[0].split(" "):
        token_types += " template"

    token_types += " table table"

    for _ in col_match[0].split(" "):
        token_types += " col"

    for _ in val_match[0].split(" "):
        token_types += " value"

    token_types += " table"

    return token_types.strip()


def load_predictions(prediction_path):
    with open(prediction_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_autotemplate_split(official_csv, prediction_json, output_csv, replace_mode):
    df = pd.read_csv(official_csv)
    predictions = load_predictions(prediction_json)

    if len(df) != len(predictions):
        raise ValueError(
            f"Row count mismatch: {official_csv} has {len(df)} rows, "
            f"but {prediction_json} has {len(predictions)} rows."
        )

    replaced_count = 0
    kept_count = 0

    for idx, row in df.iterrows():
        source = str(row["source"])
        should_replace = replace_mode == "all" or "[T]" in source or "[t]" in source.lower()

        if should_replace:
            pred_item = predictions[idx]
            predicted_labels = pred_item["predicted_labels"]
            table_name = extract_table_name(row["vega_zero"])
            official_template = predicted_labels_to_official_template(
                predicted_labels=predicted_labels,
                table_name=table_name,
                gold_template=row["vega_zero"],
            )
            new_source = replace_chart_template_in_source(source, official_template)
            new_source = " ".join(new_source.split())

            df.at[idx, "query_template"] = official_template
            df.at[idx, "source"] = new_source
            df.at[idx, "token_types"] = get_token_types(new_source)
            replaced_count += 1
        else:
            kept_count += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    return replaced_count, kept_count, len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Build official ncNet dataset files with AutoTemplate predictions."
    )
    parser.add_argument(
        "--official-data-dir",
        default="../NC/dataset/dataset_final",
        help="Directory containing official ncNet dataset_final CSV files.",
    )
    parser.add_argument(
        "--prediction-dir",
        default="processed_data",
        help="Directory containing *_with_predicted_template.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="../NC/dataset/dataset_autotemplate",
        help="Output directory for AutoTemplate ncNet CSV files.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "dev", "test", "all"],
        help="Which split to process.",
    )
    parser.add_argument(
        "--replace-mode",
        default="no_template",
        choices=["no_template", "all"],
        help="Replace only rows with [T] templates, or replace all rows.",
    )
    args = parser.parse_args()

    official_data_dir = Path(args.official_data_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    print("Building official ncNet AutoTemplate dataset...")
    print(f"Official data dir: {official_data_dir}")
    print(f"Prediction dir: {prediction_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Replace mode: {args.replace_mode}")
    print()

    for split in splits:
        official_csv = official_data_dir / f"{split}.csv"
        prediction_json = prediction_dir / f"{split}_with_predicted_template.json"
        output_csv = output_dir / f"{split}.csv"

        if not official_csv.exists():
            print(f"[SKIP] Missing official CSV: {official_csv}")
            continue
        if not prediction_json.exists():
            print(f"[SKIP] Missing prediction JSON: {prediction_json}")
            continue

        replaced_count, kept_count, total_count = build_autotemplate_split(
            official_csv=official_csv,
            prediction_json=prediction_json,
            output_csv=output_csv,
            replace_mode=args.replace_mode,
        )

        print(f"[{split}] total={total_count}, replaced={replaced_count}, kept={kept_count}")
        print(f"       saved -> {output_csv}")

    print("\nDone.")
    print("Next step: run official ncNet with -data_dir pointing to dataset_autotemplate.")


if __name__ == "__main__":
    main()
