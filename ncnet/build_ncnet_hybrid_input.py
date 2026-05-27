"""Build hybrid inputs for ncNet by combining query + predicted template + schema.

This script takes the output of predict_template.py and constructs the final
input format that ncNet will consume for training or inference.

Usage:
    python build_ncnet_hybrid_input.py --split test
    python build_ncnet_hybrid_input.py --split train --output-dir hybrid_data
"""

import argparse
import csv
import json
from pathlib import Path

from label_config import TASKS, labels_to_pt_string


def build_hybrid_input(query, pt_template, schema):
    """Construct the hybrid input: query + predicted template + schema."""
    parts = [query.strip()]
    if pt_template:
        parts.append(pt_template.strip())
    if schema.strip():
        parts.append(schema.strip())
    return " ".join(parts)


def process_split(input_path, output_json_path, output_csv_path=None):
    """Process a single split and generate hybrid inputs."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data = []
    for item in data:
        query = item["query"]
        schema = item["schema"]
        pt_template = item.get("predicted_pt_template", "")

        if not pt_template and "predicted_labels" in item:
            pt_template = labels_to_pt_string(item["predicted_labels"])

        hybrid_input = build_hybrid_input(query, pt_template, schema)

        entry = {
            "tvBench_id": item.get("tvBench_id", ""),
            "db_id": item.get("db_id", ""),
            "query": query,
            "schema": schema,
            "predicted_pt_template": pt_template,
            "hybrid_input": hybrid_input,
            "gold_template": item.get("gold_template", ""),
        }

        if "gold_labels" in item:
            entry["gold_labels"] = item["gold_labels"]
        if "predicted_labels" in item:
            entry["predicted_labels"] = item["predicted_labels"]

        output_data.append(entry)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if output_csv_path:
        fieldnames = [
            "tvBench_id", "db_id", "query", "schema",
            "predicted_pt_template", "hybrid_input", "gold_template",
        ]
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output_data)

    return len(output_data)


def main():
    parser = argparse.ArgumentParser(
        description="Build hybrid inputs (query + predicted template + schema) for ncNet"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test", "all"],
        help="Which split to process",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="processed_data",
        help="Directory containing *_with_predicted_template.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hybrid_data",
        help="Directory to save hybrid input files",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output CSV format",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        input_path = input_dir / f"{split}_with_predicted_template.json"
        if not input_path.exists():
            print(f"[SKIP] {input_path} not found")
            continue

        output_json = output_dir / f"{split}_hybrid_input.json"
        output_csv = output_dir / f"{split}_hybrid_input.csv" if args.csv else None

        count = process_split(input_path, output_json, output_csv)
        print(f"[{split}] Processed {count} samples -> {output_json}")
        if output_csv:
            print(f"         CSV -> {output_csv}")

    print("\nDone. Hybrid inputs are ready for ncNet.")
    print("Input format: query <PT> mark X aggregate X group X filter X sort X bin X </PT> schema")


if __name__ == "__main__":
    main()
