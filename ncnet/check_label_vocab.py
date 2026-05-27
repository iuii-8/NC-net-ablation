import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from label_config import LABEL_VOCABS, TASKS, normalize_label


DEFAULT_SPLITS = ["train", "dev", "test"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_split(path):
    data = load_json(path)
    counters = {task: Counter() for task in TASKS}
    unknowns = defaultdict(Counter)
    missing = Counter()

    for item in data:
        labels = item.get("labels", {})
        for task in TASKS:
            if task not in labels:
                missing[task] += 1
                continue

            raw_label = labels[task]
            normalized = normalize_label(task, raw_label)
            counters[task][normalized] += 1

            if normalized not in LABEL_VOCABS[task]:
                unknowns[task][str(raw_label)] += 1

    return len(data), counters, unknowns, missing


def format_counter(counter, total):
    rows = []
    for label, count in counter.most_common():
        ratio = count / total * 100 if total else 0.0
        rows.append(f"    {label:<10} {count:>7}  {ratio:>6.2f}%")
    return "\n".join(rows) if rows else "    <empty>"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect template label vocabularies in processed_data/*.json."
    )
    parser.add_argument(
        "--data-dir",
        default="processed_data",
        help="Directory containing train_labeled.json, dev_labeled.json, and test_labeled.json.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    has_error = False

    print("Fixed label vocabularies:")
    for task in TASKS:
        print(f"  {task}: {LABEL_VOCABS[task]}")
    print()

    merged_counters = {task: Counter() for task in TASKS}

    for split in DEFAULT_SPLITS:
        path = data_dir / f"{split}_labeled.json"
        if not path.exists():
            print(f"[WARN] Missing split file: {path}")
            has_error = True
            continue

        total, counters, unknowns, missing = inspect_split(path)
        print(f"=== {split} ({total} samples) ===")

        for task in TASKS:
            merged_counters[task].update(counters[task])
            print(f"[{task}]")
            print(format_counter(counters[task], total))

            if missing[task]:
                has_error = True
                print(f"    [MISSING] {missing[task]} samples do not contain '{task}'")

            if unknowns[task]:
                has_error = True
                print("    [UNKNOWN]")
                for label, count in unknowns[task].most_common():
                    print(f"      {label}: {count}")
        print()

    merged_total = sum(merged_counters[TASKS[0]].values())
    print(f"=== merged ({merged_total} samples) ===")
    for task in TASKS:
        print(f"[{task}]")
        print(format_counter(merged_counters[task], merged_total))
    print()

    if has_error:
        print("Label inspection finished with warnings. Please check missing or unknown labels above.")
    else:
        print("Label inspection passed. All labels are covered by LABEL_VOCABS.")


if __name__ == "__main__":
    main()
