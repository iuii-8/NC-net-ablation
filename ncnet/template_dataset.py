import argparse
import json
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows basic data inspection even before installing torch.
    torch = None

    class Dataset:  # type: ignore
        pass

from label_config import TASKS, labels_to_ids


class TemplateDataset(Dataset):
    """Dataset for the AutoTemplate multi-task predictor.

    Each sample is built from the processed *_labeled.json files produced by
    data_parser.py. The model input is query + schema, and the targets are six
    template labels: mark, aggregate, filter, group, sort, and bin.
    """

    def __init__(self, data_path, tokenizer=None, max_length=128):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples(self.data_path)

    def _load_samples(self, data_path):
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for idx, item in enumerate(raw_data):
            query = str(item.get("query", "")).strip()
            schema = str(item.get("schema", "")).strip()
            labels = item.get("labels", {})

            sample = {
                "index": idx,
                "tvBench_id": str(item.get("tvBench_id", "")),
                "db_id": str(item.get("db_id", "")),
                "query": query,
                "schema": schema,
                "text": build_template_input_text(query, schema),
                "gold_template": str(item.get("gold_template", "")),
                "labels": labels,
                "label_ids": labels_to_ids(labels),
            }
            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        output = {
            "index": sample["index"],
            "tvBench_id": sample["tvBench_id"],
            "db_id": sample["db_id"],
            "query": sample["query"],
            "schema": sample["schema"],
            "text": sample["text"],
            "gold_template": sample["gold_template"],
            "labels": sample["labels"],
        }

        if torch is not None:
            output["label_ids"] = {
                task: torch.tensor(sample["label_ids"][task], dtype=torch.long)
                for task in TASKS
            }
        else:
            output["label_ids"] = sample["label_ids"]

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sample["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            for key, value in encoded.items():
                output[key] = value.squeeze(0)

        return output


class TemplateCollator:
    """Collate function for TemplateDataset.

    Use this when a tokenizer is provided to TemplateDataset. It stacks tensor
    fields and keeps metadata fields as lists for debugging or prediction export.
    """

    def __init__(self, tasks=None):
        self.tasks = tasks or TASKS

    def __call__(self, batch):
        if torch is None:
            raise ImportError("TemplateCollator requires torch to be installed.")

        output = {}
        tensor_keys = ["input_ids", "attention_mask", "token_type_ids"]

        for key in tensor_keys:
            if key in batch[0]:
                output[key] = torch.stack([item[key] for item in batch])

        output["labels"] = {
            task: torch.stack([item["label_ids"][task] for item in batch])
            for task in self.tasks
        }

        metadata_keys = [
            "index",
            "tvBench_id",
            "db_id",
            "query",
            "schema",
            "text",
            "gold_template",
        ]
        for key in metadata_keys:
            output[key] = [item[key] for item in batch]

        return output


def build_template_input_text(query, schema):
    """Build the text input consumed by the AutoTemplate predictor."""
    query = str(query).strip()
    schema = str(schema).strip()

    if schema:
        return f"[QUERY] {query} [SCHEMA] {schema}"
    return f"[QUERY] {query} [SCHEMA]"


def preview_dataset(data_path, limit=3):
    dataset = TemplateDataset(data_path)
    print(f"Loaded {len(dataset)} samples from {data_path}")
    print()

    for idx in range(min(limit, len(dataset))):
        item = dataset[idx]
        print(f"--- sample {idx} ---")
        print(f"text: {item['text']}")
        print(f"labels: {item['labels']}")
        print(f"label_ids: {item['label_ids']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Preview AutoTemplate predictor dataset samples."
    )
    parser.add_argument(
        "--data-path",
        default="processed_data/train_labeled.json",
        help="Path to a processed *_labeled.json file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of samples to preview.",
    )
    args = parser.parse_args()

    preview_dataset(args.data_path, args.limit)


if __name__ == "__main__":
    main()
