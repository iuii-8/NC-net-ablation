"""Generate predicted templates for train/dev/test datasets.

This script loads a trained AutoTemplate Predictor and generates predicted
template strings (<PT> ... </PT>) for each sample in the dataset.

Usage:
    python predict_template.py --model-path save_template_model/best_template_predictor.pt
    python predict_template.py --model-path save_template_model/best_template_predictor.pt --split test
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from label_config import TASKS, LABEL_VOCABS, ID_TO_LABEL, labels_to_pt_string
from template_dataset import TemplateDataset, TemplateCollator
from template_model import create_bert_predictor, create_lstm_predictor


def load_model(checkpoint_path, device="cpu"):
    """Load a trained AutoTemplate Predictor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    if args["model"] == "bert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args["bert_model"])
        model = create_bert_predictor(
            model_name=args["bert_model"],
            dropout=args.get("dropout", 0.1),
        )
    else:
        vocab_dir = Path(checkpoint_path).parent
        vocab_path = vocab_dir / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer = None
        model = create_lstm_predictor(
            vocab_size=len(vocab),
            dropout=args.get("dropout", 0.3),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, args, (vocab if args["model"] == "lstm" else None)


def predict_dataset(model, dataloader, device):
    """Run prediction on an entire dataset."""
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.to(device)

            logits = model(**kwargs)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                pred_ids = {}
                pred_labels = {}
                for task in TASKS:
                    pred_id = torch.argmax(logits[task][i]).item()
                    pred_ids[task] = pred_id
                    pred_labels[task] = ID_TO_LABEL[task][pred_id]

                all_predictions.append(pred_labels)

    return all_predictions


def compute_metrics(predictions, gold_labels):
    """Compute per-task accuracy."""
    task_correct = {task: 0 for task in TASKS}
    total = len(predictions)

    for pred, gold in zip(predictions, gold_labels):
        for task in TASKS:
            if pred[task] == gold[task]:
                task_correct[task] += 1

    accuracies = {task: task_correct[task] / total for task in TASKS}
    avg_accuracy = sum(accuracies.values()) / len(TASKS)
    return accuracies, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description="Generate predicted templates")
    parser.add_argument(
        "--model-path",
        type=str,
        default="save_template_model/best_template_predictor.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which data split to predict on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="processed_data",
        help="Directory containing labeled json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory to save prediction results",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer, model_args, vocab = load_model(args.model_path, args.device)
    print(f"Model type: {model_args['model']}")

    data_path = Path(args.data_dir) / f"{args.split}_labeled.json"
    print(f"Loading data from {data_path}...")

    max_length = model_args.get("max_length", 128)

    if model_args["model"] == "bert":
        dataset = TemplateDataset(data_path, tokenizer=tokenizer, max_length=max_length)
        collator = TemplateCollator()
    else:
        from train_template_predictor import LSTMCollator
        dataset = TemplateDataset(data_path, max_length=max_length)
        collator = LSTMCollator(vocab, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print(f"Predicting on {len(dataset)} samples...")
    predictions = predict_dataset(model, dataloader, args.device)

    gold_labels = [sample["labels"] for sample in dataset.samples]
    accuracies, avg_accuracy = compute_metrics(predictions, gold_labels)

    print(f"\nPrediction Results ({args.split}):")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    for task in TASKS:
        print(f"  {task}: {accuracies[task]:.4f}")

    print("\nBuilding output...")
    output_data = []
    for i, sample in enumerate(dataset.samples):
        pred_labels = predictions[i]
        pt_string = labels_to_pt_string(pred_labels)

        entry = {
            "tvBench_id": sample["tvBench_id"],
            "db_id": sample["db_id"],
            "query": sample["query"],
            "schema": sample["schema"],
            "gold_template": sample["gold_template"],
            "gold_labels": sample["labels"],
            "predicted_labels": pred_labels,
            "predicted_pt_template": pt_string,
        }
        output_data.append(entry)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_with_predicted_template.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(output_data)} predictions to {output_path}")

    metrics_path = output_dir / f"{args.split}_prediction_metrics.json"
    metrics = {
        "split": args.split,
        "total_samples": len(predictions),
        "avg_accuracy": avg_accuracy,
        "task_accuracies": accuracies,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
