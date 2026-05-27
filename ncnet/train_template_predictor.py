"""Training script for the AutoTemplate Predictor.

This script trains a multi-task classifier to predict template attributes
(mark, aggregate, filter, group, sort, bin) from natural language queries.

Usage:
    python train_template_predictor.py --model bert
    python train_template_predictor.py --model lstm --epochs 20
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from label_config import TASKS, LABEL_VOCABS, ID_TO_LABEL
from template_dataset import TemplateDataset, TemplateCollator
from template_model import (
    create_bert_predictor,
    create_lstm_predictor,
    count_parameters,
)


def get_tokenizer(model_name="bert-base-uncased"):
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(model_name)


def build_vocab_from_dataset(dataset, min_freq=1):
    """Build vocabulary from dataset for LSTM model."""
    from collections import Counter

    counter = Counter()
    for sample in dataset.samples:
        tokens = sample["text"].lower().split()
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokenize_for_lstm(text, vocab, max_length=128):
    """Simple whitespace tokenization for LSTM model."""
    tokens = text.lower().split()[:max_length]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    if len(ids) < max_length:
        ids = ids + [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


class LSTMCollator:
    """Collate function for LSTM model (no pretrained tokenizer)."""

    def __init__(self, vocab, max_length=128):
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = {task: [] for task in TASKS}

        for item in batch:
            ids = tokenize_for_lstm(item["text"], self.vocab, self.max_length)
            mask = [1 if i != self.vocab["<PAD>"] else 0 for i in ids]

            input_ids.append(ids)
            attention_masks.append(mask)

            for task in TASKS:
                labels[task].append(item["label_ids"][task])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": {
                task: torch.stack(labels[task]) for task in TASKS
            },
        }


def _model_forward(model, batch, device):
    """Unified forward call that works for both BERT and LSTM models."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids.to(device)

    return model(**kwargs)


def compute_accuracy(logits, labels):
    """Compute accuracy for each task."""
    accuracies = {}
    for task in TASKS:
        preds = torch.argmax(logits[task], dim=-1)
        correct = (preds == labels[task]).float().sum()
        total = labels[task].size(0)
        accuracies[task] = (correct / total).item()
    return accuracies


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    task_correct = {task: 0 for task in TASKS}
    task_total = {task: 0 for task in TASKS}

    with torch.no_grad():
        for batch in dataloader:
            labels = {task: batch["labels"][task].to(device) for task in TASKS}

            logits = _model_forward(model, batch, device)
            loss, _ = model.compute_loss(logits, labels)
            total_loss += loss.item()

            for task in TASKS:
                preds = torch.argmax(logits[task], dim=-1)
                task_correct[task] += (preds == labels[task]).sum().item()
                task_total[task] += labels[task].size(0)

    avg_loss = total_loss / len(dataloader)
    accuracies = {
        task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
        for task in TASKS
    }
    avg_accuracy = sum(accuracies.values()) / len(TASKS)

    return avg_loss, accuracies, avg_accuracy


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in TASKS}

    for batch in dataloader:
        labels = {task: batch["labels"][task].to(device) for task in TASKS}

        optimizer.zero_grad()
        logits = _model_forward(model, batch, device)
        loss, batch_task_losses = model.compute_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        for task in TASKS:
            task_losses[task] += batch_task_losses[task].item()

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_task_losses = {task: task_losses[task] / num_batches for task in TASKS}

    return avg_loss, avg_task_losses


def main():
    parser = argparse.ArgumentParser(description="Train AutoTemplate Predictor")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "lstm"],
        help="Model type: bert or lstm",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="processed_data/train_labeled.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--dev-data",
        type=str,
        default="processed_data/dev_labeled.json",
        help="Path to dev data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="save_template_model",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("AutoTemplate Predictor Training")
    print("=" * 60)
    print(f"Model type: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    if args.model == "bert":
        tokenizer = get_tokenizer(args.bert_model)
        train_dataset = TemplateDataset(
            args.train_data, tokenizer=tokenizer, max_length=args.max_length
        )
        dev_dataset = TemplateDataset(
            args.dev_data, tokenizer=tokenizer, max_length=args.max_length
        )
        collator = TemplateCollator()
    else:
        train_dataset = TemplateDataset(args.train_data, max_length=args.max_length)
        dev_dataset = TemplateDataset(args.dev_data, max_length=args.max_length)

        print("Building vocabulary...")
        vocab = build_vocab_from_dataset(train_dataset)
        print(f"Vocabulary size: {len(vocab)}")

        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

        collator = LSTMCollator(vocab, max_length=args.max_length)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print("\nCreating model...")
    if args.model == "bert":
        model = create_bert_predictor(
            model_name=args.bert_model,
            dropout=args.dropout,
        )
    else:
        model = create_lstm_predictor(
            vocab_size=len(vocab),
            dropout=args.dropout,
        )

    model = model.to(args.device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )

    print("\nStarting training...")
    print("-" * 60)

    best_dev_acc = 0.0
    best_epoch = 0
    training_log = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_task_losses = train_epoch(
            model, train_loader, optimizer, scheduler, args.device
        )

        dev_loss, dev_accuracies, dev_avg_acc = evaluate(model, dev_loader, args.device)

        epoch_time = time.time() - start_time

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_task_losses": train_task_losses,
            "dev_loss": dev_loss,
            "dev_accuracies": dev_accuracies,
            "dev_avg_accuracy": dev_avg_acc,
            "time": epoch_time,
        }
        training_log.append(log_entry)

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")
        print(f"  Dev Avg Accuracy: {dev_avg_acc:.4f}")
        print("  Dev Task Accuracies:")
        for task in TASKS:
            print(f"    {task}: {dev_accuracies[task]:.4f}")

        if dev_avg_acc > best_dev_acc:
            best_dev_acc = dev_avg_acc
            best_epoch = epoch

            best_model_path = output_dir / "best_template_predictor.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_accuracy": dev_avg_acc,
                    "dev_accuracies": dev_accuracies,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"  [NEW BEST] Saved to {best_model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Dev Accuracy: {best_dev_acc:.4f} (Epoch {best_epoch})")

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    print("\nFinal Dev Accuracies (Best Model):")
    best_log = training_log[best_epoch - 1]
    for task in TASKS:
        print(f"  {task}: {best_log['dev_accuracies'][task]:.4f}")


if __name__ == "__main__":
    main()
