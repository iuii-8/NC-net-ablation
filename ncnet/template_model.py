"""AutoTemplate Predictor: Multi-task classification model for template prediction.

This module implements a BERT-based multi-task classifier that predicts six
template attributes from natural language queries and schema information:
mark, aggregate, filter, group, sort, and bin.
"""

import torch
import torch.nn as nn

from label_config import TASKS, LABEL_VOCABS


class AutoTemplatePredictor(nn.Module):
    """Multi-task classifier for predicting chart template attributes.

    Architecture:
        BERT Encoder → [CLS] hidden state → 6 classification heads

    Each classification head predicts one template attribute:
        - mark: bar / line / point / arc
        - aggregate: count / none / mean / sum / min / max
        - filter: no / yes
        - group: no / yes
        - sort: none / asc / desc
        - bin: no / yes
    """

    def __init__(
        self,
        encoder,
        hidden_size=768,
        dropout=0.1,
        task_weights=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.classifiers = nn.ModuleDict()
        for task in TASKS:
            num_classes = len(LABEL_VOCABS[task])
            self.classifiers[task] = nn.Linear(hidden_size, num_classes)

        self.task_weights = task_weights or {task: 1.0 for task in TASKS}

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        logits = {}
        for task in TASKS:
            logits[task] = self.classifiers[task](cls_hidden)

        return logits

    def compute_loss(self, logits, labels, reduction="mean"):
        """Compute weighted multi-task cross-entropy loss."""
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        total_loss = 0.0
        task_losses = {}

        for task in TASKS:
            task_loss = criterion(logits[task], labels[task])
            task_losses[task] = task_loss
            total_loss += self.task_weights[task] * task_loss

        return total_loss, task_losses

    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """Get predicted class indices for each task."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            predictions = {
                task: torch.argmax(logits[task], dim=-1)
                for task in TASKS
            }
        return predictions


class AutoTemplatePredictorLSTM(nn.Module):
    """Lightweight LSTM-based multi-task classifier (no pretrained encoder).

    Use this as a faster baseline or when BERT is not available.

    Architecture:
        Embedding → BiLSTM → Attention Pooling → 6 classification heads
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        task_weights=None,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

        self.classifiers = nn.ModuleDict()
        for task in TASKS:
            num_classes = len(LABEL_VOCABS[task])
            self.classifiers[task] = nn.Linear(hidden_size * 2, num_classes)

        self.task_weights = task_weights or {task: 1.0 for task in TASKS}

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lstm_out = lstm_out * mask

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)

        logits = {}
        for task in TASKS:
            logits[task] = self.classifiers[task](context)

        return logits

    def compute_loss(self, logits, labels, reduction="mean"):
        """Compute weighted multi-task cross-entropy loss."""
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        total_loss = 0.0
        task_losses = {}

        for task in TASKS:
            task_loss = criterion(logits[task], labels[task])
            task_losses[task] = task_loss
            total_loss += self.task_weights[task] * task_loss

        return total_loss, task_losses

    def predict(self, input_ids, attention_mask=None, **kwargs):
        """Get predicted class indices for each task."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = {
                task: torch.argmax(logits[task], dim=-1)
                for task in TASKS
            }
        return predictions


def create_bert_predictor(
    model_name="bert-base-uncased",
    dropout=0.1,
    task_weights=None,
):
    """Create an AutoTemplatePredictor with a pretrained BERT encoder."""
    from transformers import BertModel

    encoder = BertModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size

    model = AutoTemplatePredictor(
        encoder=encoder,
        hidden_size=hidden_size,
        dropout=dropout,
        task_weights=task_weights,
    )
    return model


def create_lstm_predictor(
    vocab_size,
    embedding_dim=256,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    task_weights=None,
):
    """Create an AutoTemplatePredictorLSTM model."""
    model = AutoTemplatePredictorLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        task_weights=task_weights,
    )
    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing AutoTemplatePredictor (BERT-based)...")

    try:
        model = create_bert_predictor()
        print(f"Model created successfully.")
        print(f"Trainable parameters: {count_parameters(model):,}")

        batch_size = 2
        seq_len = 64
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        logits = model(dummy_input_ids, dummy_attention_mask)
        print("\nOutput logits shapes:")
        for task, task_logits in logits.items():
            print(f"  {task}: {task_logits.shape}")

        dummy_labels = {
            task: torch.randint(0, len(LABEL_VOCABS[task]), (batch_size,))
            for task in TASKS
        }
        total_loss, task_losses = model.compute_loss(logits, dummy_labels)
        print(f"\nTotal loss: {total_loss.item():.4f}")
        for task, loss in task_losses.items():
            print(f"  {task} loss: {loss.item():.4f}")

        predictions = model.predict(dummy_input_ids, dummy_attention_mask)
        print("\nPredictions:")
        for task, preds in predictions.items():
            print(f"  {task}: {preds.tolist()}")

        print("\nBERT model test passed!")

    except ImportError as e:
        print(f"Skipping BERT test (transformers not installed): {e}")

    print("\n" + "=" * 50)
    print("Testing AutoTemplatePredictorLSTM...")

    vocab_size = 10000
    model_lstm = create_lstm_predictor(vocab_size=vocab_size)
    print(f"LSTM model created successfully.")
    print(f"Trainable parameters: {count_parameters(model_lstm):,}")

    batch_size = 2
    seq_len = 64
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model_lstm(dummy_input_ids, dummy_attention_mask)
    print("\nOutput logits shapes:")
    for task, task_logits in logits.items():
        print(f"  {task}: {task_logits.shape}")

    dummy_labels = {
        task: torch.randint(0, len(LABEL_VOCABS[task]), (batch_size,))
        for task in TASKS
    }
    total_loss, task_losses = model_lstm.compute_loss(logits, dummy_labels)
    print(f"\nTotal loss: {total_loss.item():.4f}")

    predictions = model_lstm.predict(dummy_input_ids, dummy_attention_mask)
    print("\nPredictions:")
    for task, preds in predictions.items():
        print(f"  {task}: {preds.tolist()}")

    print("\nLSTM model test passed!")
