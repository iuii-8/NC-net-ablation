"""Shared label vocabularies for the AutoTemplate predictor.

This file fixes the label space used by training, evaluation, and prediction.
Keep these vocabularies stable once the template predictor starts training.
"""

TASKS = ["mark", "aggregate", "filter", "group", "sort", "bin"]

LABEL_VOCABS = {
    "mark": ["bar", "line", "point", "arc"],
    "aggregate": ["count", "none", "mean", "sum", "min", "max"],
    "filter": ["no", "yes"],
    "group": ["no", "yes"],
    "sort": ["none", "asc", "desc"],
    "bin": ["no", "yes"],
}

LABEL_ALIASES = {
    "mark": {
        "pie": "arc",
        "scatter": "point",
    },
    "aggregate": {
        "avg": "mean",
        "average": "mean",
    },
    "sort": {
        "ascending": "asc",
        "descending": "desc",
    },
}

LABEL_TO_ID = {
    task: {label: idx for idx, label in enumerate(labels)}
    for task, labels in LABEL_VOCABS.items()
}

ID_TO_LABEL = {
    task: {idx: label for label, idx in label_to_id.items()}
    for task, label_to_id in LABEL_TO_ID.items()
}


def normalize_label(task, label):
    """Normalize label aliases into the fixed training label space."""
    value = str(label).strip().lower()
    return LABEL_ALIASES.get(task, {}).get(value, value)


def labels_to_ids(labels):
    """Convert a label dictionary into task-specific integer ids."""
    label_ids = {}
    for task in TASKS:
        label = normalize_label(task, labels.get(task, "none"))
        if label not in LABEL_TO_ID[task]:
            raise ValueError(f"Unknown label for task '{task}': {label}")
        label_ids[task] = LABEL_TO_ID[task][label]
    return label_ids


def ids_to_labels(label_ids):
    """Convert task-specific integer ids back into labels."""
    return {
        task: ID_TO_LABEL[task][int(label_id)]
        for task, label_id in label_ids.items()
    }


def labels_to_pt_string(labels):
    """Convert predicted labels into the standard predicted-template string."""
    normalized = {
        task: normalize_label(task, labels.get(task, "none"))
        for task in TASKS
    }
    return (
        f"<PT> mark {normalized['mark']} "
        f"aggregate {normalized['aggregate']} "
        f"group {normalized['group']} "
        f"filter {normalized['filter']} "
        f"sort {normalized['sort']} "
        f"bin {normalized['bin']} </PT>"
    )
