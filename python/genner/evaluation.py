"""Evaluation metrics for NER."""

from typing import Any
from collections import defaultdict


def compute_f1(predictions: list[dict], references: list[dict], label: str | None = None) -> dict[str, float]:
    """Compute F1 score for NER predictions.

    Args:
        predictions: Predicted entities
        references: Ground truth entities
        label: Optional label to filter by (e.g., "PER")

    Returns:
        Dictionary with precision, recall, and F1
    """
    if label:
        predictions = [e for e in predictions if e.get("label") == label]
        references = [e for e in references if e.get("label") == label]

    # Convert to sets for comparison
    pred_set = _entity_set(predictions)
    ref_set = _entity_set(references)

    true_positives = len(pred_set & ref_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _entity_set(entities: list[dict]) -> set[tuple]:
    """Convert entity list to set of tuples for comparison."""
    return {(e["text"], e["label"], e["start"], e["end"]) for e in entities}


def compute_f1_by_label(predictions: list[dict], references: list[dict]) -> dict[str, dict[str, float]]:
    """Compute F1 score per entity type.

    Args:
        predictions: Predicted entities
        references: Ground truth entities

    Returns:
        Dictionary mapping label to metrics
    """
    labels = set(e["label"] for e in references) | set(e["label"] for e in predictions)

    results = {}
    for label in labels:
        results[label] = compute_f1(predictions, references, label)

    return results


def compute_span_f1(predictions: list[dict], references: list[dict]) -> dict[str, float]:
    """Compute span-level F1 (ignoring labels)."""
    pred_spans = {(e["start"], e["end"]) for e in predictions}
    ref_spans = {(e["start"], e["end"]) for e in references}

    true_positives = len(pred_spans & ref_spans)
    false_positives = len(pred_spans - ref_spans)
    false_negatives = len(ref_spans - pred_spans)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
