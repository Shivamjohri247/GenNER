"""Dataset utilities for GenNER."""

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset from a JSON file.

    Args:
        path: Path to the dataset file

    Returns:
        Dataset dictionary
    """
    with open(path) as f:
        return json.load(f)


def save_dataset(data: dict[str, Any], path: str | Path) -> None:
    """Save a dataset to a JSON file.

    Args:
        data: Dataset dictionary
        path: Path to save the dataset
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_training_sample(
    text: str,
    entities: list[dict],
    entity_type: str,
    task_description: str | None = None,
) -> dict[str, Any]:
    """Create a training sample in GPT-NER format.

    Args:
        text: Input text
        entities: List of entities with text, label, start, end
        entity_type: Target entity type for this sample
        task_description: Optional custom task description

    Returns:
        Training sample dictionary
    """
    if task_description is None:
        task_description = (
            f"I am an excellent linguist. "
            f"The task is to label {entity_type} entities in the given sentence. "
            "Below are some examples."
        )

    # Filter entities of the target type
    target_entities = [e for e in entities if e["label"] == entity_type]

    # Mark entities in text
    marked_output = _mark_entities(text, target_entities)

    return {
        "task_description": task_description,
        "input": text,
        "output": marked_output,
        "entities": target_entities,
        "entity_type": entity_type,
    }


def _mark_entities(text: str, entities: list[dict], prefix: str = "@@", suffix: str = "##") -> str:
    """Mark entities in text with special tokens."""
    if not entities:
        return text

    # Sort by start position (descending to avoid offset issues)
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    result = text
    for entity in sorted_entities:
        start, end = entity["start"], entity["end"]
        marked = f"{prefix}{text[start:end]}{suffix}"
        result = result[:start] + marked + result[end:]

    return result
