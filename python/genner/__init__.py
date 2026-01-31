"""GenNER: Generic Named Entity Recognition with SLM Fine-tuning

This package provides Python bindings for the GenNER Rust library.
"""

from genner._genner import (
    # Core classes
    Model,
    Extractor,
    Trainer,
    LoRAConfig,
    TrainingConfig,
    InferenceRunner,

    # Exceptions
    GennerError,
)

__version__ = "0.1.0"

__all__ = [
    "Model",
    "Extractor",
    "Trainer",
    "LoRAConfig",
    "TrainingConfig",
    "InferenceRunner",
    "GennerError",
    "extract_entities",
    "load_dataset",
]


def extract_entities(
    text: str,
    entity_types: list[str],
    model_name: str = "Qwen/Qwen2-0.5B",
    **kwargs
) -> list[dict]:
    """Extract entities from text.

    Args:
        text: Input text to extract entities from
        entity_types: List of entity types to extract (e.g., ["PER", "LOC", "ORG"])
        model_name: Model name/path to use
        **kwargs: Additional arguments passed to Model

    Returns:
        List of extracted entities, each as a dict with keys:
        - text: the entity text
        - label: the entity type
        - start: start position in text
        - end: end position in text
        - confidence: confidence score (0-1)
    """
    # This is a simplified interface - the actual implementation
    # would use the Rust backend
    extractor = Extractor()
    model = Model(model_name, **kwargs)

    results = []
    for entity_type in entity_types:
        # In real implementation, this would call the model
        pass

    return results


def load_dataset(path: str) -> dict:
    """Load a training dataset from disk.

    Args:
        path: Path to the dataset file (.json or .data)

    Returns:
        Dataset dictionary with metadata and samples
    """
    import json

    with open(path) as f:
        return json.load(f)
