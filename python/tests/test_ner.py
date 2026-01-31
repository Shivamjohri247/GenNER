"""Python tests for GenNER"""

import pytest
from genner import Extractor, LoRAConfig, TrainingConfig


class TestExtractor:
    """Tests for the Extractor class."""

    def test_new(self):
        """Test creating a new extractor."""
        extractor = Extractor()
        assert extractor is not None

    def test_new_custom_markers(self):
        """Test creating extractor with custom markers."""
        extractor = Extractor(prefix="<<", suffix=">>")
        assert extractor is not None

    def test_mark_entities(self):
        """Test marking entities in text."""
        extractor = Extractor()

        entities = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 14, "end": 19},
        ]

        marked = extractor.mark_entities("John went to Paris", entities)
        assert "@@John@@" in marked
        assert "@@Paris@@" in marked

    def test_parse_entities(self):
        """Test parsing entities from marked text."""
        extractor = Extractor()
        entities = extractor.parse_entities("@@John## went to @@Paris##", "MIXED")
        assert len(entities) == 2

    def test_unmark(self):
        """Test unmarking text."""
        extractor = Extractor()
        unmarked = extractor.unmark("@@John## went to @@Paris##")
        assert unmarked == "John went to Paris"


class TestConfig:
    """Tests for configuration classes."""

    def test_lora_config(self):
        """Test LoRA config."""
        config = LoRAConfig(rank=8, alpha=16.0, dropout=0.1)
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1

    def test_lora_config_defaults(self):
        """Test LoRA config defaults."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32.0

    def test_training_config(self):
        """Test training config."""
        lora = LoRAConfig(rank=8)
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=16,
            num_epochs=5,
            lora=lora,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.num_epochs == 5

    def test_training_config_defaults(self):
        """Test training config defaults."""
        config = TrainingConfig()
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8
        assert config.num_epochs == 3


class TestData:
    """Tests for data utilities."""

    def test_create_training_sample(self):
        """Test creating a training sample."""
        from genner.data import create_training_sample

        sample = create_training_sample(
            text="John Smith works at Google",
            entities=[
                {"text": "John Smith", "label": "PER", "start": 0, "end": 10},
                {"text": "Google", "label": "ORG", "start": 20, "end": 26},
            ],
            entity_type="PER",
        )

        assert sample["input"] == "John Smith works at Google"
        assert "@@John Smith@@" in sample["output"]
        assert "Google" not in sample["output"] or "@@" not in sample["output"]

    def test_mark_entities(self):
        """Test entity marking."""
        from genner.data import _mark_entities

        entities = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 12, "end": 17},
        ]

        marked = _mark_entities("John went to Paris", entities)
        assert marked == "@@John## went to @@Paris##"


class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_compute_f1_perfect(self):
        """Test F1 with perfect predictions."""
        from genner.evaluation import compute_f1

        predictions = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 12, "end": 17},
        ]

        references = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 12, "end": 17},
        ]

        metrics = compute_f1(predictions, references)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_compute_f1_partial(self):
        """Test F1 with partial predictions."""
        from genner.evaluation import compute_f1

        predictions = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
        ]

        references = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 12, "end": 17},
        ]

        metrics = compute_f1(predictions, references)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
        assert metrics["f1"] == 2 * 1.0 * 0.5 / (1.0 + 0.5)

    def test_compute_f1_by_label(self):
        """Test per-label F1 computation."""
        from genner.evaluation import compute_f1_by_label

        predictions = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "Paris", "label": "LOC", "start": 12, "end": 17},
        ]

        references = [
            {"text": "John", "label": "PER", "start": 0, "end": 4},
            {"text": "London", "label": "LOC", "start": 20, "end": 26},
        ]

        metrics = compute_f1_by_label(predictions, references)
        assert "PER" in metrics
        assert "LOC" in metrics
        assert metrics["PER"]["f1"] == 1.0
        assert metrics["LOC"]["f1"] == 0.0
