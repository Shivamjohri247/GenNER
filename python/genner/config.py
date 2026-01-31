"""Configuration classes for GenNER."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    checkpoint_every: int = 500
    output_dir: str = "outputs"


@dataclass
class PipelineConfig:
    """NER pipeline configuration."""

    entity_prefix: str = "@@"
    entity_suffix: str = "##"
    num_demonstrations: int = 4
    verification_enabled: bool = True
    verification_threshold: float = 0.5
    max_seq_len: int = 2048
    device: str = "cpu"
    dtype: str = "f32"


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str
    device: str = "cpu"
    dtype: str = "f32"
    max_seq_len: int = 2048
    use_cache: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
