# GenNER: Generic Named Entity Recognition with SLM Fine-tuning

GenNER is a Rust-based library for training and deploying small language model (SLM) based Named Entity Recognition (NER) systems using the GPT-NER approach.

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](LICENSE)

## Features

- **Generation-based NER**: Uses the GPT-NER approach of transforming NER into a text generation task with `@@entity##` markers
- **SLM Support**: Designed for modern 2-7B decoder models (Qwen, Gemma, Phi, LLaMA)
- **LoRA Fine-tuning**: Parameter-efficient adaptation with LoRA/PEFT
- **Multi-task Learning**: Train multiple entity types with adapter composition
- **Incremental Learning**: Rehearsal buffers and adapter composition for continual learning
- **kNN Retrieval**: HNSW-based demonstration retrieval for few-shot inference
- **Python Bindings**: Full PyO3 integration for easy Python usage
- **Fast Inference**: Built with Rust and Candle for high performance

## Project Status

**Active Development** - Core implementation is largely complete. Full model forward passes await Candle transformer updates.

| Component | Status |
|-----------|--------|
| Core NER Pipeline | ✅ Complete |
| Training & LoRA | ✅ Complete |
| Inference Engine | ✅ Complete |
| kNN Retrieval | ✅ Complete |
| Python Bindings | ✅ Complete |
| Model Forward Pass | 🚧 Pending (candle-transformers) |
| Documentation | 🚧 In Progress |

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Shivamjohri247/GenNER.git
cd GenNER

# Install build dependencies
pip install maturin

# Build the Rust library (development mode)
maturin develop

# Or build and install the wheel
maturin build --release
pip install target/wheels/genner*.whl
```

### Requirements

- Rust 1.70+
- Python 3.9+
- maturin 1.0+

## Quick Start

### Basic Entity Marking and Parsing

```python
from genner import Extractor

# Create an extractor with default markers (@@ and ##)
extractor = Extractor()

# Mark entities in text
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = [
    {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
    {"text": "Steve Jobs", "label": "PER", "start": 28, "end": 38},
]
marked = extractor.mark_entities(text, entities)
print(marked)  # @@Apple Inc.## was founded by @@Steve Jobs## in Cupertino.

# Parse entities from marked text
marked_text = "@@John## works at @@Google##."
parsed = extractor.parse_entities(marked_text, "PER")
print(parsed)  # [{'text': 'John', 'label': 'PER', 'start': 0, 'end': 4, ...}]
```

### kNN Retrieval for Few-Shot Demonstrations

```python
from genner import Retriever
import numpy as np

# Create a retriever (768 dimensions for BERT-like embeddings)
retriever = Retriever(dimension=768)

# Add sentences with embeddings
embedding = np.random.randn(768).tolist()
retriever.add_sentence(
    text="John works at Google",
    embedding=embedding,
    entities=[{"text": "John", "label": "PER", "start": 0, "end": 4}]
)

# Build the HNSW index
retriever.build_index()

# Find k nearest neighbors
results = retriever.find_knn(k=3)
```

### Inference Configuration

```python
from genner import InferenceRunner

# Create an inference runner with custom settings
runner = InferenceRunner(
    batch_size=16,
    max_tokens=1024,
    temperature=0.7,
    use_cache=True
)

# Parse marked output from model
marked = "@@John## works at @@Google##."
result = runner.parse_marked_output(marked, ["PER", "ORG"])
```

### Building NER Prompts

```python
from genner import build_ner_prompt

# Build prompt with demonstrations
demonstrations = [
    ("Mary arrived", "@@Mary## arrived"),
    ("John left", "@@John## left"),
]
prompt = build_ner_prompt("Sarah is here", "PER", demonstrations)
print(prompt)
```

## Examples

See the `examples/` directory for more usage examples:

```bash
python examples/basic_ner.py       # Basic marking and parsing
python examples/retrieval.py       # kNN demonstration retrieval
python examples/inference.py       # Inference configuration
```

## Architecture

```
GenNER/
├── crates/
│   ├── core/       # Core library (traits, error handling, NER pipeline)
│   ├── models/     # Model implementations (Qwen, Gemma, Phi, LLaMA)
│   ├── inference/  # Inference engine (KV-cache, batching, streaming)
│   ├── python/     # PyO3 bindings
│   └── utils/      # Shared utilities
├── python/
│   ├── genner/     # Python package helpers
│   └── tests/      # Python tests
├── examples/       # Usage examples
└── data/
    ├── models/     # Downloaded SLM models
    ├── adapters/   # Trained LoRA adapters
    └── datasets/   # Training datasets
```

## GPT-NER Format

GenNER transforms NER into a generation task using special markers:

```
Input:  John went to Paris
Output: @@John## went to @@Paris##
```

The special tokens `@@` and `##` mark entity boundaries. This approach allows LLMs to perform NER through natural text generation rather than sequence labeling.

### Why Generation-based NER?

1. **Natural for LLMs**: Leverages the text generation capabilities of decoder-only models
2. **No Special Training Heads**: Works with any language model without classification layers
3. **Flexible Output**: Can handle overlapping and nested entities naturally
4. **Few-shot Ready**: Works well with in-context learning demonstrations

## Development

```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test -p genner-core
cargo test -p genner-inference

# Check code
cargo check --workspace

# Format code
cargo fmt

# Run linter
cargo clippy --workspace -- -D warnings
```

## Test Coverage

| Crate | Tests |
|-------|-------|
| genner-core | 100 |
| genner-inference | 40 |
| genner-models | 11 |
| genner-utils | 8 |
| **Total** | **159** |

## Data Format

GenNER expects training data in JSON format:

```json
{
  "dataset": "custom_ner",
  "entity_types": ["PER", "LOC", "ORG"],
  "samples": [
    {
      "text": "John Smith works at Google in New York.",
      "entities": [
        {"text": "John Smith", "label": "PER", "start": 0, "end": 10},
        {"text": "Google", "label": "ORG", "start": 20, "end": 26},
        {"text": "New York", "label": "LOC", "start": 30, "end": 38}
      ]
    }
  ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT OR Apache-2.0

## Citation

If you use GenNER in your research, please cite:

```bibtex
@software{genner2026,
  title={GenNER: Generic Named Entity Recognition with SLM Fine-tuning},
  author={Shivam Johri},
  year={2026},
  url={https://github.com/Shivamjohri247/GenNER}
}
```

## Acknowledgments

- Based on the [GPT-NER](https://arxiv.org/abs/2304.09920) paper
- Built with [Candle](https://github.com/huggingface/candle) ML framework
- Uses [PyO3](https://pyo3.rs/) for Python bindings
