# GenNER: Generic Named Entity Recognition with SLM Fine-tuning

GenNER is a Rust-based library for training and deploying small language model (SLM) based Named Entity Recognition (NER) systems using the GPT-NER approach.

## Features

- **Generation-based NER**: Uses the GPT-NER approach of transforming NER into a text generation task
- **SLM Support**: Works with modern 2-7B decoder models (Qwen, Gemma, Phi, LLaMA)
- **LoRA Fine-tuning**: Parameter-efficient adaptation with LoRA/PEFT
- **Multi-task Learning**: Train multiple entity types with adapter composition
- **Incremental Learning**: Rehearsal buffers and adapter composition for continual learning
- **kNN Retrieval**: Smart demonstration retrieval during inference
- **Python Bindings**: Full PyO3 integration for easy Python usage
- **Fast Inference**: Built with Rust and Candle for high performance

## Project Status

🚧 **Active Development** - Core implementation in progress. This project is under active development and not yet ready for production use.

## Installation

```bash
# Clone the repository
git clone https://github.com/genner/genner.git
cd genner

# Install Python dependencies
pip install maturin pyo3

# Build the Rust library (development mode)
maturin develop

# Or build in release mode
maturin build --release
pip install target/wheels/genner*.whl
```

## Quick Start

```python
import genner

# Extract entities from marked text
from genner import Extractor

extractor = Extractor(prefix="@@@", suffix="###")

# Parse marked text
marked = "@@John## went to @@Paris##"
entities = extractor.parse_entities(marked, "MIXED")

# Mark entities in text
text = "John went to Paris"
entity_dicts = [
    {"text": "John", "label": "PER", "start": 0, "end": 4},
    {"text": "Paris", "label": "LOC", "start": 14, "end": 19},
]
marked = extractor.mark_entities(text, entity_dicts)
```

## Architecture

```
GenNER/
├── crates/
│   ├── core/       # Core library (traits, error handling, NER pipeline, training)
│   ├── models/     # Model implementations (Qwen, Gemma, Phi, LLaMA)
│   ├── inference/  # Inference engine (KV-cache, batching, streaming)
│   ├── python/     # PyO3 bindings
│   └── utils/      # Shared utilities
├── python/
│   └── genner/     # Python package helpers
├── data/
│   ├── models/     # Downloaded SLM models
│   ├── adapters/   # Trained LoRA adapters
│   └── datasets/   # Training datasets
├── tests/          # Integration and unit tests
└── examples/       # Usage examples
```

## GPT-NER Format

GenNER transforms NER into a generation task:

```
Input:  John went to Paris
Output: @@John## went to @@Paris##
```

The special tokens `@@` and `##` mark entity boundaries. This approach allows LLMs to perform NER through natural text generation rather than sequence labeling.

## Development

```bash
# Run tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Check code
cargo check --workspace

# Format code
cargo fmt

# Run linter
cargo clippy --workspace -- -D warnings
```

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

## Architecture

```
GenNER/
├── crates/
│   ├── core/       # Core library (traits, error handling)
│   ├── models/     # Model implementations (Qwen, Gemma, etc.)
│   ├── inference/  # Inference engine
│   ├── python/     # PyO3 bindings
│   └── utils/      # Shared utilities
└── python/
    └── genner/     # Python package
```

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use GenNER in your research, please cite:

```bibtex
@software{genner2024,
  title={GenNER: Generic Named Entity Recognition with SLM Fine-tuning},
  author={Shivam Johri},
  year={2024},
  url={https://github.com/Shivamjohri247/GenNER}
}
```
