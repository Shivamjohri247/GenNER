#!/usr/bin/env python3
"""
Complete NER Pipeline Example

This demonstrates the full GenNER workflow:
1. Load an SLM (Small Language Model)
2. Fine-tune it on domain-specific NER data
3. Extract entities from new text
4. Use multiple domain adapters

This is the GenNER approach - combining GPT-NER format with SLM fine-tuning.
"""

import genner
import json

# =============================================================================
# PART 1: Understanding the GPT-NER Format
# =============================================================================

print("=" * 70)
print("PART 1: GPT-NER Entity Marking Format")
print("=" * 70)

# The core idea: transform NER into a text generation task
# Instead of: John -> B-PER, went -> O, to -> O, Paris -> B-LOC
# We generate:  @@John## went to @@Paris##

extractor = genner.Extractor()

# Example input text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

# Example entities (what we want to extract)
entities = [
    {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
    {"text": "Steve Jobs", "label": "PER", "start": 28, "end": 38},
    {"text": "Cupertino", "label": "LOC", "start": 42, "end": 51},
    {"text": "California", "label": "LOC", "start": 53, "end": 63},
]

# Mark entities in the GPT-NER format
marked_text = extractor.mark_entities(text, entities)
print(f"Original: {text}")
print(f"Marked:   {marked_text}")
print()

# Parse entities back from marked text
parsed = extractor.parse_entities(marked_text, "ORG")
print(f"Parsed ORG entities: {parsed}")
print()


# =============================================================================
# PART 2: Building NER Training Prompts
# =============================================================================

print("=" * 70)
print("PART 2: Building Training Prompts")
print("=" * 70)

# For fine-tuning, we need to create training examples
# Each example converts: input_text + entities -> marked_output

def create_training_example(text, entities, entity_type):
    """Create a training example in GPT-NER format"""
    extractor = genner.Extractor()
    marked = extractor.mark_entities(text, entities)

    # Build the prompt with task description
    prompt = extractor.build_simple_prompt(entity_type, text)

    return {
        "prompt": prompt,
        "completion": marked,  # What the model should generate
        "entities": entities,
        "entity_type": entity_type
    }

# Example training data for a medical domain
medical_examples = [
    {
        "text": "The patient was prescribed aspirin for chest pain.",
        "entities": [
            {"text": "aspirin", "label": "DRUG", "start": 28, "end": 35},
            {"text": "chest pain", "label": "SYMPTOM", "start": 39, "end": 49},
        ]
    },
    {
        "text": "Dr. Smith diagnosed the patient with diabetes type 2.",
        "entities": [
            {"text": "Dr. Smith", "label": "DOCTOR", "start": 0, "end": 9},
            {"text": "diabetes", "label": "DISEASE", "start": 35, "end": 44},
            {"text": "type 2", "label": "CONDITION", "start": 45, "end": 51},
        ]
    },
]

# Create training examples
for ex in medical_examples:
    training_ex = create_training_example(
        ex["text"],
        ex["entities"],
        "DRUG"  # Fine-tune for DRUG entity type
    )
    print(f"Prompt: {training_ex['prompt'][:100]}...")
    print(f"Target: {training_ex['completion']}")
    print()

print()


# =============================================================================
# PART 3: kNN Retrieval for Few-Shot Demonstrations
# =============================================================================

print("=" * 70)
print("PART 3: kNN Retrieval for Better Demonstrations")
print("=" * 70)

import numpy as np

# The key insight from the GPT-NER paper:
# Better demonstrations = better performance
# We use kNN to find similar training examples

retriever = genner.Retriever(dimension=768)  # BERT-like embedding dimension

# Index our training data with embeddings
# (In production, you'd use a real embedding model)
training_data = [
    {"text": "John works at Microsoft in Seattle.", "entities": [{"text": "John", "label": "PER"}, {"text": "Microsoft", "label": "ORG"}]},
    {"text": "Mary visited Google in Mountain View.", "entities": [{"text": "Mary", "label": "PER"}, {"text": "Google", "label": "ORG"}]},
    {"text": "Bob joined Amazon in New York.", "entities": [{"text": "Bob", "label": "PER"}, {"text": "Amazon", "label": "ORG"}]},
]

for item in training_data:
    # Use random embeddings as placeholder
    embedding = np.random.randn(768).tolist()

    # Convert entities to genner format
    entities = []
    for i, entity in enumerate(item['entities']):
        entities.append({
            "text": entity['text'],
            "label": entity['label'],
            "start": 0,  # Placeholder
            "end": len(entity['text']),
        })

    retriever.add_sentence(item['text'], embedding, entities)

retriever.build_index()
print(f"Indexed {retriever.len_sentences()} training examples")
print()

# For a new input, find similar examples
query_text = "Alice started working at Apple"
query_embedding = np.random.randn(768).tolist()

# Find k=2 most similar examples
results = retriever.find_knn(2)
print(f"Input: {query_text}")
print(f"Top 2 similar examples for demonstrations:")
for i, (sent, sim) in enumerate(results, 1):
    print(f"  {i}. {sent['text']} (similarity: {sim:.4f})")
print()


# =============================================================================
# PART 4: Complete Training Workflow
# =============================================================================

print("=" * 70)
print("PART 4: Domain-Specific Fine-Tuning Workflow")
print("=" * 70)

print("""
Step-by-step training workflow for a new domain:

1. PREPARE DATA:
   ```python
   domain_data = {
       "domain": "medical",
       "entity_types": ["DRUG", "DISEASE", "SYMPTOM"],
       "samples": [
           {
               "text": "Patient given penicillin for infection.",
               "entities": [
                   {"text": "penicillin", "label": "DRUG", "start": 13, "end": 23},
                   {"text": "infection", "label": "DISEASE", "start": 27, "end": 36},
               ]
           },
           # ... more samples
       ]
   }
   ```

2. CREATE TRAINING EXAMPLES:
   ```python
   examples = []
   for sample in domain_data["samples"]:
       for entity_type in domain_data["entity_types"]:
           example = create_training_example(
               sample["text"],
               [e for e in sample["entities"] if e["label"] == entity_type],
               entity_type
           )
           examples.append(example)
   ```

3. FINE-TUNE WITH LoRA:
   ```python
   # Configure LoRA for parameter-efficient fine-tuning
   lora_config = genner.LoRAConfig(
       rank=16,           # Adapter rank (higher = more parameters)
       alpha=32,          # Scaling factor
       dropout=0.05,     # Dropout for regularization
       target_modules=["q_proj", "v_proj"]  # Which layers to adapt
   )

   # Configure training
   training_config = genner.TrainingConfig(
       learning_rate=5e-5,
       batch_size=8,
       num_epochs=3,
       lora=lora_config
   )

   # Train the domain adapter
   trainer = genner.Trainer(
       model_path="Qwen/Qwen2-0.5B",  # Base SLM
       config=training_config
   )

   adapter = trainer.train(
       train_data=examples,
       val_data=val_examples
   )

   # Save the domain-specific adapter
   adapter.save("adapters/medical_ner.safetensors")
   ```

4. MULTI-DOMAIN SUPPORT:
   ```python
   # Train multiple domain adapters
   domains = ["medical", "legal", "financial"]

   adapters = {}
   for domain in domains:
       adapter = trainer.train(f"{domain}_train.json")
       adapters[domain] = adapter

   # Load base model once, switch adapters as needed
   model = genner.Model.load("Qwen/Qwen2-0.5B")

   # Switch to medical domain
   model.apply_adapter(adapters["medical"])
   entities = model.extract("Patient took aspirin...", ["DRUG", "DISEASE"])

   # Switch to legal domain
   model.apply_adapter(adapters["legal"])
   entities = model.extract("The contract was signed...", ["LAW", "ORG"])
   ```
""")

print()


# =============================================================================
# PART 5: Inference Configuration
# =============================================================================

print("=" * 70)
print("PART 5: Inference Configuration")
print("=" * 70)

# Configure inference parameters
runner = genner.InferenceRunner(
    batch_size=16,      # Process multiple texts at once
    max_tokens=512,     # Maximum tokens to generate
    temperature=0.0,    # 0.0 = greedy (best for factual extraction)
    use_cache=True      # Use KV-cache for efficiency
)

# Parse marked output from model
marked_output = "@@Apple Inc.## was founded by @@Steve Jobs## in @@Cupertino##."
result = runner.parse_marked_output(marked_output, ["ORG", "PER", "LOC"])

print("Marked output from model:")
print(f"  {marked_output}")
print()
print("Parsed entities:")
for entity_type, entities in result.items():
    print(f"  {entity_type}:")
    for entity in entities:
        print(f"    - {entity['text']} ({entity['start']}-{entity['end']})")
print()


# =============================================================================
# PART 6: Summary of GenNER Architecture
# =============================================================================

print("=" * 70)
print("SUMMARY: GenNER Architecture for Domain-Specific NER")
print("=" * 70)

print("""
GenNER combines the best of both worlds:

┌─────────────────────────────────────────────────────────────────┐
│                     GenNER Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌─────────────┐     ┌──────────────────┐    │
│   │   Base SLM    │────▶│  LoRA Layer  │────▶│ Domain Adapter  │    │
│   │  (Qwen/Gemma) │     │  (trainable) │     │  (per domain)   │    │
│   └──────────────┘     └─────────────┘     └──────────────────┘    │
│         │                        │                        │            │
│         ▼                        ▼                        ▼            │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              GPT-NER Format: @@entity##                      │  │
│   ├──────────────────────────────────────────────────────────────┤  │
│   │  • Mark entities with special tokens                             │  │
│   │  • Transform NER → text generation                              │  │
│   │  • Natural for decoder-only models                             │  │
│   └──────────────────────────────────────────────────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    Training Pipeline                           │  │
│   ├──────────────────────────────────────────────────────────────┤  │
│   │  1. Prepare domain data (text + entities)                        │  │
│   │  2. Create prompts with task description                         │  │
│   │  3. Fine-tune with LoRA (parameter-efficient)                   │  │
│   │  4. Save domain-specific adapter                                 │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    Inference Pipeline                          │  │
│   ├──────────────────────────────────────────────────────────────┤  │
│   │  1. Load base SLM + domain adapter                            │  │
│   │  2. Build prompt (optionally with kNN demonstrations)           │  │
│   │  3. Generate marked output: @@entity##                         │  │
│   │  4. Parse entities from marked text                             │  │
│   │  5. (Optional) Self-verify to reduce hallucinations           │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────┘

Advantages over pure GPT-NER (paper approach):
  ✓ No API costs - run locally
  ✓ Faster inference - smaller models
  ✓ Better domain adaptation - fine-tuned adapters
  ✓ Privacy - data stays local
  ✓ Multi-domain - switch adapters as needed

Advantages over traditional NER (BIO tagging):
  ✓ Natural for LLMs - generation task
  ✓ Handles nested entities naturally
  ✓ No special classification head needed
  ✓ Works with any decoder-only model
""")

print()


# =============================================================================
# PART 7: Quick Reference
# =============================================================================

print("=" * 70)
print("QUICK REFERENCE")
print("=" * 70)

print("""
KEY COMPONENTS:

1. Extractor - Mark and parse entities
   extractor = genner.Extractor()
   marked = extractor.mark_entities(text, entities)
   entities = extractor.parse_entities(marked, "ORG")

2. Retriever - kNN demonstration retrieval
   retriever = genner.Retriever(dimension=768)
   retriever.add_sentence(text, embedding, entities)
   retriever.build_index()
   results = retriever.find_knn(k)

3. InferenceRunner - Configure extraction
   runner = genner.InferenceRunner(batch_size=16, temperature=0.0)
   result = runner.parse_marked_output(marked_text, ["ORG", "PER"])

4. Trainer - Fine-tune on domain data
   config = genner.TrainingConfig(learning_rate=5e-5, num_epochs=3)
   lora = genner.LoRAConfig(rank=16, alpha=32)
   adapter = trainer.train(data_path, config)

ENTITY MARKING FORMAT:
   Original: "Apple Inc. was founded by Steve Jobs."
   Marked:   "@@Apple Inc.## was founded by @@Steve Jobs##."

TRAINING DATA FORMAT:
   {
       "text": "John works at Google.",
       "entities": [
           {"text": "John", "label": "PER", "start": 0, "end": 4},
           {"text": "Google", "label": "ORG", "start": 13, "end": 19}
       ]
   }
""")
