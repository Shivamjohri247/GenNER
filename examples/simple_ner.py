#!/usr/bin/env python3
"""
Simple NER Example - Load Model and Extract Entities

This is the simplest way to use GenNER for entity extraction.
"""

import genner

print("=" * 60)
print("GenNER: Simple Entity Extraction Example")
print("=" * 60)
print()

# =============================================================================
# METHOD 1: Mark and Parse Entities (No model needed)
# =============================================================================

print("Method 1: Manual Entity Marking/Parsing")
print("-" * 40)

# Create an extractor
extractor = genner.Extractor()

# Example 1: Mark entities in text
text = "Elon Musk founded SpaceX in 2002"
entities = [
    {"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9},
    {"text": "SpaceX", "label": "ORG", "start": 19, "end": 25},
]

marked = extractor.mark_entities(text, entities)
print(f"Input:  {text}")
print(f"Marked: {marked}")
print()

# Example 2: Parse entities from marked text
marked_text = "@@Tesla## is located in @@Palo Alto##, @@California##."
parsed = extractor.parse_entities(marked_text, "ORG")
print(f"Marked: {marked_text}")
print(f"Parsed ORG entities:")
for entity in parsed:
    print(f"  - {entity['text']} at position {entity['start']}-{entity['end']}")
print()

# Example 3: Parse multiple entity types at once
result = extractor.parse_entities_multi(
    "@@Apple## was founded by @@Steve Jobs## and @@Steve Wozniak##.",
    ["PERSON", "ORG"]
)
for entity_type, entities in result.items():
    names = [e['text'] for e in entities]
    print(f"{entity_type}: {', '.join(names)}")
print()


# =============================================================================
# METHOD 2: Parse Model Output
# =============================================================================

print("Method 2: Parse Output from Language Model")
print("-" * 40)

# After your SLM generates text with entity markers:
model_output = "@@Amazon## headquarters is in @@Seattle##, @@Washington##."

# Parse all entity types
runner = genner.InferenceRunner()
all_entities = runner.parse_marked_output(
    model_output,
    ["ORG", "LOC", "GPE"]  # Organization, Location, Geo-Political Entity
)

print(f"Model output: {model_output}")
print("Extracted entities:")
for entity_type, entities in all_entities.items():
    if entities:
        print(f"  {entity_type}:")
        for e in entities:
            print(f"    - {e['text']} ({e['start']}-{e['end']})")
print()


# =============================================================================
# METHOD 3: Build NER Prompts (for training/inference)
# =============================================================================

print("Method 3: Build Prompts for Fine-Tuning")
print("-" * 40)

# Build a prompt with demonstrations
demonstrations = [
    ("Mary arrived", "@@Mary## arrived"),
    ("John left", "@@John## left"),
]

prompt = genner.build_ner_prompt(
    text="Sarah is here",
    entity_type="PERSON",
    demonstrations=demonstrations
)

print("Prompt for fine-tuning/inference:")
print(prompt)
print()


# =============================================================================
# METHOD 4: kNN Retrieval (for better few-shot examples)
# =============================================================================

print("Method 4: kNN Retrieval for Demonstrations")
print("-" * 40)

import numpy as np

retriever = genner.Retriever(dimension=768)

# Add some reference examples
examples = [
    ("Bill Gates founded Microsoft", [
        {"text": "Bill Gates", "label": "PERSON", "start": 0, "end": 9},
        {"text": "Microsoft", "label": "ORG", "start": 19, "end": 28}
    ]),
    ("Sundar Pichai leads Google", [
        {"text": "Sundar Pichai", "label": "PERSON", "start": 0, "end": 14},
        {"text": "Google", "label": "ORG", "start": 21, "end": 27}
    ]),
]

for text, entities in examples:
    # In production, use real embeddings from a model
    embedding = np.random.randn(768).tolist()
    retriever.add_sentence(text, embedding, entities)

retriever.build_index()

# Find similar examples
query_text = "Mark Zuckerberg runs Meta"
query_embedding = np.random.randn(768).tolist()
results = retriever.find_knn(2)

print(f"Query: {query_text}")
print("Most similar examples (for demonstrations):")
for i, (example, similarity) in enumerate(results, 1):
    print(f"  {i}. {example['text']} (similarity: {similarity:.3f})")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
GenNER provides:

1. Extractor - Mark/unmark/parse entities with @@entity## format
2. InferenceRunner - Configure and parse model outputs
3. Retriever - kNN search for demonstration retrieval
4. build_ner_prompt - Create prompts for fine-tuning

For fine-tuning SLMs on your domain:
1. Prepare training data with entities
2. Create prompts using build_ner_prompt()
3. Fine-tune with LoRA (parameter-efficient)
4. Save domain-specific adapter
5. Load adapter and extract entities!

The @@entity## format makes NER a text generation task
that any decoder-only model can learn.
""")
