#!/usr/bin/env python3
"""
kNN Retrieval Example

This example demonstrates how to use the Retriever for few-shot
demonstration retrieval using HNSW indexing.
"""

import numpy as np
from genner import Retriever


def main():
    # Create a retriever with 768-dimensional embeddings (common for BERT-like models)
    retriever = Retriever(dimension=768)

    print("=== Example 1: Adding Sentences ===")

    # Add sentences with embeddings (using random embeddings for demonstration)
    sentences = [
        ("John works at Google in New York.", ["ORG", "LOC"]),
        ("Mary visited Paris last summer.", ["LOC"]),
        ("Apple Inc. is based in Cupertino.", ["ORG", "LOC"]),
        ("Dr. Smith published a paper on AI.", ["PER"]),
    ]

    for text, entity_types in sentences:
        # In production, you would use a real embedding model
        embedding = np.random.randn(768).tolist()

        # Create entities for demonstration
        entities = []
        if "ORG" in entity_types:
            entities.append({"text": "Google", "label": "ORG", "start": 13, "end": 19})
        if "LOC" in entity_types:
            entities.append({"text": "New York", "label": "LOC", "start": 23, "end": 31})

        retriever.add_sentence(text, embedding, entities)
        print(f"Added: {text}")

    print(f"\nTotal sentences: {retriever.len_sentences}")
    print(f"Total entities: {retriever.len_entities}")
    print()

    print("=== Example 2: Building Index ===")
    retriever.build_index()
    print("HNSW index built successfully")
    print()

    print("=== Example 3: Finding kNN ===")
    # Find k=3 nearest neighbors
    query_embedding = np.random.randn(768).tolist()
    results = retriever.find_knn(3)

    print(f"Top 3 nearest neighbors:")
    for i, (sentence, similarity) in enumerate(results, 1):
        print(f"  {i}. {sentence['text']} (similarity: {similarity:.4f})")
    print()

    print("=== Example 4: Saving and Loading ===")
    # Save index to disk
    retriever.save("/tmp/genner_index.bin")
    print("Index saved to /tmp/genner_index.bin")

    # Load into a new retriever
    new_retriever = Retriever(dimension=768)
    new_retriever.load("/tmp/genner_index.bin")
    print(f"Loaded index with {new_retriever.len_sentences} sentences")
    print()

    print("=== Example 5: Batch Adding ===")
    # Add multiple sentences at once
    batch_data = [
        {
            "text": "Elon Musk founded SpaceX.",
            "embedding": np.random.randn(768).tolist(),
            "entities": [
                {"text": "Elon Musk", "label": "PER", "start": 0, "end": 9},
                {"text": "SpaceX", "label": "ORG", "start": 19, "end": 25},
            ]
        },
        {
            "text": "The Eiffel Tower is in Paris.",
            "embedding": np.random.randn(768).tolist(),
            "entities": [
                {"text": "Eiffel Tower", "label": "LOC", "start": 4, "end": 16},
                {"text": "Paris", "label": "LOC", "start": 23, "end": 28},
            ]
        },
    ]

    count = retriever.add_sentences_batch(batch_data)
    print(f"Added {count} sentences in batch")
    print(f"Total sentences: {retriever.len_sentences}")


if __name__ == "__main__":
    main()
