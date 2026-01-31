#!/usr/bin/env python3
"""
Inference Configuration Example

This example demonstrates the InferenceRunner configuration
for entity extraction with various sampling strategies.
"""

from genner import InferenceRunner, Extractor, build_ner_prompt


def main():
    print("=== Example 1: Basic Inference Runner ===")
    # Create an inference runner with default settings
    runner = InferenceRunner()
    print(f"Batch size: {runner.batch_size}")
    print(f"Max tokens: {runner.max_tokens}")
    print(f"Temperature: {runner.temperature}")
    print(f"Use cache: {runner.use_cache}")
    print()

    print("=== Example 2: Custom Configuration ===")
    # Create a runner with custom settings
    runner = InferenceRunner(
        batch_size=16,
        max_tokens=1024,
        temperature=0.7,
        use_cache=True
    )
    print(f"Batch size: {runner.batch_size}")
    print(f"Max tokens: {runner.max_tokens}")
    print(f"Temperature: {runner.temperature}")
    print()

    print("=== Example 3: Updating Configuration ===")
    # Update settings after creation
    runner.set_batch_size(32)
    runner.set_temperature(0.5)
    runner.set_max_tokens(512)
    runner.set_use_cache(False)
    print(f"Updated batch size: {runner.batch_size}")
    print(f"Updated temperature: {runner.temperature}")
    print(f"Updated use cache: {runner.use_cache}")
    print()

    print("=== Example 4: Parse Marked Output ===")
    runner = InferenceRunner()
    marked_output = "@@John## works at @@Google## in @@Mountain View##."

    # Parse multiple entity types from marked text
    result = runner.parse_marked_output(marked_output, ["PER", "ORG", "LOC"])
    print(f"Marked output: {marked_output}")
    for entity_type, entities in result.items():
        print(f"  {entity_type}:")
        for entity in entities:
            print(f"    - {entity['text']} ({entity['start']}-{entity['end']})")
    print()

    print("=== Example 5: Temperature Clamping ===")
    runner = InferenceRunner()
    print(f"Initial temperature: {runner.temperature}")

    # Temperature is clamped to [0.0, 2.0]
    runner.set_temperature(5.0)
    print(f"After setting to 5.0: {runner.temperature} (clamped to 2.0)")

    runner.set_temperature(-1.0)
    print(f"After setting to -1.0: {runner.temperature} (clamped to 0.0)")
    print()

    print("=== Example 6: Statistics ===")
    runner = InferenceRunner()
    stats = runner.stats()
    print("Inference runner statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("=== Example 7: Building NER Prompts ===")
    # Build a prompt for NER extraction
    demonstrations = [
        ("Mary arrived", "@@Mary## arrived"),
        ("John left", "@@John## left"),
    ]
    prompt = build_ner_prompt("Sarah is here", "PER", demonstrations)
    print("NER Prompt:")
    print(prompt)
    print()

    print("=== Example 8: Simple Prompt Building ===")
    extractor = Extractor()
    simple_prompt = extractor.build_simple_prompt("PER", "Alice is here")
    print("Simple prompt:")
    print(simple_prompt)


if __name__ == "__main__":
    main()
