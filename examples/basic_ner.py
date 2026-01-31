#!/usr/bin/env python3
"""
Basic NER Example

This example demonstrates how to use the GenNER library for
basic Named Entity Recognition using the entity marking format.
"""

from genner import Extractor, build_ner_prompt


def main():
    # Create an extractor with default markers (@@ and ##)
    extractor = Extractor()

    # Example 1: Mark entities in text
    print("=== Example 1: Marking Entities ===")
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = [
        {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
        {"text": "Steve Jobs", "label": "PER", "start": 28, "end": 38},
        {"text": "Cupertino", "label": "LOC", "start": 42, "end": 51},
        {"text": "California", "label": "LOC", "start": 53, "end": 63},
    ]

    marked = extractor.mark_entities(text, entities)
    print(f"Original: {text}")
    print(f"Marked:   {marked}")
    print()

    # Example 2: Parse entities from marked text
    print("=== Example 2: Parsing Entities ===")
    marked_text = "@@Apple Inc.## was founded by @@Steve Jobs## in @@Cupertino##, @@California##."
    parsed_entities = extractor.parse_entities(marked_text, "ORG")
    print(f"Marked text: {marked_text}")
    print(f"Parsed ORG entities: {parsed_entities}")
    print()

    # Example 3: Parse multiple entity types
    print("=== Example 3: Parsing Multiple Entity Types ===")
    result = extractor.parse_entities_multi(
        "@@John## works at @@Google## in @@London##.",
        ["PER", "ORG", "LOC"]
    )
    print(f"Marked text: @@John## works at @@Google## in @@London##.")
    for entity_type, entities in result.items():
        print(f"  {entity_type}: {entities}")
    print()

    # Example 4: Build NER prompts
    print("=== Example 4: Building Prompts ===")
    demonstrations = [
        ("Mary arrived yesterday", "@@Mary## arrived yesterday"),
        ("Dr. Smith is here", "@@Dr. Smith## is here"),
    ]
    prompt = build_ner_prompt("John went to Paris", "PER", demonstrations)
    print("Generated prompt:")
    print(prompt)
    print()

    # Example 5: Unmark text
    print("=== Example 5: Unmarking Text ===")
    marked = "@@John## went to @@Paris##."
    unmarked = extractor.unmark(marked)
    print(f"Marked:   {marked}")
    print(f"Unmarked: {unmarked}")


if __name__ == "__main__":
    main()
