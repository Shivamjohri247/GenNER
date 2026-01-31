"""Utility functions for GenNER."""

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_string(s: str) -> str:
    """Hash a string to a stable identifier."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for consistent comparison."""
    return name.strip().lower()


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max length, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def count_tokens_estimate(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
