#!/usr/bin/env python
"""Verify the versioned paper-linked tonsil receiver statistics."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tonsil_cxcl13_cxcr5"))

from reproduce_validation import compute_metrics, validate_metrics  # noqa: E402


def main() -> None:
    metrics = compute_metrics()
    validate_metrics(metrics)
    print("Verified versioned tonsil receiver statistics")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
