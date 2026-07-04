#!/usr/bin/env python
"""List Phase 1 experiments and their config/output locations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.experiments import list_phase1_experiments


def main():
    for exp in list_phase1_experiments():
        print(f"{exp.exp_id}")
        print(f"  title:  {exp.title}")
        print(f"  config: {exp.config_path}")
        print(f"  output: {exp.output_dir}")
        print(f"  status: {exp.status}")


if __name__ == "__main__":
    main()
