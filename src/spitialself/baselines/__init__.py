"""Baseline adapters for external CCC methods.

Adapters will normalize inputs, outputs, and failure reporting for each method
so Phase 1 comparisons remain auditable and fair.
"""

from .cellnest_io import prepare_cellnest_inputs
from .commot_adapter import score_commot
from .deeptalk_io import prepare_deeptalk_inputs

__all__ = [
    "prepare_cellnest_inputs",
    "prepare_deeptalk_inputs",
    "score_commot",
]
