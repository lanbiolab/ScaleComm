"""Optional multi-subunit ligand/receptor expression aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def parse_subunits(value: object, fallback: str) -> list[str]:
    """Parse comma/semicolon/pipe-delimited subunits, falling back to one gene."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [fallback]
    text = str(value).strip()
    if not text:
        return [fallback]
    for delimiter in (";", "|", "+"):
        text = text.replace(delimiter, ",")
    genes = [item.strip() for item in text.split(",") if item.strip()]
    return genes or [fallback]


def geometric_mean_expression(values: np.ndarray, *, min_subunit_expression: float = 0.0) -> np.ndarray:
    """Aggregate subunit expression as exp(mean(log1p(x))) - 1 with a hard gate."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_events, n_subunits)")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    aggregate = np.expm1(np.log1p(np.clip(values, 0.0, None)).mean(axis=1))
    gate = np.all(values >= float(min_subunit_expression), axis=1)
    return np.where(gate, aggregate, 0.0)


def apply_complex_expression(
    *,
    expression: pd.DataFrame,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    min_subunit_expression: float = 0.0,
) -> pd.DataFrame:
    """Replace pairwise candidate expression with optional complex-aware values.

    `lr_pairs` may optionally supply `ligand_subunits` and `receptor_subunits`.
    Missing columns preserve the pairwise behavior exactly.  Candidate IDs and
    spatial metadata are untouched.
    """

    out = candidates.copy()
    expr = expression.set_index("cell_id")
    positions = pd.Series(np.arange(len(expr)), index=expr.index.astype(str))
    sender = out["sender_id"].astype(str).map(positions).to_numpy(dtype=int)
    receiver = out["receiver_id"].astype(str).map(positions).to_numpy(dtype=int)
    meta = lr_pairs.set_index("lr_id")
    ligand_values = np.zeros(len(out), dtype=float)
    receptor_values = np.zeros(len(out), dtype=float)
    for lr_id, idx in out.groupby("lr_id", sort=False).groups.items():
        row = meta.loc[lr_id]
        ligands = _available_subunits(expr.columns, parse_subunits(row.get("ligand_subunits"), str(row["ligand"])))
        receptors = _available_subunits(expr.columns, parse_subunits(row.get("receptor_subunits"), str(row["receptor"])))
        loc = np.asarray(list(idx), dtype=int)
        ligand_values[loc] = geometric_mean_expression(
            expr[ligands].to_numpy(dtype=float)[sender[loc]], min_subunit_expression=min_subunit_expression
        )
        receptor_values[loc] = geometric_mean_expression(
            expr[receptors].to_numpy(dtype=float)[receiver[loc]], min_subunit_expression=min_subunit_expression
        )
    out["ligand_expr"] = ligand_values
    out["receptor_expr"] = receptor_values
    return out


def _available_subunits(columns: Iterable[str], genes: list[str]) -> list[str]:
    available = [gene for gene in genes if gene in columns]
    if len(available) != len(genes):
        missing = sorted(set(genes) - set(available))
        raise ValueError(f"Complex subunits not present in expression table: {missing}")
    return available
