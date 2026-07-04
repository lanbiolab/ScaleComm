#!/usr/bin/env python
"""Edge-level directionality permutation audit for Phase 12.

The existing Phase 12 directionality permutation audit uses aggregated
niche-flow tables. This script moves one level closer to the spatial edges:

1. receiver-niche/coordinate pairing permutation: shuffle receiver niche labels
   within each dataset/method/LR edge table and recompute the expected-edge
   score fraction from the observed expected niche-pair rule;
2. edge-vector permutation: shuffle unit direction vectors across edges and
   recompute the expected-edge vector resultant length.

This is still not model retraining under permuted coordinates. It is an
edge-level null audit on fixed model/baseline scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_ROOT = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/directionality_edge_permutation"
N_PERMUTATIONS = 200
RNG_SEED = 12013


EDGE_INPUTS = {
    "lymph_node_full": {
        "dataset": "Native human lymph node Visium",
        "path": PHASE11_ROOT / "human_lymph_node_visium_full_subset/results/lymph_node_full_directionality_edges.csv",
    },
    "lymph_node_expanded": {
        "dataset": "Lymph node expanded benchmark",
        "path": PHASE11_ROOT / "lymph_node_phase6_expanded/results/lymph_node_directionality_edges.csv",
    },
    "visium_hd_crc": {
        "dataset": "Visium HD CRC",
        "path": PHASE11_ROOT / "visium_hd_crc_p2crc_subset/results/directionality_edges.csv",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_key, spec in EDGE_INPUTS.items():
        path = Path(spec["path"])
        if not path.exists():
            rows.append(missing_row(dataset_key, str(spec["dataset"]), path))
            continue
        edges = pd.read_csv(path)
        rows.extend(run_dataset(dataset_key, str(spec["dataset"]), edges))

    lr_summary = pd.DataFrame(rows)
    if not lr_summary.empty:
        lr_summary["receiver_pairing_bh_fdr"] = bh_by_group(
            lr_summary, ["dataset_key", "method"], "receiver_pairing_p_value"
        )
        lr_summary["edge_vector_bh_fdr"] = bh_by_group(lr_summary, ["dataset_key", "method"], "edge_vector_p_value")
        lr_summary["edge_permutation_call"] = [classify_lr(row) for row in lr_summary.itertuples(index=False)]
    dataset_summary = summarize_dataset(lr_summary)
    method_comparison = summarize_method_comparison(lr_summary)

    lr_summary.to_csv(OUT_DIR / "phase12_directionality_edge_permutation_lr_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_directionality_edge_permutation_dataset_summary.csv", index=False)
    method_comparison.to_csv(OUT_DIR / "phase12_directionality_edge_permutation_method_comparison.csv", index=False)
    write_status(dataset_summary, method_comparison)
    write_manifest()

    print(dataset_summary.to_string(index=False))
    print()
    print(method_comparison.to_string(index=False))


def run_dataset(dataset_key: str, dataset: str, edges: pd.DataFrame) -> list[dict[str, object]]:
    required = [
        "method",
        "lr_pair",
        "score",
        "sender_niche",
        "receiver_niche",
        "niche_rule_satisfied",
        "unit_dx",
        "unit_dy",
        "false_positive_high_expression",
    ]
    missing = [col for col in required if col not in edges.columns]
    if missing:
        return [
            {
                "dataset_key": dataset_key,
                "dataset": dataset,
                "method": "all",
                "lr_pair": "all",
                "status": "missing_columns",
                "missing_columns": ";".join(missing),
            }
        ]
    rng = np.random.default_rng(RNG_SEED + stable_offset(dataset_key))
    rows = []
    for (method, lr_pair), sub in edges.groupby(["method", "lr_pair"], sort=False):
        rows.append(metric_row(dataset_key, dataset, str(method), str(lr_pair), sub, rng))
    return rows


def metric_row(
    dataset_key: str,
    dataset: str,
    method: str,
    lr_pair: str,
    sub: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    score = pd.to_numeric(sub["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total_score = float(score.sum())
    expected = sub["niche_rule_satisfied"].astype(bool).to_numpy()
    sender_code, receiver_code, expected_matrix = encode_expected_pairs(sub, expected)
    unit_dx = pd.to_numeric(sub["unit_dx"], errors="coerce").fillna(0).to_numpy(dtype=float)
    unit_dy = pd.to_numeric(sub["unit_dy"], errors="coerce").fillna(0).to_numpy(dtype=float)

    observed_fraction = expected_score_fraction(score, expected)
    observed_vector = expected_vector_resultant(score, unit_dx, unit_dy, expected)
    receiver_null = np.empty(N_PERMUTATIONS, dtype=float)
    vector_null = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        perm_receiver = rng.permutation(receiver_code)
        perm_expected = expected_matrix[sender_code, perm_receiver]
        receiver_null[i] = expected_score_fraction(score, perm_expected)
        perm_vec = rng.permutation(len(score))
        vector_null[i] = expected_vector_resultant(score, unit_dx[perm_vec], unit_dy[perm_vec], expected)

    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "method": method,
        "lr_pair": lr_pair,
        "status": "completed",
        "n_edges": int(len(sub)),
        "n_expected_edges": int(expected.sum()),
        "high_expression_negative": bool(sub["false_positive_high_expression"].astype(bool).any()),
        "total_score": total_score,
        "observed_expected_edge_score_fraction": observed_fraction,
        "receiver_pairing_null_mean": nanmean(receiver_null),
        "receiver_pairing_null_q95": nanquantile(receiver_null, 0.95),
        "receiver_pairing_p_value": right_tail_p(observed_fraction, receiver_null),
        "observed_expected_vector_resultant": observed_vector,
        "edge_vector_null_mean": nanmean(vector_null),
        "edge_vector_null_q95": nanquantile(vector_null, 0.95),
        "edge_vector_p_value": right_tail_p(observed_vector, vector_null),
    }


def encode_expected_pairs(sub: pd.DataFrame, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sender_codes, sender_levels = pd.factorize(sub["sender_niche"].astype(str), sort=True)
    receiver_codes, receiver_levels = pd.factorize(sub["receiver_niche"].astype(str), sort=True)
    matrix = np.zeros((len(sender_levels), len(receiver_levels)), dtype=bool)
    if expected.any():
        matrix[sender_codes[expected], receiver_codes[expected]] = True
    return sender_codes.astype(int), receiver_codes.astype(int), matrix


def expected_score_fraction(score: np.ndarray, expected: np.ndarray) -> float:
    total = float(np.nansum(score))
    if total <= 0 or len(score) == 0:
        return np.nan
    return float(np.nansum(score[expected]) / total)


def expected_vector_resultant(score: np.ndarray, unit_dx: np.ndarray, unit_dy: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.nansum(score[expected]))
    if denom <= 0 or len(score) == 0 or not expected.any():
        return np.nan
    vx = float(np.nansum(score[expected] * unit_dx[expected]))
    vy = float(np.nansum(score[expected] * unit_dy[expected]))
    return float(np.sqrt(vx * vx + vy * vy) / denom)


def summarize_dataset(lr_summary: pd.DataFrame) -> pd.DataFrame:
    if lr_summary.empty:
        return pd.DataFrame()
    rows = []
    complete = lr_summary[lr_summary["status"].eq("completed")].copy()
    for (dataset_key, method), sub in complete.groupby(["dataset_key", "method"], sort=False):
        supported = sub["edge_permutation_call"].isin(["receiver_pairing_supported", "receiver_and_vector_supported"])
        both = sub["edge_permutation_call"].eq("receiver_and_vector_supported")
        best = sub.sort_values(["receiver_pairing_bh_fdr", "observed_expected_edge_score_fraction"], ascending=[True, False]).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "n_receiver_pairing_fdr_0_1": int(sub["receiver_pairing_bh_fdr"].le(0.1).sum()),
                "n_edge_vector_fdr_0_1": int(sub["edge_vector_bh_fdr"].le(0.1).sum()),
                "n_receiver_and_vector_fdr_0_1": int(both.sum()),
                "n_supported_edge_null": int(supported.sum()),
                "median_receiver_pairing_fdr": safe_series_median(sub["receiver_pairing_bh_fdr"]),
                "median_edge_vector_fdr": safe_series_median(sub["edge_vector_bh_fdr"]),
                "best_lr": str(best.iloc[0]["lr_pair"]) if len(best) else "",
                "best_lr_receiver_pairing_fdr": float(best.iloc[0]["receiver_pairing_bh_fdr"]) if len(best) else np.nan,
                "best_lr_edge_vector_fdr": float(best.iloc[0]["edge_vector_bh_fdr"]) if len(best) else np.nan,
                "edge_permutation_dataset_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_method_comparison(lr_summary: pd.DataFrame) -> pd.DataFrame:
    if lr_summary.empty:
        return pd.DataFrame()
    complete = lr_summary[lr_summary["status"].eq("completed")].copy()
    rows = []
    for (dataset_key, lr_pair), sub in complete.groupby(["dataset_key", "lr_pair"], sort=False):
        methods = {row.method: row for row in sub.itertuples(index=False)}
        if not {"spatialself", "commot"}.issubset(methods):
            continue
        ss = methods["spatialself"]
        cm = methods["commot"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": ss.dataset,
                "lr_pair": lr_pair,
                "spatialself_expected_edge_fraction": float(ss.observed_expected_edge_score_fraction),
                "commot_expected_edge_fraction": float(cm.observed_expected_edge_score_fraction),
                "spatialself_receiver_pairing_fdr": float(ss.receiver_pairing_bh_fdr),
                "commot_receiver_pairing_fdr": float(cm.receiver_pairing_bh_fdr),
                "spatialself_edge_vector_fdr": float(ss.edge_vector_bh_fdr),
                "commot_edge_vector_fdr": float(cm.edge_vector_bh_fdr),
                "spatialself_edge_supported": bool(
                    ss.edge_permutation_call in {"receiver_pairing_supported", "receiver_and_vector_supported"}
                ),
                "commot_edge_supported": bool(
                    cm.edge_permutation_call in {"receiver_pairing_supported", "receiver_and_vector_supported"}
                ),
                "edge_permutation_comparison_call": compare_pair(ss, cm),
            }
        )
    return pd.DataFrame(rows)


def classify_lr(row: object) -> str:
    if str(getattr(row, "status", "")) != "completed":
        return "not_evaluable"
    recv = to_float(getattr(row, "receiver_pairing_bh_fdr", np.nan))
    vec = to_float(getattr(row, "edge_vector_bh_fdr", np.nan))
    if recv <= 0.1 and vec <= 0.1:
        return "receiver_and_vector_supported"
    if recv <= 0.1:
        return "receiver_pairing_supported"
    if vec <= 0.1:
        return "vector_only_supported"
    return "not_supported_by_edge_permutation"


def dataset_call(sub: pd.DataFrame) -> str:
    both = int(sub["edge_permutation_call"].eq("receiver_and_vector_supported").sum())
    recv = int(sub["edge_permutation_call"].isin(["receiver_pairing_supported", "receiver_and_vector_supported"]).sum())
    if both >= 2:
        return "multiple_lr_receiver_and_vector_supported"
    if recv >= 2:
        return "multiple_lr_receiver_pairing_supported"
    if recv == 1:
        return "one_lr_receiver_pairing_supported"
    return "not_supported_by_edge_permutation"


def compare_pair(ss: object, cm: object) -> str:
    ss_supported = bool(ss.edge_permutation_call in {"receiver_pairing_supported", "receiver_and_vector_supported"})
    cm_supported = bool(cm.edge_permutation_call in {"receiver_pairing_supported", "receiver_and_vector_supported"})
    if ss_supported and not cm_supported:
        return "spatialself_supported_commot_not"
    if cm_supported and not ss_supported:
        return "commot_supported_spatialself_not"
    delta = float(ss.observed_expected_edge_score_fraction) - float(cm.observed_expected_edge_score_fraction)
    if delta >= 0.1:
        return "spatialself_higher_expected_edge_fraction"
    if delta <= -0.1:
        return "commot_higher_expected_edge_fraction"
    return "similar_or_mixed"


def right_tail_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    return float((1.0 + np.sum(null_values >= observed)) / (1.0 + len(null_values)))


def nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def nanquantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else np.nan


def safe_series_median(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else np.nan


def bh_by_group(table: pd.DataFrame, group_cols: list[str], p_col: str) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=float)
    if table.empty or p_col not in table.columns:
        return out
    for _, idx in table.groupby(group_cols).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        out[idx] = bh_adjust(table.iloc[idx][p_col].to_numpy(dtype=float))
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(len(pvalues), np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    p = pvalues[idx]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def missing_row(dataset_key: str, dataset: str, path: Path) -> dict[str, object]:
    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "method": "all",
        "lr_pair": "all",
        "status": "missing_input",
        "missing_input": str(path),
    }


def stable_offset(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 10000)


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.3g}"


def write_status(dataset_summary: pd.DataFrame, method_comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Edge-Level Permutation Audit",
        "",
        "## Scope",
        "",
        "This audit runs edge-level nulls on fixed Phase 11 directionality edge scores. It shuffles receiver niche/coordinate pairing and edge vectors, then recomputes expected-edge score fractions and vector resultant lengths.",
        "It is stronger than aggregated niche-flow label permutation, but still not model retraining under coordinate permutations.",
        "",
        "## Dataset Summary",
        "",
    ]
    if dataset_summary.empty:
        lines.append("- No completed edge-level directionality rows.")
    else:
        for row in dataset_summary.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.method}` `{row.edge_permutation_dataset_call}`: "
                f"LR={int(row.n_lr)}, receiver q<=0.1={int(row.n_receiver_pairing_fdr_0_1)}, "
                f"vector q<=0.1={int(row.n_edge_vector_fdr_0_1)}, both={int(row.n_receiver_and_vector_fdr_0_1)}, "
                f"best={row.best_lr} recv_q={fmt(row.best_lr_receiver_pairing_fdr)}, vec_q={fmt(row.best_lr_edge_vector_fdr)}"
            )
    if not method_comparison.empty:
        lines += ["", "## SpatialSelf vs COMMOT", ""]
        for dataset_key, sub in method_comparison.groupby("dataset_key", sort=False):
            counts = sub["edge_permutation_comparison_call"].value_counts()
            lines.append(
                f"- `{dataset_key}`: " + "; ".join(f"{call}={int(n)}" for call, n in counts.items())
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this as an edge-level fixed-score null, not causal ligand transport proof.",
        "- The receiver-pairing null breaks source/receiver niche-coordinate alignment while preserving edge scores.",
        "- The vector null breaks directional geometry while preserving expected-edge labels and scores.",
        "- Fully end-to-end coordinate permutation would require rerunning model scoring under permuted coordinates.",
    ]
    (OUT_DIR / "phase12_directionality_edge_permutation_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_directionality_edge_permutation_audit",
        "n_permutations": N_PERMUTATIONS,
        "nulls": ["receiver_niche_coordinate_pairing_permutation", "edge_vector_permutation"],
        "inputs": {key: str(spec["path"]) for key, spec in EDGE_INPUTS.items()},
        "outputs": [
            "phase12_directionality_edge_permutation_lr_summary.csv",
            "phase12_directionality_edge_permutation_dataset_summary.csv",
            "phase12_directionality_edge_permutation_method_comparison.csv",
            "phase12_directionality_edge_permutation_status.md",
        ],
        "claim_boundary": "Edge-level fixed-score directionality null; not model retraining under coordinate permutation or causal transport proof.",
    }
    (OUT_DIR / "phase12_directionality_edge_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
