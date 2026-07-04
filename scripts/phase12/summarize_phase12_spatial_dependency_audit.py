#!/usr/bin/env python
"""Spatial-dependency effect audit for Phase 12 expanded atlas rankings.

The conservative all-null FDR is currently blocked by the spatial-distance null.
This script does not replace that FDR. Instead, it quantifies the effect size of
the spatial-distance null for every LR pair:

- observed calibrated score;
- mean/q95/null maximum under shuffled distances;
- observed-minus-null delta;
- z-score against the shuffled-distance null;
- within-LR empirical p-value and BH q-value.

The goal is to separate two failure modes:

1. no spatial dependence at all;
2. spatial dependence exists for selected LR, but the global max-null criterion
   is too conservative for whole-atlas discovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase12.run_phase12_expanded_atlas_permutation_audit import (  # noqa: E402
    build_null_table,
    prepare_base_table,
    prepare_candidate_arrays,
    stable_dataset_offset,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
REPLICATE_ROOT = EXPANDED_DIR / "replicates"
OUT_DIR = PHASE12_DIR / "spatial_dependency_audit"
N_PERMUTATIONS = 160
RNG_SEED = 12192026
TOP_K_VALUES = (10, 20, 50, 100)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    lr_rows = []
    null_rows = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        result = run_dataset(dataset_key, ranking[ranking["dataset_key"].eq(dataset_key)].copy())
        lr_rows.append(result["lr"])
        null_rows.append(result["null"])
    lr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    topk = summarize_topk(lr)
    dataset = summarize_dataset(lr, topk, null)
    story = summarize_story(dataset)

    lr.to_csv(OUT_DIR / "phase12_spatial_dependency_lr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_spatial_dependency_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_spatial_dependency_dataset.csv", index=False)
    null.to_csv(OUT_DIR / "phase12_spatial_dependency_null_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_spatial_dependency_story.csv", index=False)
    write_status(dataset, topk, story)
    write_manifest(lr, topk, dataset, null, story)

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr.groupby("dataset_key").head(15).to_string(index=False))
    print()
    print(story.to_string(index=False))


def run_dataset(dataset_key: str, ranking: pd.DataFrame) -> dict[str, pd.DataFrame]:
    replicate_dir = REPLICATE_ROOT / dataset_key
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    base = prepare_base_table(ranking.sort_values("rank"), lr_pairs)
    arrays = prepare_candidate_arrays(candidates, base)
    rng = np.random.default_rng(RNG_SEED + stable_dataset_offset(dataset_key))

    null_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    strict_null_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    for perm_idx in range(N_PERMUTATIONS):
        null_table = build_null_table(base, arrays, "spatial_distance_permutation", rng)
        null_scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
        strict_null_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)

    lr = build_lr_table(dataset_key, base, null_scores, strict_null_scores)
    null_summary = pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "n_permutations": N_PERMUTATIONS,
                "n_lr": int(len(base)),
                "null_score_mean": float(np.mean(null_scores)),
                "null_score_q95": float(np.quantile(null_scores, 0.95)),
                "null_score_q99": float(np.quantile(null_scores, 0.99)),
                "null_score_max": float(np.max(null_scores)),
                "strict_null_score_mean": float(np.mean(strict_null_scores)),
                "strict_null_score_q95": float(np.quantile(strict_null_scores, 0.95)),
                "strict_null_score_q99": float(np.quantile(strict_null_scores, 0.99)),
                "strict_null_score_max": float(np.max(strict_null_scores)),
                "observed_top_score": float(np.max(base["calibrated_score"])),
                "observed_top_strict_score": float(np.max(base["strict_discovery_score"])),
            }
        ]
    )
    return {"lr": lr, "null": null_summary}


def build_lr_table(
    dataset_key: str,
    base: pd.DataFrame,
    null_scores: np.ndarray,
    strict_null_scores: np.ndarray,
) -> pd.DataFrame:
    observed = base["calibrated_score"].to_numpy(dtype=float)
    strict_observed = base["strict_discovery_score"].to_numpy(dtype=float)
    null_mean = np.mean(null_scores, axis=0)
    null_sd = np.std(null_scores, axis=0, ddof=1)
    strict_null_mean = np.mean(strict_null_scores, axis=0)
    strict_null_sd = np.std(strict_null_scores, axis=0, ddof=1)

    out = base.copy()
    out["dataset_key"] = dataset_key
    out["spatial_null_mean"] = null_mean
    out["spatial_null_q50"] = np.quantile(null_scores, 0.50, axis=0)
    out["spatial_null_q95"] = np.quantile(null_scores, 0.95, axis=0)
    out["spatial_null_max"] = np.max(null_scores, axis=0)
    out["spatial_delta"] = observed - null_mean
    out["spatial_z"] = np.divide(out["spatial_delta"], null_sd, out=np.full(len(out), np.nan), where=null_sd > 0)
    out["spatial_within_empirical_p"] = [
        (1.0 + np.sum(null_scores[:, idx] >= observed[idx])) / (1.0 + N_PERMUTATIONS)
        for idx in range(len(out))
    ]
    out["spatial_delta_positive"] = out["spatial_delta"].gt(0)

    out["strict_spatial_null_mean"] = strict_null_mean
    out["strict_spatial_null_q50"] = np.quantile(strict_null_scores, 0.50, axis=0)
    out["strict_spatial_null_q95"] = np.quantile(strict_null_scores, 0.95, axis=0)
    out["strict_spatial_null_max"] = np.max(strict_null_scores, axis=0)
    out["strict_spatial_delta"] = strict_observed - strict_null_mean
    out["strict_spatial_z"] = np.divide(
        out["strict_spatial_delta"],
        strict_null_sd,
        out=np.full(len(out), np.nan),
        where=strict_null_sd > 0,
    )
    out["strict_spatial_within_empirical_p"] = [
        (1.0 + np.sum(strict_null_scores[:, idx] >= strict_observed[idx])) / (1.0 + N_PERMUTATIONS)
        for idx in range(len(out))
    ]
    out["strict_spatial_delta_positive"] = out["strict_spatial_delta"].gt(0)

    out["spatial_within_bh_q"] = bh_adjust(out["spatial_within_empirical_p"].to_numpy(dtype=float))
    out["strict_spatial_within_bh_q"] = bh_adjust(out["strict_spatial_within_empirical_p"].to_numpy(dtype=float))
    out["spatial_dependency_call"] = [
        lr_call(row) for row in out.itertuples(index=False)
    ]

    columns = [
        "dataset_key",
        "rank",
        "strict_rank",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "true_lr",
        "selected_reason",
        "biological_category",
        "mechanism",
        "calibrated_score",
        "strict_discovery_score",
        "spatial_null_mean",
        "spatial_null_q50",
        "spatial_null_q95",
        "spatial_null_max",
        "spatial_delta",
        "spatial_z",
        "spatial_within_empirical_p",
        "spatial_within_bh_q",
        "spatial_delta_positive",
        "strict_spatial_null_mean",
        "strict_spatial_null_q50",
        "strict_spatial_null_q95",
        "strict_spatial_null_max",
        "strict_spatial_delta",
        "strict_spatial_z",
        "strict_spatial_within_empirical_p",
        "strict_spatial_within_bh_q",
        "strict_spatial_delta_positive",
        "spatial_dependency_call",
    ]
    return out[columns].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def summarize_topk(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "n_spatial_delta_positive": int(top["spatial_delta_positive"].sum()),
                    "n_true_spatial_delta_positive": int(
                        (top["true_lr"].astype(bool) & top["spatial_delta_positive"]).sum()
                    ),
                    "median_spatial_delta": float(top["spatial_delta"].median()),
                    "median_spatial_z": float(top["spatial_z"].median()),
                    "n_spatial_within_q_0_1": int(top["spatial_within_bh_q"].le(0.1).sum()),
                    "n_strict_spatial_within_q_0_1": int(top["strict_spatial_within_bh_q"].le(0.1).sum()),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(lr: pd.DataFrame, topk: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        true = sub["true_lr"].astype(bool)
        top10 = pick(topk, dataset_key=dataset_key, top_k=10)
        null_row = pick(null, dataset_key=dataset_key)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "n_spatial_delta_positive": int(sub["spatial_delta_positive"].sum()),
                "n_true_spatial_delta_positive": int((true & sub["spatial_delta_positive"]).sum()),
                "n_spatial_within_q_0_1": int(sub["spatial_within_bh_q"].le(0.1).sum()),
                "n_true_spatial_within_q_0_1": int((true & sub["spatial_within_bh_q"].le(0.1)).sum()),
                "n_strict_spatial_within_q_0_1": int(sub["strict_spatial_within_bh_q"].le(0.1).sum()),
                "n_true_strict_spatial_within_q_0_1": int((true & sub["strict_spatial_within_bh_q"].le(0.1)).sum()),
                "top10_true_lr": int(get(top10, "n_true_lr", 0)),
                "top10_true_spatial_delta_positive": int(get(top10, "n_true_spatial_delta_positive", 0)),
                "top10_median_spatial_delta": float(get(top10, "median_spatial_delta", np.nan)),
                "top10_median_spatial_z": float(get(top10, "median_spatial_z", np.nan)),
                "observed_top_exceeds_null_max": bool(
                    to_float(get(null_row, "observed_top_score", np.nan)) > to_float(get(null_row, "null_score_max", np.nan))
                ),
                "observed_top_strict_exceeds_null_max": bool(
                    to_float(get(null_row, "observed_top_strict_score", np.nan))
                    > to_float(get(null_row, "strict_null_score_max", np.nan))
                ),
                "best_spatial_dependency_lr": best_lr(sub, "spatial_delta"),
                "best_true_spatial_dependency_lr": best_true_lr(sub, "spatial_delta"),
                "spatial_dependency_audit_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    true_delta = int(pd.to_numeric(dataset["n_true_spatial_delta_positive"], errors="coerce").fillna(0).sum())
    true_q = int(pd.to_numeric(dataset["n_true_spatial_within_q_0_1"], errors="coerce").fillna(0).sum())
    strict_true_q = int(pd.to_numeric(dataset["n_true_strict_spatial_within_q_0_1"], errors="coerce").fillna(0).sum())
    if true_q > 0 or strict_true_q > 0:
        call = "spatial_dependency_within_lr_fdr_supported"
    elif true_delta > 0:
        call = "spatial_dependency_effects_detected_without_bh_fdr"
    else:
        call = "spatial_dependency_not_supported"
    return pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset)),
                "n_true_lr_with_positive_spatial_delta": true_delta,
                "n_true_lr_spatial_within_q_0_1": true_q,
                "n_true_lr_strict_spatial_within_q_0_1": strict_true_q,
                "overall_spatial_dependency_call": call,
                "claim_boundary": "Spatial-dependency effect sizes and within-LR q-values diagnose the spatial null. They do not replace global whole-atlas discovery FDR.",
            }
        ]
    )


def lr_call(row: Any) -> str:
    q = to_float(getattr(row, "spatial_within_bh_q", np.nan))
    delta = to_float(getattr(row, "spatial_delta", np.nan))
    z = to_float(getattr(row, "spatial_z", np.nan))
    strict_q = to_float(getattr(row, "strict_spatial_within_bh_q", np.nan))
    if (np.isfinite(q) and q <= 0.1) or (np.isfinite(strict_q) and strict_q <= 0.1):
        return "within_lr_spatial_fdr_supported"
    if np.isfinite(delta) and delta > 0 and np.isfinite(z) and z >= 1.0:
        return "positive_spatial_effect"
    if np.isfinite(delta) and delta > 0:
        return "weak_positive_spatial_effect"
    return "no_positive_spatial_effect"


def dataset_call(sub: pd.DataFrame) -> str:
    true = sub["true_lr"].astype(bool)
    n_true_q = int((true & sub["spatial_within_bh_q"].le(0.1)).sum())
    n_true_delta = int((true & sub["spatial_delta_positive"]).sum())
    if n_true_q > 0:
        return "within_lr_spatial_fdr_supported_for_true_lr"
    if n_true_delta > 0:
        return "positive_spatial_effects_without_bh_fdr"
    return "no_true_lr_spatial_dependency_detected"


def best_lr(sub: pd.DataFrame, column: str) -> str:
    if sub.empty:
        return ""
    row = sub.sort_values(column, ascending=False).iloc[0]
    return f"{row.lr_pair}:{row[column]:.4g}"


def best_true_lr(sub: pd.DataFrame, column: str) -> str:
    true = sub[sub["true_lr"].astype(bool)]
    if true.empty:
        return ""
    row = true.sort_values(column, ascending=False).iloc[0]
    return f"{row.lr_pair}:{row[column]:.4g}"


def pick(table: pd.DataFrame, **filters: object) -> dict[str, Any]:
    if table.empty:
        return {}
    sub = table
    for key, value in filters.items():
        sub = sub[sub[key].eq(value)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def get(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return value


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


def to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Spatial-Dependency Audit",
        "",
        "## Scope",
        "",
        "This audit quantifies observed-minus-null effects under the spatial-distance permutation null. It diagnoses the spatial-null bottleneck and does not replace global whole-atlas FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.spatial_dependency_audit_call}`: LR={int(row.n_lr)}, "
            f"true LR={int(row.n_true_lr)}, true positive spatial delta={int(row.n_true_spatial_delta_positive)}, "
            f"true spatial q<=0.1={int(row.n_true_spatial_within_q_0_1)}, "
            f"strict true spatial q<=0.1={int(row.n_true_strict_spatial_within_q_0_1)}, "
            f"best true delta={row.best_true_spatial_dependency_lr}."
        )
    lines += ["", "## Top-K", ""]
    for row in topk[topk["top_k"].isin([10, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"true positive delta={int(row.n_true_spatial_delta_positive)}, "
            f"median delta={fmt(row.median_spatial_delta)}, median z={fmt(row.median_spatial_z)}, "
            f"within q<=0.1={int(row.n_spatial_within_q_0_1)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_spatial_dependency_call}`: true positive delta={int(s.n_true_lr_with_positive_spatial_delta)}, "
            f"true within q<=0.1={int(s.n_true_lr_spatial_within_q_0_1)}, "
            f"strict true within q<=0.1={int(s.n_true_lr_strict_spatial_within_q_0_1)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_spatial_dependency_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    lr: pd.DataFrame,
    topk: pd.DataFrame,
    dataset: pd.DataFrame,
    null: pd.DataFrame,
    story: pd.DataFrame,
) -> None:
    manifest = {
        "task": "phase12_spatial_dependency_audit",
        "n_permutations": N_PERMUTATIONS,
        "n_lr_rows": int(len(lr)),
        "n_topk_rows": int(len(topk)),
        "n_dataset_rows": int(len(dataset)),
        "n_null_rows": int(len(null)),
        "n_story_rows": int(len(story)),
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_spatial_dependency_lr.csv"),
            rel(OUT_DIR / "phase12_spatial_dependency_topk.csv"),
            rel(OUT_DIR / "phase12_spatial_dependency_dataset.csv"),
            rel(OUT_DIR / "phase12_spatial_dependency_null_summary.csv"),
            rel(OUT_DIR / "phase12_spatial_dependency_story.csv"),
            rel(OUT_DIR / "phase12_spatial_dependency_status.md"),
        ],
        "claim_boundary": "Spatial-distance null effect-size audit; not global whole-atlas discovery FDR.",
    }
    (OUT_DIR / "phase12_spatial_dependency_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
