#!/usr/bin/env python
"""Mechanism/category-stratified all-null FDR audit for Phase 12.

The default expanded-atlas permutation audit uses all permuted LR scores as one
global null distribution. That is maximally conservative, but it mixes LR
mechanisms with different distance rules and score ranges. This script reruns
the same three nulls and computes stratum-specific empirical p-values:

- mechanism-stratified null: compare each LR only to permuted LR with the same
  communication mechanism;
- category-stratified null: compare each LR only to permuted LR with the same
  broad biological category.

It is still post-hoc and still uses the same calibrated scores. It is intended
to test whether the spatial-distance bottleneck is partly caused by pooling
heterogeneous LR mechanisms, not to silently replace the conservative all-LR
null.
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
    NULL_TYPES,
    build_null_table,
    prepare_base_table,
    prepare_candidate_arrays,
    stable_dataset_offset,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
REPLICATE_ROOT = EXPANDED_DIR / "replicates"
OUT_DIR = PHASE12_DIR / "stratified_atlas_fdr"
N_PERMUTATIONS = 120
RNG_SEED = 12212026
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

    lr.to_csv(OUT_DIR / "phase12_stratified_atlas_fdr_lr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_stratified_atlas_fdr_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_stratified_atlas_fdr_dataset.csv", index=False)
    null.to_csv(OUT_DIR / "phase12_stratified_atlas_fdr_null_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_stratified_atlas_fdr_story.csv", index=False)
    write_status(dataset, lr, story)
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

    observed = base["calibrated_score"].to_numpy(dtype=float)
    strict_observed = base["strict_discovery_score"].to_numpy(dtype=float)
    null_scores: dict[str, np.ndarray] = {}
    strict_null_scores: dict[str, np.ndarray] = {}
    null_rows = []
    for null_type in NULL_TYPES:
        scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        strict_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            null_table = build_null_table(base, arrays, null_type, rng)
            scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
            strict_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)
        null_scores[null_type] = scores
        strict_null_scores[null_type] = strict_scores
        null_rows.append(
            {
                "dataset_key": dataset_key,
                "null_type": null_type,
                "n_permutations": N_PERMUTATIONS,
                "n_lr": int(len(base)),
                "null_score_mean": float(np.mean(scores)),
                "null_score_q95": float(np.quantile(scores, 0.95)),
                "null_score_q99": float(np.quantile(scores, 0.99)),
                "null_score_max": float(np.max(scores)),
                "strict_null_score_mean": float(np.mean(strict_scores)),
                "strict_null_score_q95": float(np.quantile(strict_scores, 0.95)),
                "strict_null_score_q99": float(np.quantile(strict_scores, 0.99)),
                "strict_null_score_max": float(np.max(strict_scores)),
                "observed_top_score": float(np.max(observed)),
                "observed_top_strict_score": float(np.max(strict_observed)),
            }
        )

    lr = build_lr_table(dataset_key, base, null_scores, strict_null_scores)
    return {"lr": lr, "null": pd.DataFrame(null_rows)}


def build_lr_table(
    dataset_key: str,
    base: pd.DataFrame,
    null_scores: dict[str, np.ndarray],
    strict_null_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = base.copy()
    out["dataset_key"] = dataset_key
    mechanism = out["mechanism"].astype(str).to_numpy()
    category = out["biological_category"].astype(str).to_numpy()
    observed = out["calibrated_score"].to_numpy(dtype=float)
    strict_observed = out["strict_discovery_score"].to_numpy(dtype=float)

    mechanism_p_cols = []
    category_p_cols = []
    strict_mechanism_p_cols = []
    strict_category_p_cols = []
    for null_type in NULL_TYPES:
        mech_p = stratified_empirical_p(observed, null_scores[null_type], mechanism)
        cat_p = stratified_empirical_p(observed, null_scores[null_type], category)
        strict_mech_p = stratified_empirical_p(strict_observed, strict_null_scores[null_type], mechanism)
        strict_cat_p = stratified_empirical_p(strict_observed, strict_null_scores[null_type], category)
        out[f"{null_type}_mechanism_empirical_p"] = mech_p
        out[f"{null_type}_category_empirical_p"] = cat_p
        out[f"strict_{null_type}_mechanism_empirical_p"] = strict_mech_p
        out[f"strict_{null_type}_category_empirical_p"] = strict_cat_p
        out[f"{null_type}_mechanism_bh_q"] = bh_adjust(mech_p)
        out[f"{null_type}_category_bh_q"] = bh_adjust(cat_p)
        out[f"strict_{null_type}_mechanism_bh_q"] = bh_adjust(strict_mech_p)
        out[f"strict_{null_type}_category_bh_q"] = bh_adjust(strict_cat_p)
        mechanism_p_cols.append(f"{null_type}_mechanism_empirical_p")
        category_p_cols.append(f"{null_type}_category_empirical_p")
        strict_mechanism_p_cols.append(f"strict_{null_type}_mechanism_empirical_p")
        strict_category_p_cols.append(f"strict_{null_type}_category_empirical_p")

    out["mechanism_stratified_conservative_empirical_p"] = out[mechanism_p_cols].max(axis=1)
    out["category_stratified_conservative_empirical_p"] = out[category_p_cols].max(axis=1)
    out["strict_mechanism_stratified_conservative_empirical_p"] = out[strict_mechanism_p_cols].max(axis=1)
    out["strict_category_stratified_conservative_empirical_p"] = out[strict_category_p_cols].max(axis=1)
    out["mechanism_stratified_conservative_bh_q"] = bh_adjust(
        out["mechanism_stratified_conservative_empirical_p"].to_numpy(dtype=float)
    )
    out["category_stratified_conservative_bh_q"] = bh_adjust(
        out["category_stratified_conservative_empirical_p"].to_numpy(dtype=float)
    )
    out["strict_mechanism_stratified_conservative_bh_q"] = bh_adjust(
        out["strict_mechanism_stratified_conservative_empirical_p"].to_numpy(dtype=float)
    )
    out["strict_category_stratified_conservative_bh_q"] = bh_adjust(
        out["strict_category_stratified_conservative_empirical_p"].to_numpy(dtype=float)
    )
    out["stratified_fdr_call"] = [lr_call(row) for row in out.itertuples(index=False)]

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
        "mechanism_stratified_conservative_empirical_p",
        "mechanism_stratified_conservative_bh_q",
        "category_stratified_conservative_empirical_p",
        "category_stratified_conservative_bh_q",
        "strict_mechanism_stratified_conservative_empirical_p",
        "strict_mechanism_stratified_conservative_bh_q",
        "strict_category_stratified_conservative_empirical_p",
        "strict_category_stratified_conservative_bh_q",
        "stratified_fdr_call",
    ]
    for null_type in NULL_TYPES:
        columns += [
            f"{null_type}_mechanism_bh_q",
            f"{null_type}_category_bh_q",
            f"strict_{null_type}_mechanism_bh_q",
            f"strict_{null_type}_category_bh_q",
        ]
    return out[columns].sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def stratified_empirical_p(observed: np.ndarray, null_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pvals = np.full(len(observed), np.nan, dtype=float)
    for label in pd.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        null_flat = null_scores[:, idx].ravel()
        for lr_idx in idx:
            pvals[lr_idx] = (1.0 + np.sum(null_flat >= observed[lr_idx])) / (1.0 + len(null_flat))
    return pvals


def summarize_topk(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            true = top["true_lr"].astype(bool)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(true.sum()),
                    "n_mechanism_q_0_1": int(top["mechanism_stratified_conservative_bh_q"].le(0.1).sum()),
                    "n_true_mechanism_q_0_1": int(
                        (true & top["mechanism_stratified_conservative_bh_q"].le(0.1)).sum()
                    ),
                    "n_category_q_0_1": int(top["category_stratified_conservative_bh_q"].le(0.1).sum()),
                    "n_true_category_q_0_1": int(
                        (true & top["category_stratified_conservative_bh_q"].le(0.1)).sum()
                    ),
                    "n_strict_mechanism_q_0_1": int(
                        top["strict_mechanism_stratified_conservative_bh_q"].le(0.1).sum()
                    ),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(lr: pd.DataFrame, topk: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        true = sub["true_lr"].astype(bool)
        top10 = pick(topk, dataset_key=dataset_key, top_k=10)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "top10_true_lr": int(get(top10, "n_true_lr", 0)),
                "mechanism_q_0_1": int(sub["mechanism_stratified_conservative_bh_q"].le(0.1).sum()),
                "true_mechanism_q_0_1": int((true & sub["mechanism_stratified_conservative_bh_q"].le(0.1)).sum()),
                "category_q_0_1": int(sub["category_stratified_conservative_bh_q"].le(0.1).sum()),
                "true_category_q_0_1": int((true & sub["category_stratified_conservative_bh_q"].le(0.1)).sum()),
                "strict_mechanism_q_0_1": int(sub["strict_mechanism_stratified_conservative_bh_q"].le(0.1).sum()),
                "true_strict_mechanism_q_0_1": int(
                    (true & sub["strict_mechanism_stratified_conservative_bh_q"].le(0.1)).sum()
                ),
                "strict_category_q_0_1": int(sub["strict_category_stratified_conservative_bh_q"].le(0.1).sum()),
                "true_strict_category_q_0_1": int(
                    (true & sub["strict_category_stratified_conservative_bh_q"].le(0.1)).sum()
                ),
                "best_true_mechanism_q_lr": best_true_lr(sub, "mechanism_stratified_conservative_bh_q", ascending=True),
                "best_true_category_q_lr": best_true_lr(sub, "category_stratified_conservative_bh_q", ascending=True),
                "stratified_atlas_fdr_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    mech_true = int(pd.to_numeric(dataset["true_mechanism_q_0_1"], errors="coerce").fillna(0).sum())
    cat_true = int(pd.to_numeric(dataset["true_category_q_0_1"], errors="coerce").fillna(0).sum())
    strict_mech_true = int(pd.to_numeric(dataset["true_strict_mechanism_q_0_1"], errors="coerce").fillna(0).sum())
    if mech_true > 0:
        call = "mechanism_stratified_all_null_fdr_supported"
    elif cat_true > 0:
        call = "category_stratified_all_null_fdr_supported"
    elif strict_mech_true > 0:
        call = "strict_mechanism_stratified_support_only"
    else:
        call = "stratified_all_null_fdr_not_supported"
    return pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset)),
                "n_true_mechanism_stratified_q_0_1": mech_true,
                "n_true_category_stratified_q_0_1": cat_true,
                "n_true_strict_mechanism_stratified_q_0_1": strict_mech_true,
                "overall_stratified_atlas_fdr_call": call,
                "claim_boundary": "Mechanism/category-stratified all-null FDR is a post-hoc sensitivity audit. It can support a stratified-FDR claim only if reported separately from the stricter pooled whole-atlas FDR.",
            }
        ]
    )


def lr_call(row: Any) -> str:
    mech_q = to_float(getattr(row, "mechanism_stratified_conservative_bh_q", np.nan))
    cat_q = to_float(getattr(row, "category_stratified_conservative_bh_q", np.nan))
    strict_mech_q = to_float(getattr(row, "strict_mechanism_stratified_conservative_bh_q", np.nan))
    if np.isfinite(mech_q) and mech_q <= 0.1:
        return "mechanism_stratified_all_null_fdr_supported"
    if np.isfinite(cat_q) and cat_q <= 0.1:
        return "category_stratified_all_null_fdr_supported"
    if np.isfinite(strict_mech_q) and strict_mech_q <= 0.1:
        return "strict_mechanism_stratified_support"
    return "not_stratified_fdr_supported"


def dataset_call(sub: pd.DataFrame) -> str:
    true = sub["true_lr"].astype(bool)
    mech_true = int((true & sub["mechanism_stratified_conservative_bh_q"].le(0.1)).sum())
    cat_true = int((true & sub["category_stratified_conservative_bh_q"].le(0.1)).sum())
    if mech_true > 0:
        return "mechanism_stratified_true_lr_fdr_supported"
    if cat_true > 0:
        return "category_stratified_true_lr_fdr_supported"
    return "stratified_fdr_not_supported"


def best_true_lr(sub: pd.DataFrame, column: str, *, ascending: bool) -> str:
    true = sub[sub["true_lr"].astype(bool)].copy()
    true[column] = pd.to_numeric(true[column], errors="coerce")
    true = true[np.isfinite(true[column])]
    if true.empty:
        return ""
    row = true.sort_values(column, ascending=ascending).iloc[0]
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


def write_status(dataset: pd.DataFrame, lr: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Stratified Atlas FDR",
        "",
        "## Scope",
        "",
        "This post-hoc sensitivity audit recomputes receiver, spatial-distance and LR-identity nulls, then evaluates empirical p-values within LR mechanism or biological-category strata.",
        "It does not replace the stricter pooled whole-atlas FDR unless explicitly reported as a separate stratified FDR analysis.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.stratified_atlas_fdr_call}`: LR={int(row.n_lr)}, "
            f"true mechanism q<=0.1={int(row.true_mechanism_q_0_1)}, "
            f"true category q<=0.1={int(row.true_category_q_0_1)}, "
            f"strict true mechanism q<=0.1={int(row.true_strict_mechanism_q_0_1)}, "
            f"best true mechanism q={row.best_true_mechanism_q_lr}."
        )
    lines += ["", "## Top LR", ""]
    for row in lr.sort_values(["dataset_key", "rank"]).groupby("dataset_key").head(10).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` rank {int(row.rank)} `{row.lr_pair}`: {row.stratified_fdr_call}; "
            f"mechanism q={fmt(row.mechanism_stratified_conservative_bh_q)}, "
            f"category q={fmt(row.category_stratified_conservative_bh_q)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_stratified_atlas_fdr_call}`: true mechanism q<=0.1={int(s.n_true_mechanism_stratified_q_0_1)}, "
            f"true category q<=0.1={int(s.n_true_category_stratified_q_0_1)}, "
            f"strict true mechanism q<=0.1={int(s.n_true_strict_mechanism_stratified_q_0_1)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_stratified_atlas_fdr_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    lr: pd.DataFrame,
    topk: pd.DataFrame,
    dataset: pd.DataFrame,
    null: pd.DataFrame,
    story: pd.DataFrame,
) -> None:
    manifest = {
        "task": "phase12_stratified_atlas_fdr",
        "n_permutations": N_PERMUTATIONS,
        "null_types": list(NULL_TYPES),
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
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_lr.csv"),
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_topk.csv"),
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_dataset.csv"),
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_null_summary.csv"),
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_story.csv"),
            rel(OUT_DIR / "phase12_stratified_atlas_fdr_status.md"),
        ],
        "claim_boundary": "Mechanism/category-stratified all-null FDR sensitivity audit; not the stricter pooled whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_stratified_atlas_fdr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
