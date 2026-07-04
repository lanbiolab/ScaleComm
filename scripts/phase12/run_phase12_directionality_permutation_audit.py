#!/usr/bin/env python
"""Permutation audit for Phase 12 directionality metrics.

This operates on Phase 11 aggregated niche-flow tables. It tests whether the
observed expected-flow concentration is higher than two label-based nulls:
LR-label permutation within a dataset/method and niche-flow label permutation
within each dataset/method/LR. It is not a coordinate-level edge rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_TRAJ_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/trajectory_summary"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/directionality_permutation"
N_PERMUTATIONS = 1000
RNG_SEED = 12012


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flow = pd.read_csv(PHASE11_TRAJ_DIR / "phase11_signal_trajectory_niche_flow_all.csv")
    observed = build_observed(flow)
    perm_lr = lr_label_permutation(flow, observed, N_PERMUTATIONS)
    perm_niche = niche_label_permutation(flow, observed, N_PERMUTATIONS)
    combined = combine_pvalues(observed, perm_lr, perm_niche)
    dataset_summary = summarize_dataset(combined)
    comparison = summarize_method_comparison(combined)

    observed.to_csv(OUT_DIR / "phase12_directionality_permutation_observed_lr.csv", index=False)
    perm_lr.to_csv(OUT_DIR / "phase12_directionality_lr_label_permutation_pvalues.csv", index=False)
    perm_niche.to_csv(OUT_DIR / "phase12_directionality_niche_label_permutation_pvalues.csv", index=False)
    combined.to_csv(OUT_DIR / "phase12_directionality_permutation_lr_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_directionality_permutation_dataset_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_directionality_permutation_method_comparison.csv", index=False)
    write_status(dataset_summary, comparison)
    write_manifest()
    print(dataset_summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


def build_observed(flow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, dataset, method, lr_pair), sub in flow.groupby(
        ["dataset_key", "dataset", "method", "lr_pair"], sort=False
    ):
        rows.append(metric_row(dataset_key, dataset, method, lr_pair, sub, sub["expected_flow"].astype(bool).to_numpy()))
    out = pd.DataFrame(rows)
    out["observed_rank_within_dataset_method"] = out.groupby(["dataset_key", "method"])["expected_flow_score_fraction"].rank(
        ascending=False, method="min"
    )
    return out


def metric_row(dataset_key: str, dataset: str, method: str, lr_pair: str, sub: pd.DataFrame, expected_mask: np.ndarray) -> dict[str, object]:
    scores = pd.to_numeric(sub["total_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total = float(scores.sum())
    expected_score = float(scores[expected_mask].sum()) if len(scores) else 0.0
    expected_fraction = expected_score / total if total > 0 else np.nan
    if np.any(expected_mask):
        expected_flow_scores = scores[expected_mask]
        observed_best_expected_fraction = float(expected_flow_scores.max() / total) if total > 0 else np.nan
        rank_values = pd.Series(scores).rank(ascending=False, method="min").to_numpy()
        best_rank = int(np.min(rank_values[expected_mask]))
    else:
        observed_best_expected_fraction = np.nan
        best_rank = -1
    if len(scores):
        top_idx = int(np.argmax(scores))
        top_expected = bool(expected_mask[top_idx])
    else:
        top_expected = False
    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "method": method,
        "lr_pair": lr_pair,
        "n_flows": int(len(sub)),
        "n_expected_flows": int(np.sum(expected_mask)),
        "total_score": total,
        "expected_flow_score": expected_score,
        "expected_flow_score_fraction": expected_fraction,
        "best_expected_flow_fraction": observed_best_expected_fraction,
        "expected_flow_best_rank": best_rank,
        "top_flow_expected": top_expected,
    }


def lr_label_permutation(flow: pd.DataFrame, observed: pd.DataFrame, n_permutations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    obs_lookup = observed.set_index(["dataset_key", "method", "lr_pair"])
    for (dataset_key, method), group in flow.groupby(["dataset_key", "method"], sort=False):
        lr_pairs = list(group["lr_pair"].drop_duplicates())
        lr_expected = {
            lr: group[group["lr_pair"].eq(lr)]["expected_flow"].astype(bool).to_numpy()
            for lr in lr_pairs
        }
        lr_sub = {lr: group[group["lr_pair"].eq(lr)].copy() for lr in lr_pairs}
        null = {lr: [] for lr in lr_pairs}
        for _ in range(n_permutations):
            shuffled = rng.permutation(lr_pairs)
            for lr, donor_lr in zip(lr_pairs, shuffled):
                sub = lr_sub[lr]
                mask = resize_mask(lr_expected[donor_lr], len(sub))
                scores = pd.to_numeric(sub["total_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
                total = float(scores.sum())
                frac = float(scores[mask].sum() / total) if total > 0 and len(scores) else np.nan
                null[lr].append(frac)
        for lr in lr_pairs:
            obs = obs_lookup.loc[(dataset_key, method, lr)]
            null_values = np.asarray(null[lr], dtype=float)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": obs["dataset"],
                    "method": method,
                    "lr_pair": lr,
                    "null_type": "lr_label_permutation",
                    "n_permutations": n_permutations,
                    "observed_expected_flow_score_fraction": float(obs["expected_flow_score_fraction"]),
                    "null_mean": float(np.nanmean(null_values)),
                    "null_q95": float(np.nanquantile(null_values, 0.95)),
                    "null_q99": float(np.nanquantile(null_values, 0.99)),
                    "permutation_p_value": right_tail_p(float(obs["expected_flow_score_fraction"]), null_values),
                }
            )
    out = pd.DataFrame(rows)
    out["permutation_bh_fdr"] = bh_by_group(out, ["dataset_key", "method", "null_type"], "permutation_p_value")
    return out


def niche_label_permutation(flow: pd.DataFrame, observed: pd.DataFrame, n_permutations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 17)
    rows = []
    obs_lookup = observed.set_index(["dataset_key", "method", "lr_pair"])
    for (dataset_key, method, lr_pair), sub in flow.groupby(["dataset_key", "method", "lr_pair"], sort=False):
        scores = pd.to_numeric(sub["total_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        expected_mask = sub["expected_flow"].astype(bool).to_numpy()
        n_expected = int(expected_mask.sum())
        null_values = []
        for _ in range(n_permutations):
            mask = np.zeros(len(scores), dtype=bool)
            if n_expected > 0 and len(scores):
                idx = rng.choice(len(scores), size=min(n_expected, len(scores)), replace=False)
                mask[idx] = True
            total = float(scores.sum())
            null_values.append(float(scores[mask].sum() / total) if total > 0 and len(scores) else np.nan)
        null_values = np.asarray(null_values, dtype=float)
        obs = obs_lookup.loc[(dataset_key, method, lr_pair)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": obs["dataset"],
                "method": method,
                "lr_pair": lr_pair,
                "null_type": "niche_label_permutation",
                "n_permutations": n_permutations,
                "observed_expected_flow_score_fraction": float(obs["expected_flow_score_fraction"]),
                "null_mean": float(np.nanmean(null_values)),
                "null_q95": float(np.nanquantile(null_values, 0.95)),
                "null_q99": float(np.nanquantile(null_values, 0.99)),
                "permutation_p_value": right_tail_p(float(obs["expected_flow_score_fraction"]), null_values),
            }
        )
    out = pd.DataFrame(rows)
    out["permutation_bh_fdr"] = bh_by_group(out, ["dataset_key", "method", "null_type"], "permutation_p_value")
    return out


def combine_pvalues(observed: pd.DataFrame, perm_lr: pd.DataFrame, perm_niche: pd.DataFrame) -> pd.DataFrame:
    out = observed.copy()
    for table, prefix in [(perm_lr, "lr_label"), (perm_niche, "niche_label")]:
        cols = [
            "dataset_key",
            "method",
            "lr_pair",
            "null_mean",
            "null_q95",
            "null_q99",
            "permutation_p_value",
            "permutation_bh_fdr",
        ]
        rename = {
            "null_mean": f"{prefix}_null_mean",
            "null_q95": f"{prefix}_null_q95",
            "null_q99": f"{prefix}_null_q99",
            "permutation_p_value": f"{prefix}_permutation_p_value",
            "permutation_bh_fdr": f"{prefix}_permutation_bh_fdr",
        }
        out = out.merge(table[cols].rename(columns=rename), on=["dataset_key", "method", "lr_pair"], how="left")
    out["passes_lr_label_fdr_0_1"] = out["lr_label_permutation_bh_fdr"].le(0.1)
    out["passes_niche_label_fdr_0_1"] = out["niche_label_permutation_bh_fdr"].le(0.1)
    out["passes_both_fdr_0_1"] = out["passes_lr_label_fdr_0_1"] & out["passes_niche_label_fdr_0_1"]
    out["directionality_permutation_call"] = [
        permutation_call(row) for row in out.itertuples(index=False)
    ]
    return out


def summarize_dataset(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in combined.groupby(["dataset_key", "method"], sort=False):
        best = sub.sort_values(["passes_both_fdr_0_1", "expected_flow_score_fraction"], ascending=False).iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method": method,
                "n_lr": int(len(sub)),
                "n_lr_label_fdr_0_1": int(sub["passes_lr_label_fdr_0_1"].sum()),
                "n_niche_label_fdr_0_1": int(sub["passes_niche_label_fdr_0_1"].sum()),
                "n_both_fdr_0_1": int(sub["passes_both_fdr_0_1"].sum()),
                "mean_expected_flow_fraction": float(sub["expected_flow_score_fraction"].mean()),
                "median_lr_label_fdr": float(sub["lr_label_permutation_bh_fdr"].median()),
                "median_niche_label_fdr": float(sub["niche_label_permutation_bh_fdr"].median()),
                "best_lr": best["lr_pair"],
                "best_lr_expected_flow_fraction": float(best["expected_flow_score_fraction"]),
                "best_lr_lr_label_fdr": float(best["lr_label_permutation_bh_fdr"]),
                "best_lr_niche_label_fdr": float(best["niche_label_permutation_bh_fdr"]),
                "permutation_audit_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_method_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, lr_pair), sub in combined.groupby(["dataset_key", "lr_pair"], sort=False):
        by_method = {row.method: row for row in sub.itertuples(index=False)}
        if not {"spatialself", "commot"}.issubset(by_method):
            continue
        ss = by_method["spatialself"]
        cm = by_method["commot"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": ss.dataset,
                "lr_pair": lr_pair,
                "spatialself_expected_flow_fraction": float(ss.expected_flow_score_fraction),
                "commot_expected_flow_fraction": float(cm.expected_flow_score_fraction),
                "spatialself_both_fdr_0_1": bool(ss.passes_both_fdr_0_1),
                "commot_both_fdr_0_1": bool(cm.passes_both_fdr_0_1),
                "spatialself_lr_label_fdr": float(ss.lr_label_permutation_bh_fdr),
                "commot_lr_label_fdr": float(cm.lr_label_permutation_bh_fdr),
                "spatialself_niche_label_fdr": float(ss.niche_label_permutation_bh_fdr),
                "commot_niche_label_fdr": float(cm.niche_label_permutation_bh_fdr),
                "comparison_call": compare_pair(ss, cm),
            }
        )
    return pd.DataFrame(rows)


def resize_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == target_len:
        return mask.copy()
    out = np.zeros(target_len, dtype=bool)
    n_true = min(int(mask.sum()), target_len)
    if n_true:
        out[:n_true] = True
    return out


def right_tail_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    return float((1 + np.sum(null_values >= observed)) / (1 + len(null_values)))


def bh_by_group(table: pd.DataFrame, group_cols: list[str], p_col: str) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=float)
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


def permutation_call(row: object) -> str:
    if bool(row.passes_both_fdr_0_1):
        return "label_permutation_supported"
    if bool(row.passes_lr_label_fdr_0_1) or bool(row.passes_niche_label_fdr_0_1):
        return "single_null_supported_only"
    return "not_supported_by_label_permutation"


def dataset_call(sub: pd.DataFrame) -> str:
    n_both = int(sub["passes_both_fdr_0_1"].sum())
    n_single = int((sub["passes_lr_label_fdr_0_1"] | sub["passes_niche_label_fdr_0_1"]).sum())
    if n_both >= 2:
        return "multiple_lr_supported_by_label_permutation"
    if n_both == 1:
        return "one_lr_supported_by_label_permutation"
    if n_single:
        return "single_null_support_only"
    return "not_supported_by_label_permutation"


def compare_pair(ss: object, cm: object) -> str:
    if bool(ss.passes_both_fdr_0_1) and not bool(cm.passes_both_fdr_0_1):
        return "spatialself_supported_commot_not"
    if bool(cm.passes_both_fdr_0_1) and not bool(ss.passes_both_fdr_0_1):
        return "commot_supported_spatialself_not"
    delta = float(ss.expected_flow_score_fraction) - float(cm.expected_flow_score_fraction)
    if delta >= 0.1:
        return "spatialself_higher_fraction"
    if delta <= -0.1:
        return "commot_higher_fraction"
    return "similar_or_mixed"


def write_status(dataset_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Directionality Permutation Audit",
        "",
        "## Scope",
        "",
        "This audit tests aggregated niche-flow directionality against LR-label and niche-label permutation nulls. It uses the Phase 11 flow table and does not rerun coordinate-level cell-cell edges.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}` `{row.permutation_audit_call}`: "
            f"both-null FDR<=0.1={int(row.n_both_fdr_0_1)}/{int(row.n_lr)}, "
            f"LR-label FDR<=0.1={int(row.n_lr_label_fdr_0_1)}, "
            f"niche-label FDR<=0.1={int(row.n_niche_label_fdr_0_1)}, "
            f"best={row.best_lr} frac={row.best_lr_expected_flow_fraction:.3g}, "
            f"q_lr={row.best_lr_lr_label_fdr:.3g}, q_niche={row.best_lr_niche_label_fdr:.3g}"
        )
    if not comparison.empty:
        lines += ["", "## SpatialSelf vs COMMOT", ""]
        for dataset_key, sub in comparison.groupby("dataset_key", sort=False):
            calls = sub["comparison_call"].value_counts()
            lines.append(
                f"- `{dataset_key}`: "
                + "; ".join(f"{call}={int(count)}" for call, count in calls.items())
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- These p-values are label-permutation p-values on aggregated niche-flow scores.",
        "- They are stronger than the previous non-expected-flow empirical null, but still not coordinate-level reruns or causal transport evidence.",
        "- Do not claim global SpatialSelf directionality superiority unless the dataset/method comparison table supports it.",
    ]
    (OUT_DIR / "phase12_directionality_permutation_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_directionality_permutation_audit",
        "input": "outputs/phase11/commot_style_spatial_story/trajectory_summary/phase11_signal_trajectory_niche_flow_all.csv",
        "n_permutations": N_PERMUTATIONS,
        "nulls": ["lr_label_permutation", "niche_label_permutation"],
        "outputs": [
            "phase12_directionality_permutation_observed_lr.csv",
            "phase12_directionality_lr_label_permutation_pvalues.csv",
            "phase12_directionality_niche_label_permutation_pvalues.csv",
            "phase12_directionality_permutation_lr_summary.csv",
            "phase12_directionality_permutation_dataset_summary.csv",
            "phase12_directionality_permutation_method_comparison.csv",
            "phase12_directionality_permutation_audit_status.md",
        ],
        "claim_boundary": "Aggregated niche-flow label-permutation audit, not coordinate-level edge permutation or causal transport proof.",
    }
    (OUT_DIR / "phase12_directionality_permutation_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
