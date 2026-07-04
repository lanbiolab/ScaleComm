#!/usr/bin/env python
"""LR-standardized all-null FDR diagnostic across expanded-atlas variants.

The original broad atlas FDR pools raw calibrated scores from every LR under
each null. That is deliberately conservative, but it can be dominated by LR
pairs whose null score baseline is intrinsically high. This diagnostic asks a
different question: after each LR is standardized against its own null score
distribution, do true LR pairs still stand out in a whole-atlas BH procedure?

This is a calibration diagnostic, not a replacement for the original pooled
spatial-distance FDR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_expanded_atlas_permutation_audit as perm  # noqa: E402
from scripts.phase12.calibrate_phase12_expanded_atlas_background_suppressed import (  # noqa: E402
    add_background_suppression_calibration,
)
from scripts.phase12.calibrate_phase12_expanded_atlas_ranking import add_calibration as add_base_calibration  # noqa: E402


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "lr_standardized_null_fdr"
N_PERMUTATIONS = perm.N_PERMUTATIONS
RNG_SEED = perm.RNG_SEED + 12012
NULL_TYPES = perm.NULL_TYPES

METHOD_SPECS = {
    "main_calibrated": {
        "ranking": PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded/replicates",
        "calibration": "base",
    },
    "anchor_background_guarded_training": {
        "ranking": PHASE12_DIR
        / "atlas_model_expanded_anchor_background_guarded_training/phase12_anchor_background_guarded_training_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded_anchor_background_guarded_training/replicates",
        "calibration": "background",
    },
    "anchor_background_range_guarded_training": {
        "ranking": PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded_anchor_background_range_guarded_training/replicates",
        "calibration": "background",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_rows = []
    null_rows = []
    for method, spec in METHOD_SPECS.items():
        ranking_path = Path(spec["ranking"])
        if not ranking_path.exists():
            continue
        set_calibration(str(spec["calibration"]))
        ranking = pd.read_csv(ranking_path)
        for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
            dataset_ranking = ranking[ranking["dataset_key"].astype(str).eq(dataset_key)].sort_values("rank")
            replicate_dir = Path(spec["replicates"]) / dataset_key
            result = run_dataset(method, dataset_key, dataset_ranking.reset_index(drop=True), replicate_dir)
            lr_rows.append(result["lr"])
            null_rows.append(result["null"])

    lr = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    dataset = summarize_dataset(lr)
    method = summarize_method(dataset)
    story = summarize_story(method)

    lr.to_csv(OUT_DIR / "phase12_lr_standardized_null_fdr_lr.csv", index=False)
    null.to_csv(OUT_DIR / "phase12_lr_standardized_null_fdr_null_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_lr_standardized_null_fdr_dataset_summary.csv", index=False)
    method.to_csv(OUT_DIR / "phase12_lr_standardized_null_fdr_method_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_lr_standardized_null_fdr_story.csv", index=False)
    write_status(method, dataset, story)
    write_manifest()

    print(method.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def set_calibration(kind: str) -> None:
    if kind == "background":
        perm.add_calibration = add_background_suppression_calibration
    else:
        perm.add_calibration = add_base_calibration


def run_dataset(method: str, dataset_key: str, ranking: pd.DataFrame, replicate_dir: Path) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = perm.prepare_base_table(ranking, lr_pairs)
    arrays = perm.prepare_candidate_arrays(candidates, base)
    rng = np.random.default_rng(RNG_SEED + perm.stable_dataset_offset(f"{method}:{dataset_key}"))

    observed = base["calibrated_score"].to_numpy(dtype=float)
    strict_observed = base["strict_discovery_score"].to_numpy(dtype=float)
    null_summary = []
    p_by_null = {}
    strict_p_by_null = {}
    z_by_null = {}
    strict_z_by_null = {}
    for null_type in NULL_TYPES:
        scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        strict_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            null_table = perm.build_null_table(base, arrays, null_type, rng)
            scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
            strict_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)
        z_result = standardized_global_p(observed, scores)
        strict_z_result = standardized_global_p(strict_observed, strict_scores)
        p_by_null[null_type] = z_result["p"]
        strict_p_by_null[null_type] = strict_z_result["p"]
        z_by_null[null_type] = z_result["z_observed"]
        strict_z_by_null[null_type] = strict_z_result["z_observed"]
        null_summary.append(
            {
                "method": method,
                "dataset_key": dataset_key,
                "null_type": null_type,
                "n_permutations": N_PERMUTATIONS,
                "n_lr": int(len(base)),
                "mean_null_sd": float(np.nanmean(z_result["sd"])),
                "median_null_sd": float(np.nanmedian(z_result["sd"])),
                "pooled_standardized_null_q95": float(np.nanquantile(z_result["z_null"].ravel(), 0.95)),
                "pooled_standardized_null_q99": float(np.nanquantile(z_result["z_null"].ravel(), 0.99)),
                "max_observed_standardized_z": float(np.nanmax(z_result["z_observed"])),
                "strict_pooled_standardized_null_q95": float(
                    np.nanquantile(strict_z_result["z_null"].ravel(), 0.95)
                ),
                "strict_pooled_standardized_null_q99": float(
                    np.nanquantile(strict_z_result["z_null"].ravel(), 0.99)
                ),
                "strict_max_observed_standardized_z": float(np.nanmax(strict_z_result["z_observed"])),
            }
        )

    lr = base.copy()
    lr["method"] = method
    lr["dataset_key"] = dataset_key
    lr["lr_standardized_conservative_p"] = np.maximum.reduce([p_by_null[null_type] for null_type in NULL_TYPES])
    lr["lr_standardized_conservative_bh_fdr"] = bh(lr["lr_standardized_conservative_p"].to_numpy(dtype=float))
    lr["strict_lr_standardized_conservative_p"] = np.maximum.reduce(
        [strict_p_by_null[null_type] for null_type in NULL_TYPES]
    )
    lr["strict_lr_standardized_conservative_bh_fdr"] = bh(
        lr["strict_lr_standardized_conservative_p"].to_numpy(dtype=float)
    )
    for null_type in NULL_TYPES:
        prefix = short_null(null_type)
        lr[f"{prefix}_standardized_z"] = z_by_null[null_type]
        lr[f"{prefix}_standardized_p"] = p_by_null[null_type]
        lr[f"strict_{prefix}_standardized_z"] = strict_z_by_null[null_type]
        lr[f"strict_{prefix}_standardized_p"] = strict_p_by_null[null_type]
    lr["lr_standardized_call"] = classify_lr_table(lr, "lr_standardized_conservative_bh_fdr")
    lr["strict_lr_standardized_call"] = classify_lr_table(lr, "strict_lr_standardized_conservative_bh_fdr")
    return {"lr": lr.sort_values(["rank", "lr_pair"]).reset_index(drop=True), "null": pd.DataFrame(null_summary)}


def standardized_global_p(observed: np.ndarray, null_scores: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(null_scores, axis=0)
    sd = np.nanstd(null_scores, axis=0, ddof=1)
    fallback = np.nanmedian(sd[sd > 0]) if np.any(sd > 0) else 1.0
    sd = np.where(sd > 1e-12, sd, fallback)
    z_observed = (observed - mean) / sd
    z_null = (null_scores - mean[None, :]) / sd[None, :]
    pooled = z_null.ravel()
    p = np.asarray([(1.0 + np.sum(pooled >= z)) / (1.0 + len(pooled)) for z in z_observed], dtype=float)
    return {"p": p, "z_observed": z_observed, "z_null": z_null, "sd": sd}


def classify_lr_table(table: pd.DataFrame, q_col: str) -> list[str]:
    q = pd.to_numeric(table[q_col], errors="coerce")
    true = table["true_lr"].astype(bool)
    control = control_mask(table)
    out = []
    for is_true, is_control, q_value in zip(true, control, q):
        if pd.notna(q_value) and q_value <= 0.1 and is_true:
            out.append("true_lr_standardized_all_null_supported")
        elif pd.notna(q_value) and q_value <= 0.1 and is_control:
            out.append("control_standardized_all_null_leakage")
        elif pd.notna(q_value) and q_value <= 0.1:
            out.append("background_standardized_all_null_supported")
        else:
            out.append("not_supported")
    return out


def summarize_dataset(lr: pd.DataFrame) -> pd.DataFrame:
    if lr.empty:
        return pd.DataFrame()
    rows = []
    for (method, dataset_key), sub in lr.groupby(["method", "dataset_key"], sort=False):
        true = sub["true_lr"].astype(bool)
        control = control_mask(sub)
        q = pd.to_numeric(sub["lr_standardized_conservative_bh_fdr"], errors="coerce").fillna(1.0).le(0.1)
        strict_q = pd.to_numeric(sub["strict_lr_standardized_conservative_bh_fdr"], errors="coerce").fillna(1.0).le(0.1)
        rows.append(
            {
                "method": method,
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "standardized_all_lr_q_0_1": int(q.sum()),
                "standardized_true_q_0_1": int((true & q).sum()),
                "standardized_control_q_0_1": int((control & q).sum()),
                "standardized_background_q_0_1": int(((~true) & (~control) & q).sum()),
                "strict_standardized_all_lr_q_0_1": int(strict_q.sum()),
                "strict_standardized_true_q_0_1": int((true & strict_q).sum()),
                "strict_standardized_control_q_0_1": int((control & strict_q).sum()),
                "best_true_standardized_q": min_or_nan(sub.loc[true, "lr_standardized_conservative_bh_fdr"]),
                "best_true_strict_standardized_q": min_or_nan(
                    sub.loc[true, "strict_lr_standardized_conservative_bh_fdr"]
                ),
                "best_true_lr": best_true_lr(sub, "lr_standardized_conservative_bh_fdr"),
                "dataset_call": dataset_call(sub, q, strict_q, true, control),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows = []
    for method, sub in dataset.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_datasets": int(len(sub)),
                "standardized_true_q_0_1": int(sub["standardized_true_q_0_1"].sum()),
                "standardized_control_q_0_1": int(sub["standardized_control_q_0_1"].sum()),
                "strict_standardized_true_q_0_1": int(sub["strict_standardized_true_q_0_1"].sum()),
                "strict_standardized_control_q_0_1": int(sub["strict_standardized_control_q_0_1"].sum()),
                "best_true_standardized_q": float(pd.to_numeric(sub["best_true_standardized_q"], errors="coerce").min()),
                "n_clean_supported_datasets": int(
                    sub["dataset_call"].astype(str).isin(
                        ["standardized_true_support_clean_controls", "strict_standardized_true_support_clean_controls"]
                    ).sum()
                ),
                "method_call": method_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_standardized_true_q_0_1", "standardized_control_q_0_1", "standardized_true_q_0_1"],
        ascending=[False, True, False],
    )


def summarize_story(method: pd.DataFrame) -> pd.DataFrame:
    if method.empty:
        return pd.DataFrame()
    best = method.iloc[0]
    if int(best["strict_standardized_true_q_0_1"]) > 0 and int(best["strict_standardized_control_q_0_1"]) == 0:
        call = "lr_standardized_all_null_supportive_with_caveat"
    elif int(best["standardized_true_q_0_1"]) > 0:
        call = "lr_standardized_support_contaminated_or_non_strict"
    else:
        call = "lr_standardized_all_null_not_supportive"
    return pd.DataFrame(
        [
            {
                "topic": "lr_standardized_null_fdr",
                "overall_call": call,
                "n_methods": int(len(method)),
                "best_method": best["method"],
                "best_method_strict_true_q_0_1": int(best["strict_standardized_true_q_0_1"]),
                "best_method_strict_control_q_0_1": int(best["strict_standardized_control_q_0_1"]),
                "best_method_true_q_0_1": int(best["standardized_true_q_0_1"]),
                "best_method_control_q_0_1": int(best["standardized_control_q_0_1"]),
                "claim_boundary": (
                    "LR-standardized all-null FDR is a diagnostic alternative to raw pooled null scoring. "
                    "It can support a calibration direction, but it does not replace the original conservative "
                    "pooled whole-atlas FDR unless explicitly adopted and justified as the discovery procedure."
                ),
            }
        ]
    )


def dataset_call(sub: pd.DataFrame, q: pd.Series, strict_q: pd.Series, true: pd.Series, control: pd.Series) -> str:
    true_hits = int((true & q).sum())
    control_hits = int((control & q).sum())
    strict_true_hits = int((true & strict_q).sum())
    strict_control_hits = int((control & strict_q).sum())
    if strict_true_hits > 0 and strict_control_hits == 0:
        return "strict_standardized_true_support_clean_controls"
    if true_hits > 0 and control_hits == 0:
        return "standardized_true_support_clean_controls"
    if true_hits > 0:
        return "standardized_support_with_control_caveat"
    return "not_supported"


def method_call(sub: pd.DataFrame) -> str:
    strict_true = int(sub["strict_standardized_true_q_0_1"].sum())
    strict_control = int(sub["strict_standardized_control_q_0_1"].sum())
    true = int(sub["standardized_true_q_0_1"].sum())
    control = int(sub["standardized_control_q_0_1"].sum())
    if strict_true > 0 and strict_control == 0:
        return "strict_standardized_support_clean_controls"
    if true > 0 and control == 0:
        return "standardized_support_clean_controls"
    if true > 0 or strict_true > 0:
        return "standardized_support_with_control_caveat"
    return "not_supported"


def control_mask(table: pd.DataFrame) -> pd.Series:
    true = table["true_lr"].astype(bool)
    true_receptors = set(table.loc[true, "receptor"].astype(str))
    true_ligands = set(table.loc[true, "ligand"].astype(str))
    return (
        table["false_positive_high_expression"].astype(bool)
        | (~true & table["receptor"].astype(str).isin(true_receptors))
        | (~true & table["ligand"].astype(str).isin(true_ligands))
    )


def best_true_lr(table: pd.DataFrame, q_col: str) -> str:
    true = table[table["true_lr"].astype(bool)].sort_values([q_col, "rank"])
    return str(true.iloc[0]["lr_pair"]) if len(true) else "none"


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    values = p[idx]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[idx[order]] = np.minimum(adjusted, 1.0)
    return out


def short_null(null_type: str) -> str:
    return {
        "receiver_permutation": "receiver",
        "spatial_distance_permutation": "spatial_distance",
        "lr_identity_permutation": "lr_identity",
    }[null_type]


def write_status(method: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 LR-Standardized Null FDR",
        "",
        "## Scope",
        "",
        "Each LR is standardized against its own receiver, spatial-distance and LR-identity null score distribution before whole-atlas BH.",
        "This tests whether broad pooled FDR is blocked by LR-specific null baselines.",
        "It is a diagnostic alternative, not a replacement for the original pooled whole-atlas FDR.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.method}` `{row.method_call}`: strict true={int(row.strict_standardized_true_q_0_1)}, "
            f"strict controls={int(row.strict_standardized_control_q_0_1)}, "
            f"true={int(row.standardized_true_q_0_1)}, controls={int(row.standardized_control_q_0_1)}, "
            f"best true q={fmt(row.best_true_standardized_q)}."
        )
    lines += ["", "## Dataset Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.method}` / `{row.dataset_key}` `{row.dataset_call}`: "
            f"strict true={int(row.strict_standardized_true_q_0_1)}, "
            f"strict controls={int(row.strict_standardized_control_q_0_1)}, "
            f"true={int(row.standardized_true_q_0_1)}, controls={int(row.standardized_control_q_0_1)}, "
            f"best={row.best_true_lr} q={fmt(row.best_true_standardized_q)}."
        )
    if not story.empty:
        first = story.iloc[0]
        lines += [
            "",
            "## Decision",
            "",
            f"- Overall: `{first['overall_call']}`.",
            f"- Best method: `{first['best_method']}`.",
            "- Manuscript boundary: use as calibration diagnostic only unless adopted as a formal discovery procedure.",
        ]
    (OUT_DIR / "phase12_lr_standardized_null_fdr_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_lr_standardized_null_fdr",
        "n_permutations": N_PERMUTATIONS,
        "null_types": list(NULL_TYPES),
        "methods": list(METHOD_SPECS),
        "outputs": [
            "phase12_lr_standardized_null_fdr_lr.csv",
            "phase12_lr_standardized_null_fdr_null_summary.csv",
            "phase12_lr_standardized_null_fdr_dataset_summary.csv",
            "phase12_lr_standardized_null_fdr_method_summary.csv",
            "phase12_lr_standardized_null_fdr_story.csv",
        ],
        "claim_boundary": "LR-standardized all-null FDR diagnostic; not original pooled whole-atlas FDR closure.",
    }
    (OUT_DIR / "phase12_lr_standardized_null_fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
