#!/usr/bin/env python
"""Matched-spatial and range-weighted FDR diagnostics across model variants."""

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
import scripts.phase12.summarize_phase12_matched_spatial_atlas_fdr as matched_mod  # noqa: E402
import scripts.phase12.summarize_phase12_range_weighted_atlas_fdr as range_weighted_mod  # noqa: E402
from scripts.phase12.calibrate_phase12_expanded_atlas_ranking import add_calibration as add_base_calibration  # noqa: E402
from scripts.phase12.calibrate_phase12_expanded_atlas_background_suppressed import (  # noqa: E402
    add_background_suppression_calibration,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "variant_matched_spatial_fdr"
N_PERMUTATIONS = perm.N_PERMUTATIONS
RNG_SEED = perm.RNG_SEED + 9090

METHOD_SPECS = {
    "main_calibrated": {
        "ranking": PHASE12_DIR / "atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded/replicates",
        "permutation_lr": PHASE12_DIR
        / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
        "calibration": "base",
    },
    "anchor_background_suppressed": {
        "ranking": PHASE12_DIR
        / "atlas_model_expanded_anchor_background_suppressed/phase12_anchor_background_suppressed_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded_spatial_null_anchor_preserving/replicates",
        "permutation_lr": PHASE12_DIR
        / "anchor_background_suppressed_permutation_audit/phase12_anchor_background_suppressed_permutation_lr_fdr.csv",
        "calibration": "background",
    },
    "anchor_background_guarded_training": {
        "ranking": PHASE12_DIR
        / "atlas_model_expanded_anchor_background_guarded_training/phase12_anchor_background_guarded_training_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded_anchor_background_guarded_training/replicates",
        "permutation_lr": PHASE12_DIR
        / "anchor_background_guarded_training_permutation_audit/"
        / "phase12_anchor_background_guarded_training_permutation_lr_fdr.csv",
        "calibration": "background",
    },
    "anchor_background_range_guarded_training": {
        "ranking": PHASE12_DIR
        / "atlas_model_expanded_anchor_background_range_guarded_training/"
        / "phase12_anchor_background_range_guarded_training_lr_ranking.csv",
        "replicates": PHASE12_DIR / "atlas_model_expanded_anchor_background_range_guarded_training/replicates",
        "permutation_lr": PHASE12_DIR
        / "anchor_background_range_guarded_training_permutation_audit/"
        / "phase12_anchor_background_range_guarded_training_permutation_lr_fdr.csv",
        "calibration": "background",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matched_rows = []
    match_rows = []
    weighted_rows = []
    null_rows = []
    for method, spec in METHOD_SPECS.items():
        if not Path(spec["ranking"]).exists() or not Path(spec["permutation_lr"]).exists():
            continue
        set_calibration(str(spec["calibration"]))
        result = run_method(method, spec)
        matched_rows.append(result["matched"])
        match_rows.append(result["matches"])
        weighted_rows.append(result["weighted"])
        null_rows.append(result["null"])

    matched = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()
    matches = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    weighted = pd.concat(weighted_rows, ignore_index=True) if weighted_rows else pd.DataFrame()
    null = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    dataset = summarize_dataset(matched, weighted)
    method = summarize_method(dataset)
    story = summarize_story(method)

    matched.to_csv(OUT_DIR / "phase12_variant_matched_spatial_lr.csv", index=False)
    matches.to_csv(OUT_DIR / "phase12_variant_matched_spatial_matches.csv", index=False)
    weighted.to_csv(OUT_DIR / "phase12_variant_range_weighted_lr.csv", index=False)
    null.to_csv(OUT_DIR / "phase12_variant_matched_spatial_null_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_variant_matched_spatial_dataset_summary.csv", index=False)
    method.to_csv(OUT_DIR / "phase12_variant_matched_spatial_method_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_variant_matched_spatial_story.csv", index=False)
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


def run_method(method: str, spec: dict[str, object]) -> dict[str, pd.DataFrame]:
    ranking = pd.read_csv(Path(spec["ranking"]))
    original = pd.read_csv(Path(spec["permutation_lr"]))
    matched_tables = []
    match_tables = []
    null_tables = []
    for dataset_key in sorted(ranking["dataset_key"].dropna().astype(str).unique()):
        dataset_ranking = ranking[ranking["dataset_key"].astype(str).eq(dataset_key)].sort_values("rank").reset_index(drop=True)
        dataset_original = original[original["dataset_key"].astype(str).eq(dataset_key)].copy()
        replicate_dir = Path(spec["replicates"]) / dataset_key
        result = run_dataset(method, dataset_key, dataset_ranking, dataset_original, replicate_dir)
        matched_tables.append(result["matched"])
        match_tables.append(result["matches"])
        null_tables.append(result["null"])
    matched = pd.concat(matched_tables, ignore_index=True) if matched_tables else pd.DataFrame()
    matches = pd.concat(match_tables, ignore_index=True) if match_tables else pd.DataFrame()
    null = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()
    weighted = build_weighted(method, ranking, matched)
    return {"matched": matched, "matches": matches, "weighted": weighted, "null": null}


def run_dataset(
    method: str,
    dataset_key: str,
    ranking: pd.DataFrame,
    original: pd.DataFrame,
    replicate_dir: Path,
) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    base = perm.prepare_base_table(ranking, lr_pairs)
    arrays = perm.prepare_candidate_arrays(candidates, base)
    rng = np.random.default_rng(RNG_SEED + perm.stable_dataset_offset(f"{method}:{dataset_key}"))
    spatial_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    strict_spatial_scores = np.zeros((N_PERMUTATIONS, len(base)), dtype=float)
    for perm_idx in range(N_PERMUTATIONS):
        null_table = perm.build_null_table(base, arrays, "spatial_distance_permutation", rng)
        spatial_scores[perm_idx, :] = null_table["calibrated_score"].to_numpy(dtype=float)
        strict_spatial_scores[perm_idx, :] = null_table["strict_discovery_score"].to_numpy(dtype=float)
    lr, matches = matched_mod.build_lr_table(dataset_key, base, original, spatial_scores, strict_spatial_scores)
    lr["method"] = method
    matches["method"] = method
    flat = spatial_scores.ravel()
    null = pd.DataFrame(
        [
            {
                "method": method,
                "dataset_key": dataset_key,
                "n_permutations": N_PERMUTATIONS,
                "n_lr": int(len(base)),
                "pooled_spatial_null_q95": float(np.nanquantile(flat, 0.95)),
                "pooled_spatial_null_q99": float(np.nanquantile(flat, 0.99)),
                "matched_null_median_n_lr": float(lr["matched_null_n_lr"].median()),
                "matched_null_min_n_lr": int(lr["matched_null_n_lr"].min()),
            }
        ]
    )
    return {"matched": lr, "matches": matches, "null": null}


def build_weighted(method: str, ranking: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()
    source = matched.drop(columns=["method"], errors="ignore").copy()
    table = range_weighted_mod.build_table(ranking, source)
    table["method"] = method
    return table


def summarize_dataset(matched: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if matched.empty:
        return pd.DataFrame()
    weighted_index = {
        (method, dataset_key): sub
        for (method, dataset_key), sub in weighted.groupby(["method", "dataset_key"], sort=False)
    } if not weighted.empty else {}
    for (method, dataset_key), sub in matched.groupby(["method", "dataset_key"], sort=False):
        true = sub["true_lr"].astype(bool)
        control = control_mask(sub)
        w = weighted_index.get((method, dataset_key), pd.DataFrame())
        rows.append(
            {
                "method": method,
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "matched_spatial_true_q_0_1": int((true & sub["matched_spatial_bh_fdr"].le(0.1)).sum()),
                "matched_all_true_q_0_1": int((true & sub["matched_all_null_bh_fdr"].le(0.1)).sum()),
                "strict_matched_all_true_q_0_1": int((true & sub["strict_matched_all_null_bh_fdr"].le(0.1)).sum()),
                "matched_all_control_q_0_1": int((control & sub["matched_all_null_bh_fdr"].le(0.1)).sum()),
                "best_true_matched_all_q": min_or_nan(sub.loc[true, "matched_all_null_bh_fdr"]),
                "best_true_matched_spatial_q": min_or_nan(sub.loc[true, "matched_spatial_bh_fdr"]),
                "weighted_true_q_0_1": weighted_count(w, "true"),
                "weighted_control_q_0_1": weighted_count(w, "control"),
                "best_true_weighted_q": weighted_best_true_q(w),
                "dataset_call": dataset_call(sub, w),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if dataset.empty:
        return pd.DataFrame()
    for method, sub in dataset.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_datasets": int(len(sub)),
                "matched_all_true_q_0_1": int(pd.to_numeric(sub["matched_all_true_q_0_1"], errors="coerce").sum()),
                "strict_matched_all_true_q_0_1": int(
                    pd.to_numeric(sub["strict_matched_all_true_q_0_1"], errors="coerce").sum()
                ),
                "matched_all_control_q_0_1": int(pd.to_numeric(sub["matched_all_control_q_0_1"], errors="coerce").sum()),
                "weighted_true_q_0_1": int(pd.to_numeric(sub["weighted_true_q_0_1"], errors="coerce").sum()),
                "weighted_control_q_0_1": int(pd.to_numeric(sub["weighted_control_q_0_1"], errors="coerce").sum()),
                "best_true_matched_all_q": float(pd.to_numeric(sub["best_true_matched_all_q"], errors="coerce").min()),
                "best_true_weighted_q": float(pd.to_numeric(sub["best_true_weighted_q"], errors="coerce").min()),
                "method_call": method_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_matched_all_true_q_0_1", "matched_all_true_q_0_1", "weighted_control_q_0_1"],
        ascending=[False, False, True],
    )


def summarize_story(method: pd.DataFrame) -> pd.DataFrame:
    if method.empty:
        return pd.DataFrame()
    best = method.iloc[0]
    return pd.DataFrame(
        [
            {
                "topic": "variant_matched_spatial_fdr",
                "overall_call": "variant_matched_spatial_supportive_with_caveat",
                "n_methods": int(len(method)),
                "best_method": best["method"],
                "best_method_strict_matched_true_q_0_1": int(best["strict_matched_all_true_q_0_1"]),
                "best_method_matched_true_q_0_1": int(best["matched_all_true_q_0_1"]),
                "best_method_weighted_true_q_0_1": int(best["weighted_true_q_0_1"]),
                "best_method_weighted_control_q_0_1": int(best["weighted_control_q_0_1"]),
                "claim_boundary": (
                    "Matched-spatial and range-weighted diagnostics can guide spatial-null calibration across variants, "
                    "but they remain post-hoc diagnostics and do not replace broad pooled whole-atlas FDR."
                ),
            }
        ]
    )


def control_mask(table: pd.DataFrame) -> pd.Series:
    return (
        table["false_positive_high_expression"].astype(bool)
        | (~table["true_lr"].astype(bool) & table["receptor"].astype(str).isin(set(table.loc[table["true_lr"].astype(bool), "receptor"].astype(str))))
        | (~table["true_lr"].astype(bool) & table["ligand"].astype(str).isin(set(table.loc[table["true_lr"].astype(bool), "ligand"].astype(str))))
    )


def weighted_count(table: pd.DataFrame, kind: str) -> int:
    if table.empty:
        return 0
    true = table["true_lr"].astype(bool)
    control = table["primary_control_decoy"].astype(bool) if "primary_control_decoy" in table else control_mask(table)
    q = pd.to_numeric(table["weighted_matched_all_null_bh_fdr"], errors="coerce").fillna(1.0).le(0.1)
    if kind == "true":
        return int((true & q).sum())
    return int((control & q).sum())


def weighted_best_true_q(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    true = table["true_lr"].astype(bool)
    return min_or_nan(table.loc[true, "weighted_matched_all_null_bh_fdr"])


def dataset_call(matched: pd.DataFrame, weighted: pd.DataFrame) -> str:
    true = matched["true_lr"].astype(bool)
    control = control_mask(matched)
    strict_true = int((true & matched["strict_matched_all_null_bh_fdr"].le(0.1)).sum())
    true_hits = int((true & matched["matched_all_null_bh_fdr"].le(0.1)).sum())
    control_hits = int((control & matched["matched_all_null_bh_fdr"].le(0.1)).sum())
    weighted_true = weighted_count(weighted, "true")
    weighted_control = weighted_count(weighted, "control")
    if strict_true > 0 and control_hits == 0:
        return "strict_matched_support_clean_controls"
    if true_hits > 0 and control_hits == 0:
        return "matched_support_clean_controls"
    if weighted_true > 0 and weighted_control == 0:
        return "weighted_support_clean_controls"
    if true_hits > 0 or weighted_true > 0:
        return "support_with_control_caveat"
    return "not_supported"


def method_call(sub: pd.DataFrame) -> str:
    strict_true = int(pd.to_numeric(sub["strict_matched_all_true_q_0_1"], errors="coerce").sum())
    matched_true = int(pd.to_numeric(sub["matched_all_true_q_0_1"], errors="coerce").sum())
    controls = int(pd.to_numeric(sub["matched_all_control_q_0_1"], errors="coerce").sum())
    weighted_true = int(pd.to_numeric(sub["weighted_true_q_0_1"], errors="coerce").sum())
    weighted_controls = int(pd.to_numeric(sub["weighted_control_q_0_1"], errors="coerce").sum())
    if strict_true > 0 and controls == 0:
        return "strict_matched_support_clean_controls"
    if matched_true > 0 and controls == 0:
        return "matched_support_clean_controls"
    if weighted_true > 0 and weighted_controls == 0:
        return "weighted_support_clean_controls"
    if matched_true > 0 or weighted_true > 0:
        return "support_with_control_caveat"
    return "not_supported"


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def write_status(method: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Variant Matched-Spatial FDR",
        "",
        "## Scope",
        "",
        "This extends matched-spatial and range-weighted diagnostics from the main calibrated atlas to anchor/background model variants.",
        "It is post-hoc and does not replace broad pooled whole-atlas FDR.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.method}` `{row.method_call}`: strict matched true={int(row.strict_matched_all_true_q_0_1)}, "
            f"matched true={int(row.matched_all_true_q_0_1)}, matched controls={int(row.matched_all_control_q_0_1)}, "
            f"weighted true={int(row.weighted_true_q_0_1)}, weighted controls={int(row.weighted_control_q_0_1)}, "
            f"best matched q={fmt(row.best_true_matched_all_q)}."
        )
    lines += ["", "## Dataset Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.method}` / `{row.dataset_key}` `{row.dataset_call}`: "
            f"matched true={int(row.matched_all_true_q_0_1)}, strict true={int(row.strict_matched_all_true_q_0_1)}, "
            f"controls={int(row.matched_all_control_q_0_1)}, weighted true={int(row.weighted_true_q_0_1)}, "
            f"weighted controls={int(row.weighted_control_q_0_1)}."
        )
    if not story.empty:
        first = story.iloc[0]
        lines += [
            "",
            "## Decision",
            "",
            f"- Overall: `{first['overall_call']}`.",
            f"- Best method: `{first['best_method']}`.",
            "- Manuscript boundary: use as calibration direction only, not broad FDR closure.",
        ]
    (OUT_DIR / "phase12_variant_matched_spatial_fdr_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_variant_matched_spatial_fdr",
        "n_permutations": N_PERMUTATIONS,
        "methods": list(METHOD_SPECS),
        "outputs": [
            "phase12_variant_matched_spatial_lr.csv",
            "phase12_variant_matched_spatial_matches.csv",
            "phase12_variant_range_weighted_lr.csv",
            "phase12_variant_matched_spatial_dataset_summary.csv",
            "phase12_variant_matched_spatial_method_summary.csv",
            "phase12_variant_matched_spatial_story.csv",
        ],
        "claim_boundary": "Post-hoc matched-spatial/range-weighted variant diagnostic; not broad pooled FDR closure.",
    }
    (OUT_DIR / "phase12_variant_matched_spatial_fdr_manifest.json").write_text(
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
