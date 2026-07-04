#!/usr/bin/env python
"""Light exact-score null audit for the target-aware background-guard smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.run_phase12_spatial_specificity_permutation_audit as spatial_perm  # noqa: E402
from scripts.phase12.calibrate_phase12_guarded_spatial_specificity import (  # noqa: E402
    add_guarded_spatial_specificity_calibration,
)


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
MODEL_DIR = PHASE12_DIR / "integrated_background_target_guarded_training_smoke"
REPLICATE_ROOT = PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training" / "replicates"
OUT_DIR = PHASE12_DIR / "integrated_background_target_guarded_permutation_smoke"
DATASET_KEY = "post_xenium_colon_full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=121318)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_perm.EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded_integrated_guarded_spatial_training"
    spatial_perm.N_PERMUTATIONS = int(args.n_permutations)
    spatial_perm.RNG_SEED = int(args.seed)
    spatial_perm.add_spatial_specificity_calibration = add_guarded_spatial_specificity_calibration
    spatial_perm.score_table = guarded_score_table

    ranking = pd.read_csv(MODEL_DIR / "phase12_integrated_background_target_guarded_lr_ranking.csv")
    ranking = ranking[ranking["dataset_key"].astype(str).eq(DATASET_KEY)].sort_values("rank").reset_index(drop=True)
    result = spatial_perm.run_dataset(DATASET_KEY, ranking)
    lr_fdr = result["lr_fdr"]
    null_summary = result["null_summary"]
    if not lr_fdr.empty:
        lr_fdr["method"] = "spatialself_wavelet_integrated_background_target_guarded_permutation_smoke"
    topk = spatial_perm.summarize_topk(lr_fdr)
    status = pd.DataFrame(
        [
            {
                "dataset_key": DATASET_KEY,
                "status": "completed_smoke",
                "n_permutations_per_null": int(args.n_permutations),
                "runtime_seconds": float("nan"),
            }
        ]
    )
    dataset = spatial_perm.summarize_dataset(lr_fdr, topk, null_summary, status)
    if not dataset.empty:
        dataset["target_guarded_permutation_smoke_call"] = dataset["spatial_specificity_permutation_call"].map(
            lambda x: str(x).replace("spatial_specificity", "target_guarded")
        )
    story = spatial_perm.summarize_story(dataset)
    if not story.empty:
        story["topic"] = "integrated_background_target_guarded_permutation_smoke"
        story["claim_boundary"] = "Low-permutation exact-score smoke for target-aware background-guard path."

    lr_fdr.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_lr_fdr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv", index=False)
    null_summary.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_null_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_story.csv", index=False)
    write_status(dataset, topk, null_summary, args)
    write_manifest(args)

    print(dataset.to_string(index=False))
    print()
    print(topk[topk["top_k"].isin([10, 50, 100])].to_string(index=False))
    print()
    print(lr_fdr.head(20).to_string(index=False))


def guarded_score_table(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for col in spatial_perm.FEATURE_COLS:
        out[col] = features[col].to_numpy()
    out = add_guarded_spatial_specificity_calibration(out)
    out["method"] = "spatialself_wavelet_integrated_background_target_guarded_permutation_smoke"
    out["rank"] = out["calibrated_score"].rank(ascending=False, method="min").astype(int)
    out["strict_rank"] = out["strict_discovery_score"].rank(ascending=False, method="min").astype(int)
    out["rank_q95"] = out["calibrated_q95_score"].rank(ascending=False, method="min").astype(int)
    return out


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, null_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Integrated Background-Target Guarded Permutation Smoke",
        "",
        "## Scope",
        "",
        "Low-permutation exact-score null audit for the target-aware background-guard training smoke.",
        "This recomputes the guarded spatial specificity score under receiver, spatial-distance and LR-identity nulls.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        call = getattr(row, "target_guarded_permutation_smoke_call", row.spatial_specificity_permutation_call)
        lines.append(
            f"- `{row.dataset_key}` `{call}`: permutations/null={int(args.n_permutations)}, "
            f"LR={int(row.n_lr)}, top10 true={int(row.top10_true_lr)}, "
            f"all LR q<=0.1={int(row.all_lr_conservative_fdr_0_1)}, "
            f"best true={row.best_true_lr} q={fmt(row.best_true_lr_fdr)}, "
            f"top10 risk={fmt(row.top10_negative_or_risk_fraction)}"
        )
    lines += ["", "## Claim Boundary", "", "This is a smoke audit, not formal FDR."]
    (OUT_DIR / "phase12_integrated_background_target_guarded_permutation_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "task": "phase12_integrated_background_target_guarded_permutation_smoke",
        "dataset": DATASET_KEY,
        "n_permutations_per_null": int(args.n_permutations),
        "null_types": list(spatial_perm.NULL_TYPES),
        "inputs": [
            rel(MODEL_DIR / "phase12_integrated_background_target_guarded_lr_ranking.csv"),
            rel(REPLICATE_ROOT / DATASET_KEY / "candidate_interactions.csv"),
            rel(REPLICATE_ROOT / DATASET_KEY / "lr_pairs.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_lr_fdr.csv"),
            rel(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_topk_summary.csv"),
            rel(OUT_DIR / "phase12_integrated_background_target_guarded_permutation_null_summary.csv"),
        ],
        "claim_boundary": "Low-permutation exact-score smoke only.",
    }
    (OUT_DIR / "phase12_integrated_background_target_guarded_permutation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(numeric):
        return "NA"
    return f"{numeric:.3g}"


if __name__ == "__main__":
    main()
