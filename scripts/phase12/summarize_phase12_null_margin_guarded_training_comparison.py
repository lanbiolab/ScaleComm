#!/usr/bin/env python
"""Summarize null-margin guarded training ranking and exact-score audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
MODEL_DIR = PHASE12_DIR / "atlas_model_expanded_null_margin_guarded_training"
PERM_DIR = PHASE12_DIR / "null_margin_guarded_training_permutation_audit"
OUT_DIR = PHASE12_DIR / "null_margin_guarded_training_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(MODEL_DIR / "phase12_null_margin_guarded_training_dataset_summary.csv")
    comparison = pd.read_csv(MODEL_DIR / "phase12_null_margin_guarded_training_comparison.csv")
    perm = pd.read_csv(PERM_DIR / "phase12_null_margin_guarded_training_permutation_dataset_summary.csv")
    dataset = build_dataset(model, comparison, perm)
    story = build_story(dataset)
    dataset.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_dataset_comparison.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_null_margin_guarded_training_story.csv", index=False)
    write_status(dataset, story)
    write_manifest()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_dataset(model: pd.DataFrame, comparison: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_key",
        "top10_true_lr",
        "top10_negative_or_risk",
        "top10_control_panel_empirical_fdr",
        "top20_broad_other_lr",
        "top100_true_lr",
        "hard_negative_best_comparison_beats",
        "null_margin_guarded_training_call",
    ]
    out = model[[col for col in cols if col in model.columns]].copy()
    comp_cols = [
        "dataset_key",
        "delta_top10_true_lr_vs_main",
        "delta_top10_negative_or_risk_vs_main",
        "delta_top10_true_lr_vs_anchor_bg",
        "delta_top10_negative_or_risk_vs_anchor_bg",
        "comparison_call",
    ]
    out = out.merge(comparison[[col for col in comp_cols if col in comparison.columns]], on="dataset_key", how="left")
    perm_cols = [
        "dataset_key",
        "top10_conservative_fdr_0_1",
        "top50_conservative_fdr_0_1",
        "all_lr_conservative_fdr_0_1",
        "all_true_lr_conservative_fdr_0_1",
        "best_true_lr",
        "best_true_lr_fdr",
        "top10_negative_or_risk_fraction",
        "null_margin_guarded_permutation_call",
    ]
    out = out.merge(perm[[col for col in perm_cols if col in perm.columns]], on="dataset_key", how="left")
    out["dataset_decision"] = [dataset_decision(row) for row in out.itertuples(index=False)]
    return out


def build_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n = len(dataset)
    top10_improved = int(pd.to_numeric(dataset["delta_top10_true_lr_vs_main"], errors="coerce").fillna(0).gt(0).sum())
    no_added_risk = int(pd.to_numeric(dataset["delta_top10_negative_or_risk_vs_main"], errors="coerce").fillna(0).le(0).sum())
    clean_top10 = int(pd.to_numeric(dataset["top10_negative_or_risk"], errors="coerce").fillna(999).eq(0).sum())
    hard_clean = int(pd.to_numeric(dataset["hard_negative_best_comparison_beats"], errors="coerce").fillna(999).eq(0).sum())
    fdr_closed = int(pd.to_numeric(dataset["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).gt(0).sum())
    if fdr_closed > 0:
        call = "null_margin_guarded_training_fdr_candidate"
    elif top10_improved == n and hard_clean == n:
        call = "null_margin_guarded_training_improves_ranking_not_fdr"
    else:
        call = "null_margin_guarded_training_mixed_not_fdr"
    return pd.DataFrame(
        [
            {
                "topic": "null_margin_guarded_training",
                "overall_call": call,
                "n_datasets": n,
                "n_top10_true_improved_vs_main": top10_improved,
                "n_no_added_top10_risk_vs_main": no_added_risk,
                "n_clean_top10_risk": clean_top10,
                "n_hard_negative_best_beat_clean": hard_clean,
                "n_fdr_closed_datasets": fdr_closed,
                "total_true_q_0_1": int(
                    pd.to_numeric(dataset["all_true_lr_conservative_fdr_0_1"], errors="coerce").fillna(0).sum()
                ),
                "claim_boundary": (
                    "Stronger null-margin training improves top-k ranking and hard-negative suppression, "
                    "but exact-score broad whole-atlas FDR remains unclosed."
                ),
            }
        ]
    )


def dataset_decision(row: object) -> str:
    true_hits = int(getattr(row, "all_true_lr_conservative_fdr_0_1", 0))
    top10_true = int(getattr(row, "top10_true_lr", 0))
    top10_risk = int(getattr(row, "top10_negative_or_risk", 999))
    hard_beats = int(getattr(row, "hard_negative_best_comparison_beats", 999))
    if true_hits > 0 and top10_risk == 0:
        return "candidate_fdr_support_clean_top10"
    if top10_true >= 5 and top10_risk == 0 and hard_beats == 0:
        return "strong_ranking_support_fdr_not_closed"
    if top10_true >= 3 and hard_beats == 0:
        return "ranking_support_with_control_caveat_fdr_not_closed"
    return "mixed_or_negative_result"


def write_status(dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    first = story.iloc[0]
    lines = [
        "# Phase 12 Null-Margin Guarded Training Comparison",
        "",
        f"- Overall: `{first.overall_call}`.",
        f"- Top10 true improved vs main: {int(first.n_top10_true_improved_vs_main)}/{int(first.n_datasets)}.",
        f"- Hard-negative best-beat clean: {int(first.n_hard_negative_best_beat_clean)}/{int(first.n_datasets)}.",
        f"- FDR-closed datasets: {int(first.n_fdr_closed_datasets)}/{int(first.n_datasets)}.",
        "",
        "## Dataset Decisions",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_decision}`: top10 true={int(row.top10_true_lr)}, "
            f"top10 risk={int(row.top10_negative_or_risk)}, true q<=0.1={int(row.all_true_lr_conservative_fdr_0_1)}, "
            f"best true={row.best_true_lr} q={fmt(row.best_true_lr_fdr)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this as evidence that stronger null-margin training improves top-k recovery and hard-negative suppression.",
        "- Do not claim whole-atlas FDR closure; exact-score audit still has zero true q<=0.1 discoveries.",
    ]
    (OUT_DIR / "phase12_null_margin_guarded_training_comparison_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_null_margin_guarded_training_comparison",
        "inputs": [
            rel(MODEL_DIR / "phase12_null_margin_guarded_training_dataset_summary.csv"),
            rel(MODEL_DIR / "phase12_null_margin_guarded_training_comparison.csv"),
            rel(PERM_DIR / "phase12_null_margin_guarded_training_permutation_dataset_summary.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_null_margin_guarded_training_dataset_comparison.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_story.csv"),
            rel(OUT_DIR / "phase12_null_margin_guarded_training_comparison_status.md"),
        ],
        "claim_boundary": "Ranking/control improvement but no broad whole-atlas FDR closure.",
    }
    (OUT_DIR / "phase12_null_margin_guarded_training_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
