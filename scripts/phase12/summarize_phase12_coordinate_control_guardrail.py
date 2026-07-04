#!/usr/bin/env python
"""Control-calibrated guardrail for Phase 12 coordinate model-rerun outputs.

The coordinate model-rerun p-value compares each LR pair against its own
coordinate-shuffled null. Phase 12 post-Xenium reruns showed that this can make
high-expression false LR controls nominally significant as well. This script
adds a diagnostic discovery guardrail: an LR must pass the coordinate-rerun FDR
and exceed the high-expression negative-control score distribution.

This is a post-hoc control-panel calibration, not a replacement for formal
whole-atlas FDR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORD_DIR = PROJECT_ROOT / "outputs/phase12/coordinate_model_rerun_smoke"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/coordinate_control_guardrail"
ALPHA = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdr = read_coordinate_fdr()
    if fdr.empty:
        raise FileNotFoundError(f"No coordinate model-rerun FDR tables found under {COORD_DIR}")
    lr_summary = summarize_lr(fdr)
    dataset_summary = summarize_dataset(lr_summary)

    lr_summary.to_csv(OUT_DIR / "phase12_coordinate_control_guardrail_lr_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_coordinate_control_guardrail_dataset_summary.csv", index=False)
    write_status(dataset_summary, lr_summary)
    write_manifest(lr_summary, dataset_summary)
    print(dataset_summary.to_string(index=False))
    print()
    print(lr_summary.groupby("dataset_key").head(15).to_string(index=False))


def read_coordinate_fdr() -> pd.DataFrame:
    frames = []
    for path in sorted(COORD_DIR.glob("*/*_coordinate_model_rerun_lr_fdr.csv")):
        frame = pd.read_csv(path)
        if "dataset_key" not in frame.columns:
            frame["dataset_key"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_lr(fdr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in fdr.groupby("dataset_key", sort=False):
        sub = sub.copy()
        sub["lr_pair"] = sub["ligand"].astype(str) + "-" + sub["receptor"].astype(str)
        stat = pd.to_numeric(sub["model_rerun_group_statistic"], errors="coerce")
        bh = pd.to_numeric(sub["model_rerun_bh_fdr"], errors="coerce")
        p_value = pd.to_numeric(sub["model_rerun_p_value"], errors="coerce")
        high_neg = sub.get("false_positive_high_expression", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        true_lr = sub.get("true_lr", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        category = sub.get("biological_category", pd.Series("", index=sub.index)).astype(str)
        selected = sub.get("selected_reason", pd.Series("", index=sub.index)).astype(str)
        high_neg_stats = stat[high_neg & stat.notna()]
        high_neg_q95 = float(high_neg_stats.quantile(0.95)) if len(high_neg_stats) else np.nan
        high_neg_max = float(high_neg_stats.max()) if len(high_neg_stats) else np.nan
        nominal = bh.le(ALPHA)
        beats_high_neg_q95 = stat.gt(high_neg_q95) if np.isfinite(high_neg_q95) else pd.Series(False, index=sub.index)
        control_guardrail = nominal & beats_high_neg_q95 & ~high_neg
        informative_category = ~category.isin(["other", "high_expression_negative_risk"])
        non_background_reason = ~selected.isin(["category_balanced", "background_fill"])
        story_guardrail = control_guardrail & informative_category & non_background_reason
        broad_background = (
            selected.isin(["category_balanced", "background_fill"])
            | category.isin(["other", "unknown", "nan", ""])
        ) & ~true_lr & ~high_neg
        broad_background_control_leakage = control_guardrail & broad_background & ~story_guardrail

        for idx, row in sub.iterrows():
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "null_model": row.get("null_model", ""),
                    "lr_id": row.get("lr_id", ""),
                    "lr_pair": row["lr_pair"],
                    "true_lr": bool(true_lr.loc[idx]),
                    "false_positive_high_expression": bool(high_neg.loc[idx]),
                    "biological_category": row.get("biological_category", ""),
                    "selected_reason": row.get("selected_reason", ""),
                    "model_rerun_group_statistic": float(stat.loc[idx]) if pd.notna(stat.loc[idx]) else np.nan,
                    "model_rerun_p_value": float(p_value.loc[idx]) if pd.notna(p_value.loc[idx]) else np.nan,
                    "model_rerun_bh_fdr": float(bh.loc[idx]) if pd.notna(bh.loc[idx]) else np.nan,
                    "negative_control_q95": high_neg_q95,
                    "negative_control_max": high_neg_max,
                    "nominal_coordinate_discovery": bool(nominal.loc[idx]),
                    "beats_high_expression_negative_q95": bool(beats_high_neg_q95.loc[idx]),
                    "control_guardrail_discovery": bool(control_guardrail.loc[idx]),
                    "story_guardrail_discovery": bool(story_guardrail.loc[idx]),
                    "broad_background_lr": bool(broad_background.loc[idx]),
                    "broad_background_control_leakage": bool(broad_background_control_leakage.loc[idx]),
                    "guardrail_call": guardrail_call(
                        nominal=bool(nominal.loc[idx]),
                        high_neg=bool(high_neg.loc[idx]),
                        control_guardrail=bool(control_guardrail.loc[idx]),
                        story_guardrail=bool(story_guardrail.loc[idx]),
                        informative_category=bool(informative_category.loc[idx]),
                        broad_background=bool(broad_background.loc[idx]),
                    ),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["dataset_key", "story_guardrail_discovery", "control_guardrail_discovery", "model_rerun_group_statistic"],
        ascending=[True, False, False, False],
    )


def summarize_dataset(lr_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr_summary.groupby("dataset_key", sort=False):
        nominal = sub["nominal_coordinate_discovery"].astype(bool)
        control = sub["control_guardrail_discovery"].astype(bool)
        story = sub["story_guardrail_discovery"].astype(bool)
        true_lr = sub["true_lr"].astype(bool)
        high_neg = sub["false_positive_high_expression"].astype(bool)
        broad = sub["broad_background_control_leakage"].astype(bool)
        background_lr = sub["broad_background_lr"].astype(bool)
        nonstory = control & ~story
        category_balanced = sub["selected_reason"].astype(str).isin(["category_balanced", "background_fill"])
        other_category = sub["biological_category"].astype(str).isin(["other", "unknown", "nan", ""])
        n_perm = int(
            pd.to_numeric(
                sub.get("n_completed_permutations", pd.Series(dtype=float)),
                errors="coerce",
            ).max()
        ) if "n_completed_permutations" in sub.columns else 0
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_nominal_coordinate_discoveries": int(nominal.sum()),
                "n_nominal_true_lr": int((nominal & true_lr).sum()),
                "n_nominal_high_expression_negative": int((nominal & high_neg).sum()),
                "n_control_guardrail_discoveries": int(control.sum()),
                "n_control_guardrail_true_lr": int((control & true_lr).sum()),
                "n_control_guardrail_high_expression_negative": int((control & high_neg).sum()),
                "n_control_guardrail_nonstory": int(nonstory.sum()),
                "n_control_guardrail_broad_background": int(broad.sum()),
                "n_control_guardrail_category_balanced": int((control & category_balanced & ~true_lr & ~high_neg).sum()),
                "n_control_guardrail_other_category": int((control & other_category & ~true_lr & ~high_neg).sum()),
                "n_background_lr": int(background_lr.sum()),
                "control_guardrail_broad_background_fraction": float(broad.sum() / control.sum()) if int(control.sum()) else np.nan,
                "n_story_guardrail_discoveries": int(story.sum()),
                "n_story_guardrail_true_lr": int((story & true_lr).sum()),
                "n_story_guardrail_high_expression_negative": int((story & high_neg).sum()),
                "negative_control_q95": float(pd.to_numeric(sub["negative_control_q95"], errors="coerce").dropna().iloc[0])
                if pd.to_numeric(sub["negative_control_q95"], errors="coerce").notna().any()
                else np.nan,
                "negative_control_max": float(pd.to_numeric(sub["negative_control_max"], errors="coerce").dropna().iloc[0])
                if pd.to_numeric(sub["negative_control_max"], errors="coerce").notna().any()
                else np.nan,
                "top_story_guardrail_lr": ";".join(
                    sub[story].sort_values("model_rerun_group_statistic", ascending=False)["lr_pair"].head(8)
                ),
                "guardrail_interpretation": interpret_dataset(sub),
            }
        )
    return pd.DataFrame(rows)


def guardrail_call(
    *,
    nominal: bool,
    high_neg: bool,
    control_guardrail: bool,
    story_guardrail: bool,
    informative_category: bool,
    broad_background: bool,
) -> str:
    if high_neg and nominal:
        return "nominal_high_expression_negative_suppressed_by_guardrail"
    if story_guardrail:
        return "story_guardrail_supported"
    if control_guardrail and broad_background:
        return "control_guardrail_broad_background_leakage"
    if control_guardrail and not informative_category:
        return "control_guardrail_supported_broad_category_caveat"
    if control_guardrail:
        return "control_guardrail_supported"
    if nominal:
        return "nominal_only_does_not_beat_negative_controls"
    return "not_coordinate_significant"


def interpret_dataset(sub: pd.DataFrame) -> str:
    nominal_high = int(
        (sub["nominal_coordinate_discovery"].astype(bool) & sub["false_positive_high_expression"].astype(bool)).sum()
    )
    story_true = int((sub["story_guardrail_discovery"].astype(bool) & sub["true_lr"].astype(bool)).sum())
    story_high = int(
        (sub["story_guardrail_discovery"].astype(bool) & sub["false_positive_high_expression"].astype(bool)).sum()
    )
    if story_true > 0 and story_high == 0 and nominal_high > 0:
        return "guardrail rescues story candidates while suppressing known high-expression controls"
    if story_true > 0:
        return "guardrail yields story candidates but residual control risk should be inspected"
    if nominal_high > 0:
        return "nominal coordinate discoveries are contaminated by high-expression controls"
    return "no strong coordinate-rerun story guardrail discoveries"


def write_status(dataset_summary: pd.DataFrame, lr_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Coordinate Control Guardrail",
        "",
        "## Scope",
        "",
        "This post-hoc diagnostic layer combines coordinate model-rerun BH FDR with a high-expression negative-control threshold. It does not replace formal whole-atlas FDR; it separates nominal coordinate-null discoveries from candidates that also beat the observed high-expression negative-control panel.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: nominal={int(row.n_nominal_coordinate_discoveries)} "
            f"(true={int(row.n_nominal_true_lr)}, highNeg={int(row.n_nominal_high_expression_negative)}); "
            f"control-guardrail={int(row.n_control_guardrail_discoveries)} "
            f"(true={int(row.n_control_guardrail_true_lr)}, highNeg={int(row.n_control_guardrail_high_expression_negative)}); "
            f"broad-background leakage={int(row.n_control_guardrail_broad_background)} "
            f"({float(row.control_guardrail_broad_background_fraction):.1%} of control-guardrail calls); "
            f"story-guardrail={int(row.n_story_guardrail_discoveries)} "
            f"(true={int(row.n_story_guardrail_true_lr)}, highNeg={int(row.n_story_guardrail_high_expression_negative)}). "
            f"{row.guardrail_interpretation}"
        )
        if str(row.top_story_guardrail_lr):
            lines.append(f"  - top story-guardrail LR: {row.top_story_guardrail_lr}")
    lines.extend(["", "## Claim Boundary", ""])
    lines.append(
        "Use this as a control-calibrated candidate filter and leakage diagnostic only. It can support a conservative story panel, but it is not unbiased whole-atlas discovery FDR and it depends on the constructed high-expression negative-control set. The broad-background leakage columns explicitly show that the high-expression guardrail alone is not strict enough for whole-atlas discovery."
    )
    (OUT_DIR / "phase12_coordinate_control_guardrail_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr_summary: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_coordinate_control_guardrail",
        "alpha": ALPHA,
        "datasets": sorted(dataset_summary["dataset_key"].astype(str).tolist()) if not dataset_summary.empty else [],
        "n_lr_rows": int(len(lr_summary)),
        "outputs": {
            "lr_summary": rel(OUT_DIR / "phase12_coordinate_control_guardrail_lr_summary.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_coordinate_control_guardrail_dataset_summary.csv"),
            "status": rel(OUT_DIR / "phase12_coordinate_control_guardrail_status.md"),
        },
        "claim_boundary": "Post-hoc high-expression-control guardrail; not formal whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_coordinate_control_guardrail_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
