#!/usr/bin/env python
"""Formal closure audit for Phase 12 coordinate model-rerun nulls.

The coordinate rerun layer retrains SpatialSelf after shuffling coordinates,
but the current output is intentionally low-permutation. This script turns that
smoke output into a manuscript-safe closure table: story candidates can be
supported if they pass the coordinate rerun FDR and the story guardrail, while
the same table records broad-background leakage that blocks any whole-atlas
coordinate-null discovery claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
COORD_DIR = PHASE12_DIR / "coordinate_model_rerun_smoke"
GUARD_DIR = PHASE12_DIR / "coordinate_control_guardrail"
OUT_DIR = PHASE12_DIR / "coordinate_null_closure"
ALPHA = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdr = read_coordinate_fdr()
    guard = read_csv(GUARD_DIR / "phase12_coordinate_control_guardrail_lr_summary.csv")
    guard_dataset = read_csv(GUARD_DIR / "phase12_coordinate_control_guardrail_dataset_summary.csv")
    if fdr.empty or guard.empty:
        raise FileNotFoundError("Coordinate FDR or guardrail summary is missing")
    lr = build_lr_closure(fdr, guard)
    dataset = build_dataset_closure(lr, guard_dataset)
    story = build_story_closure(dataset, lr)

    lr.to_csv(OUT_DIR / "phase12_coordinate_null_closure_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_coordinate_null_closure_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_coordinate_null_closure_story.csv", index=False)
    write_status(dataset, lr, story)
    write_manifest(dataset, lr, story)
    print(dataset.to_string(index=False))
    print()
    print(lr.sort_values(["dataset_key", "closure_tier_rank", "model_rerun_bh_fdr", "lr_pair"]).groupby("dataset_key").head(12).to_string(index=False))


def read_coordinate_fdr() -> pd.DataFrame:
    frames = []
    for path in sorted(COORD_DIR.glob("*/*_coordinate_model_rerun_lr_fdr.csv")):
        frame = pd.read_csv(path)
        frame["dataset_key"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_lr_closure(fdr: pd.DataFrame, guard: pd.DataFrame) -> pd.DataFrame:
    fdr = fdr.copy()
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    keep = [
        "dataset_key",
        "lr_id",
        "lr_pair",
        "null_model",
        "model_rerun_group_statistic",
        "model_rerun_p_value",
        "model_rerun_bh_fdr",
        "n_completed_permutations",
        "null_mean",
        "null_q95",
        "true_lr",
        "false_positive_high_expression",
        "biological_category",
        "selected_reason",
    ]
    base = fdr[keep].copy()
    guard_cols = [
        "dataset_key",
        "lr_id",
        "negative_control_q95",
        "negative_control_max",
        "nominal_coordinate_discovery",
        "beats_high_expression_negative_q95",
        "control_guardrail_discovery",
        "story_guardrail_discovery",
        "broad_background_lr",
        "broad_background_control_leakage",
        "guardrail_call",
    ]
    merged = base.merge(guard[guard_cols], on=["dataset_key", "lr_id"], how="left")
    for col in [
        "true_lr",
        "false_positive_high_expression",
        "nominal_coordinate_discovery",
        "beats_high_expression_negative_q95",
        "control_guardrail_discovery",
        "story_guardrail_discovery",
        "broad_background_lr",
        "broad_background_control_leakage",
    ]:
        merged[col] = merged.get(col, pd.Series(False, index=merged.index)).fillna(False).astype(bool)
    merged["coordinate_fdr_pass"] = pd.to_numeric(merged["model_rerun_bh_fdr"], errors="coerce").le(ALPHA)
    merged["coordinate_null_delta_vs_q95"] = (
        pd.to_numeric(merged["model_rerun_group_statistic"], errors="coerce")
        - pd.to_numeric(merged["null_q95"], errors="coerce")
    )
    merged["closure_tier"] = [closure_tier(row) for row in merged.itertuples(index=False)]
    merged["closure_tier_rank"] = merged["closure_tier"].map(tier_rank).fillna(99).astype(int)
    merged["allowed_coordinate_claim"] = [allowed_claim(row) for row in merged.itertuples(index=False)]
    return merged.sort_values(["dataset_key", "closure_tier_rank", "model_rerun_bh_fdr", "lr_pair"])


def closure_tier(row: Any) -> str:
    if bool(row.story_guardrail_discovery) and bool(row.coordinate_fdr_pass) and bool(row.true_lr):
        return "A_story_coordinate_null_supported"
    if bool(row.story_guardrail_discovery) and bool(row.true_lr):
        return "B_story_guardrail_supported_low_perm_caveat"
    if bool(row.broad_background_control_leakage):
        return "D_broad_background_leakage"
    if bool(row.false_positive_high_expression) and bool(row.nominal_coordinate_discovery):
        return "D_high_expression_nominal_leakage_suppressed"
    if bool(row.control_guardrail_discovery):
        return "C_control_guardrail_nonstory"
    return "E_not_coordinate_supported"


def tier_rank(tier: str) -> int:
    return {
        "A_story_coordinate_null_supported": 1,
        "B_story_guardrail_supported_low_perm_caveat": 2,
        "C_control_guardrail_nonstory": 3,
        "D_broad_background_leakage": 4,
        "D_high_expression_nominal_leakage_suppressed": 5,
        "E_not_coordinate_supported": 6,
    }.get(str(tier), 99)


def allowed_claim(row: Any) -> str:
    if row.closure_tier == "A_story_coordinate_null_supported":
        return "Selected story LR is supported by end-to-end coordinate-rerun null and high-expression-control guardrail."
    if row.closure_tier == "B_story_guardrail_supported_low_perm_caveat":
        return "Selected story LR passes the story guardrail, but coordinate FDR is limited by low permutations."
    if row.closure_tier.startswith("D_"):
        return "Use as leakage/limitation evidence; do not claim whole-atlas coordinate-null discovery."
    if row.closure_tier == "C_control_guardrail_nonstory":
        return "Non-story control-guardrail hit; not suitable as biological discovery without extra validation."
    return "No coordinate-null support."


def build_dataset_closure(lr: pd.DataFrame, guard_dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        gd = pick(guard_dataset, dataset_key=dataset_key)
        story = sub["closure_tier"].astype(str).eq("A_story_coordinate_null_supported")
        broad = sub["closure_tier"].astype(str).eq("D_broad_background_leakage")
        high = sub["closure_tier"].astype(str).eq("D_high_expression_nominal_leakage_suppressed")
        completed = int(pd.to_numeric(sub["n_completed_permutations"], errors="coerce").dropna().max())
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_completed_permutations": completed,
                "n_story_coordinate_supported": int(story.sum()),
                "n_true_story_coordinate_supported": int((story & sub["true_lr"].astype(bool)).sum()),
                "n_high_expression_story_supported": int((story & sub["false_positive_high_expression"].astype(bool)).sum()),
                "n_broad_background_leakage": int(broad.sum()),
                "n_high_expression_nominal_leakage_suppressed": int(high.sum()),
                "control_guardrail_broad_background_fraction": get(gd, "control_guardrail_broad_background_fraction", np.nan),
                "top_story_coordinate_supported_lr": ";".join(
                    sub[story].sort_values("model_rerun_group_statistic", ascending=False)["lr_pair"].head(8).astype(str)
                ),
                "dataset_coordinate_null_call": dataset_call(story, broad, high, completed),
            }
        )
    return pd.DataFrame(rows)


def dataset_call(story: pd.Series, broad: pd.Series, high: pd.Series, completed: int) -> str:
    if int(story.sum()) > 0 and completed >= 20:
        if int(broad.sum()) > 0:
            return "story_coordinate_null_supported_whole_atlas_blocked_by_broad_leakage"
        return "story_coordinate_null_supported"
    if int(broad.sum()) > 0 or int(high.sum()) > 0:
        return "coordinate_null_leakage_only"
    return "coordinate_null_not_supportive"


def build_story_closure(dataset: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    n_datasets = int(len(dataset))
    n_supported = int(dataset["n_story_coordinate_supported"].gt(0).sum()) if not dataset.empty else 0
    n_completed = int(pd.to_numeric(dataset.get("n_completed_permutations", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    n_high_story = int(pd.to_numeric(dataset.get("n_high_expression_story_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not dataset.empty else 0
    n_broad = int(pd.to_numeric(dataset.get("n_broad_background_leakage", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not dataset.empty else 0
    if n_datasets >= 2 and n_supported >= 2 and n_high_story == 0:
        call = "coordinate_null_story_support_available_whole_atlas_blocked"
    elif n_supported > 0:
        call = "coordinate_null_story_support_partial"
    else:
        call = "coordinate_null_not_supportive"
    top = lr[lr["closure_tier"].astype(str).eq("A_story_coordinate_null_supported")]
    return pd.DataFrame(
        [
            {
                "n_datasets": n_datasets,
                "n_completed_coordinate_reruns": n_completed,
                "n_datasets_with_story_coordinate_support": n_supported,
                "n_high_expression_story_supported": n_high_story,
                "n_broad_background_leakage": n_broad,
                "overall_coordinate_null_closure_call": call,
                "top_story_lr": ";".join(top.sort_values(["dataset_key", "model_rerun_group_statistic"], ascending=[True, False])["lr_pair"].head(12).astype(str)),
                "claim_boundary": "Coordinate-null support is valid for selected story LR after guardrails; broad-background leakage blocks whole-atlas coordinate-null discovery claims.",
            }
        ]
    )


def write_status(dataset: pd.DataFrame, lr: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Coordinate Null Closure",
        "",
        "## Scope",
        "",
        "This audit closes the coordinate model-rerun layer at story level. It combines end-to-end coordinate-shuffle model reruns, BH FDR, high-expression-control guardrails and broad-background leakage flags. It does not claim unbiased whole-atlas coordinate FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_coordinate_null_call}`: reruns={int(row.n_completed_permutations)}, "
            f"story-supported={int(row.n_story_coordinate_supported)}, high-expression-story={int(row.n_high_expression_story_supported)}, "
            f"broad-background leakage={int(row.n_broad_background_leakage)}, top={row.top_story_coordinate_supported_lr or 'none'}."
        )
    lines += ["", "## Top Story-Supported LR", ""]
    top = lr[lr["closure_tier"].astype(str).eq("A_story_coordinate_null_supported")]
    for row in top.sort_values(["dataset_key", "model_rerun_group_statistic"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.lr_pair}`: BH={fmt(row.model_rerun_bh_fdr)}, p={fmt(row.model_rerun_p_value)}, "
            f"observed={fmt(row.model_rerun_group_statistic)}, null_q95={fmt(row.null_q95)}, call={row.allowed_coordinate_claim}"
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_coordinate_null_closure_call}`: datasets={int(s.n_datasets)}, reruns={int(s.n_completed_coordinate_reruns)}, "
            f"supported datasets={int(s.n_datasets_with_story_coordinate_support)}, broad leakage={int(s.n_broad_background_leakage)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_coordinate_null_closure_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(dataset: pd.DataFrame, lr: pd.DataFrame, story: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_coordinate_null_closure",
        "alpha": ALPHA,
        "inputs": [
            rel(COORD_DIR),
            rel(GUARD_DIR / "phase12_coordinate_control_guardrail_lr_summary.csv"),
            rel(GUARD_DIR / "phase12_coordinate_control_guardrail_dataset_summary.csv"),
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_coordinate_null_closure_lr.csv"),
            rel(OUT_DIR / "phase12_coordinate_null_closure_dataset.csv"),
            rel(OUT_DIR / "phase12_coordinate_null_closure_story.csv"),
            rel(OUT_DIR / "phase12_coordinate_null_closure_status.md"),
        ],
        "n_lr_rows": int(len(lr)),
        "n_dataset_rows": int(len(dataset)),
        "overall_call": str(story["overall_coordinate_null_closure_call"].iloc[0]) if not story.empty else "",
        "claim_boundary": "Selected-story coordinate-null closure; whole-atlas coordinate discovery remains blocked by broad-background leakage.",
    }
    (OUT_DIR / "phase12_coordinate_null_closure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pick(table: pd.DataFrame, **criteria: Any) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=table.index)
    for key, value in criteria.items():
        if key not in table.columns:
            return pd.Series(dtype=object)
        mask &= table[key].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, pd.Series):
        return obj[key] if key in obj.index else default
    return getattr(obj, key, default)


def fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
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
