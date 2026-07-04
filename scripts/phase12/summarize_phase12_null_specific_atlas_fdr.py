#!/usr/bin/env python
"""Null-specific atlas FDR diagnostics for Phase 12.

The conservative expanded-atlas permutation audit combines three nulls by
taking the worst empirical p-value:

1. receiver permutation;
2. spatial-distance permutation;
3. LR-identity permutation.

This script keeps those conservative results intact, but decomposes the failure
mode. It reports which nulls support discoveries, which null is the bottleneck,
and whether a two-null "pairing + identity" audit would pass before adding the
spatial-distance null.

This is a diagnostic audit, not a replacement for whole-atlas discovery FDR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PERM_DIR = EXPANDED_DIR / "permutation_audit"
OUT_DIR = PHASE12_DIR / "null_specific_atlas_fdr"
NULL_TYPES = ("receiver_permutation", "spatial_distance_permutation", "lr_identity_permutation")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    null_summary = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_null_summary.csv")
    lr = add_two_null_fdr(lr)
    lr_diag = build_lr_diagnostics(lr)
    dataset = build_dataset_diagnostics(lr_diag, null_summary)
    story = build_story(dataset)

    lr_diag.to_csv(OUT_DIR / "phase12_null_specific_atlas_fdr_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_null_specific_atlas_fdr_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_null_specific_atlas_fdr_story.csv", index=False)
    write_status(dataset, lr_diag, story)
    write_manifest(lr_diag, dataset, story)

    print(dataset.to_string(index=False))
    print()
    print(lr_diag.groupby("dataset_key").head(15).to_string(index=False))
    print()
    print(story.to_string(index=False))


def add_two_null_fdr(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        sub = sub.copy()
        sub["two_null_pairing_identity_empirical_p"] = np.maximum(
            pd.to_numeric(sub["receiver_permutation_global_empirical_p"], errors="coerce"),
            pd.to_numeric(sub["lr_identity_permutation_global_empirical_p"], errors="coerce"),
        )
        sub["two_null_pairing_identity_bh_fdr"] = bh_adjust(sub["two_null_pairing_identity_empirical_p"].to_numpy())
        sub["strict_two_null_pairing_identity_empirical_p"] = np.maximum(
            pd.to_numeric(sub["strict_receiver_permutation_global_empirical_p"], errors="coerce"),
            pd.to_numeric(sub["strict_lr_identity_permutation_global_empirical_p"], errors="coerce"),
        )
        sub["strict_two_null_pairing_identity_bh_fdr"] = bh_adjust(
            sub["strict_two_null_pairing_identity_empirical_p"].to_numpy()
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_lr_diagnostics(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in lr.itertuples(index=False):
        q_by_null = {
            null_type: to_float(getattr(row, f"{null_type}_global_bh_fdr", np.nan))
            for null_type in NULL_TYPES
        }
        strict_q_by_null = {
            null_type: to_float(getattr(row, f"strict_{null_type}_global_bh_fdr", np.nan))
            for null_type in NULL_TYPES
        }
        rows.append(
            {
                "dataset_key": row.dataset_key,
                "rank": int(row.rank),
                "strict_rank": int(row.strict_rank),
                "lr_pair": row.lr_pair,
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "true_lr": bool(row.true_lr),
                "selected_reason": row.selected_reason,
                "biological_category": row.biological_category,
                "calibrated_score": row.calibrated_score,
                "strict_discovery_score": row.strict_discovery_score,
                "receiver_q": q_by_null["receiver_permutation"],
                "spatial_distance_q": q_by_null["spatial_distance_permutation"],
                "lr_identity_q": q_by_null["lr_identity_permutation"],
                "conservative_q": to_float(row.conservative_global_bh_fdr),
                "two_null_pairing_identity_q": to_float(row.two_null_pairing_identity_bh_fdr),
                "strict_receiver_q": strict_q_by_null["receiver_permutation"],
                "strict_spatial_distance_q": strict_q_by_null["spatial_distance_permutation"],
                "strict_lr_identity_q": strict_q_by_null["lr_identity_permutation"],
                "strict_conservative_q": to_float(row.strict_conservative_global_bh_fdr),
                "strict_two_null_pairing_identity_q": to_float(row.strict_two_null_pairing_identity_bh_fdr),
                "bottleneck_null": max_q_null(q_by_null),
                "strict_bottleneck_null": max_q_null(strict_q_by_null),
                "null_specific_call": lr_call(row, q_by_null),
                "strict_null_specific_call": strict_lr_call(row, strict_q_by_null),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def build_dataset_diagnostics(lr: pd.DataFrame, null_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in lr.groupby("dataset_key", sort=False):
        true = sub["true_lr"].astype(bool)
        top10 = sub.sort_values("rank").head(10)
        strict_top10 = sub.sort_values("strict_rank").head(10)
        null_sub = null_summary[null_summary["dataset_key"].eq(dataset_key)].copy()
        spatial = pick_null(null_sub, "spatial_distance_permutation")
        receiver = pick_null(null_sub, "receiver_permutation")
        identity = pick_null(null_sub, "lr_identity_permutation")
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "top10_true_lr": int(top10["true_lr"].astype(bool).sum()),
                "strict_top10_true_lr": int(strict_top10["true_lr"].astype(bool).sum()),
                "receiver_q_0_1": int(sub["receiver_q"].le(0.1).sum()),
                "receiver_true_q_0_1": int((true & sub["receiver_q"].le(0.1)).sum()),
                "lr_identity_q_0_1": int(sub["lr_identity_q"].le(0.1).sum()),
                "lr_identity_true_q_0_1": int((true & sub["lr_identity_q"].le(0.1)).sum()),
                "spatial_distance_q_0_1": int(sub["spatial_distance_q"].le(0.1).sum()),
                "spatial_distance_true_q_0_1": int((true & sub["spatial_distance_q"].le(0.1)).sum()),
                "two_null_pairing_identity_q_0_1": int(sub["two_null_pairing_identity_q"].le(0.1).sum()),
                "two_null_pairing_identity_true_q_0_1": int(
                    (true & sub["two_null_pairing_identity_q"].le(0.1)).sum()
                ),
                "strict_two_null_pairing_identity_q_0_1": int(
                    sub["strict_two_null_pairing_identity_q"].le(0.1).sum()
                ),
                "strict_two_null_pairing_identity_true_q_0_1": int(
                    (true & sub["strict_two_null_pairing_identity_q"].le(0.1)).sum()
                ),
                "conservative_q_0_1": int(sub["conservative_q"].le(0.1).sum()),
                "strict_conservative_q_0_1": int(sub["strict_conservative_q"].le(0.1).sum()),
                "dominant_bottleneck_null": mode_value(sub["bottleneck_null"]),
                "strict_dominant_bottleneck_null": mode_value(sub["strict_bottleneck_null"]),
                "observed_top_exceeds_receiver_null_max": bool(get(receiver, "observed_top_exceeds_null_max", False)),
                "observed_top_exceeds_spatial_null_max": bool(get(spatial, "observed_top_exceeds_null_max", False)),
                "observed_top_exceeds_identity_null_max": bool(get(identity, "observed_top_exceeds_null_max", False)),
                "observed_top_exceeds_strict_spatial_null_max": bool(
                    get(spatial, "observed_top_strict_exceeds_null_max", False)
                ),
                "null_specific_audit_call": dataset_call(sub),
            }
        )
    return pd.DataFrame(rows)


def build_story(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    two_null_hits = int(pd.to_numeric(dataset["two_null_pairing_identity_q_0_1"], errors="coerce").fillna(0).sum())
    two_null_true_hits = int(
        pd.to_numeric(dataset["two_null_pairing_identity_true_q_0_1"], errors="coerce").fillna(0).sum()
    )
    spatial_hits = int(pd.to_numeric(dataset["spatial_distance_q_0_1"], errors="coerce").fillna(0).sum())
    conservative_hits = int(pd.to_numeric(dataset["conservative_q_0_1"], errors="coerce").fillna(0).sum())
    if conservative_hits > 0:
        call = "conservative_all_null_fdr_supported"
    elif two_null_true_hits > 0 and spatial_hits == 0:
        call = "pairing_identity_support_spatial_null_bottleneck"
    elif two_null_hits > 0:
        call = "two_null_support_with_caveats"
    else:
        call = "null_specific_audit_not_supportive"
    return pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset)),
                "n_two_null_pairing_identity_q_0_1": two_null_hits,
                "n_two_null_pairing_identity_true_q_0_1": two_null_true_hits,
                "n_spatial_distance_q_0_1": spatial_hits,
                "n_conservative_q_0_1": conservative_hits,
                "overall_null_specific_call": call,
                "claim_boundary": "Receiver-pairing and LR-identity null support is diagnostic. It does not replace the conservative all-null whole-atlas FDR because the spatial-distance null remains a bottleneck.",
            }
        ]
    )


def lr_call(row: Any, q_by_null: dict[str, float]) -> str:
    two_null = to_float(getattr(row, "two_null_pairing_identity_bh_fdr", np.nan))
    conservative = to_float(getattr(row, "conservative_global_bh_fdr", np.nan))
    spatial = q_by_null["spatial_distance_permutation"]
    if np.isfinite(conservative) and conservative <= 0.1:
        return "all_null_fdr_supported"
    if np.isfinite(two_null) and two_null <= 0.1 and np.isfinite(spatial) and spatial > 0.1:
        return "pairing_identity_supported_spatial_bottleneck"
    if np.isfinite(q_by_null["receiver_permutation"]) and q_by_null["receiver_permutation"] <= 0.1:
        return "receiver_pairing_supported_only"
    return "not_null_supported"


def strict_lr_call(row: Any, q_by_null: dict[str, float]) -> str:
    two_null = to_float(getattr(row, "strict_two_null_pairing_identity_bh_fdr", np.nan))
    conservative = to_float(getattr(row, "strict_conservative_global_bh_fdr", np.nan))
    spatial = q_by_null["spatial_distance_permutation"]
    if np.isfinite(conservative) and conservative <= 0.1:
        return "strict_all_null_fdr_supported"
    if np.isfinite(two_null) and two_null <= 0.1 and np.isfinite(spatial) and spatial > 0.1:
        return "strict_pairing_identity_supported_spatial_bottleneck"
    if np.isfinite(q_by_null["receiver_permutation"]) and q_by_null["receiver_permutation"] <= 0.1:
        return "strict_receiver_pairing_supported_only"
    return "strict_not_null_supported"


def dataset_call(sub: pd.DataFrame) -> str:
    conservative = int(sub["conservative_q"].le(0.1).sum())
    two_null_true = int((sub["true_lr"].astype(bool) & sub["two_null_pairing_identity_q"].le(0.1)).sum())
    spatial = int(sub["spatial_distance_q"].le(0.1).sum())
    if conservative > 0:
        return "all_null_fdr_supported"
    if two_null_true > 0 and spatial == 0:
        return "pairing_identity_support_spatial_distance_bottleneck"
    if two_null_true > 0:
        return "pairing_identity_support_with_spatial_caveat"
    return "not_supported_by_null_specific_audit"


def max_q_null(q_by_null: dict[str, float]) -> str:
    finite = {key: val for key, val in q_by_null.items() if np.isfinite(val)}
    if not finite:
        return ""
    return max(finite, key=finite.get)


def mode_value(series: pd.Series) -> str:
    counts = series.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def pick_null(table: pd.DataFrame, null_type: str) -> dict[str, Any]:
    if table.empty:
        return {}
    sub = table[table["null_type"].astype(str).eq(null_type)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def get(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if value is None:
        return default
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
        "# Phase 12 Null-Specific Atlas FDR",
        "",
        "## Scope",
        "",
        "This diagnostic audit decomposes expanded-atlas permutation FDR into receiver-pairing, spatial-distance and LR-identity nulls. It does not replace the conservative all-null FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.null_specific_audit_call}`: LR={int(row.n_lr)}, "
            f"receiver q<=0.1 true={int(row.receiver_true_q_0_1)}, "
            f"LR-identity q<=0.1 true={int(row.lr_identity_true_q_0_1)}, "
            f"two-null q<=0.1 true={int(row.two_null_pairing_identity_true_q_0_1)}, "
            f"spatial-distance q<=0.1 true={int(row.spatial_distance_true_q_0_1)}, "
            f"conservative q<=0.1={int(row.conservative_q_0_1)}, "
            f"bottleneck={row.dominant_bottleneck_null}."
        )
    lines += ["", "## Top LR Diagnostics", ""]
    for row in lr.sort_values(["dataset_key", "rank"]).groupby("dataset_key").head(8).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` rank {int(row.rank)} `{row.lr_pair}`: {row.null_specific_call}; "
            f"receiver q={fmt(row.receiver_q)}, identity q={fmt(row.lr_identity_q)}, "
            f"spatial q={fmt(row.spatial_distance_q)}, conservative q={fmt(row.conservative_q)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_null_specific_call}`: two-null discoveries={int(s.n_two_null_pairing_identity_q_0_1)}, "
            f"two-null true discoveries={int(s.n_two_null_pairing_identity_true_q_0_1)}, "
            f"spatial-distance discoveries={int(s.n_spatial_distance_q_0_1)}, "
            f"conservative discoveries={int(s.n_conservative_q_0_1)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_null_specific_atlas_fdr_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_null_specific_atlas_fdr",
        "n_lr_rows": int(len(lr)),
        "n_dataset_rows": int(len(dataset)),
        "n_story_rows": int(len(story)),
        "inputs": [
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_null_summary.csv",
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_null_specific_atlas_fdr_lr.csv"),
            rel(OUT_DIR / "phase12_null_specific_atlas_fdr_dataset.csv"),
            rel(OUT_DIR / "phase12_null_specific_atlas_fdr_story.csv"),
            rel(OUT_DIR / "phase12_null_specific_atlas_fdr_status.md"),
        ],
        "claim_boundary": "Diagnostic decomposition of permutation nulls; not a replacement for conservative all-null whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_null_specific_atlas_fdr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
