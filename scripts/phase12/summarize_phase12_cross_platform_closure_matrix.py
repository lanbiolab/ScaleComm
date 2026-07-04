#!/usr/bin/env python
"""Cross-platform closure matrix for Phase 12 Goal 4.

This audit consolidates non-CRC validation evidence into one table. It does not
try to turn caveated tonsil/melanoma results into clean wins; instead it
separates main-story-ready, conservative A-candidate, supportive platform
validation, exploratory mixed evidence, and not-clean cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "cross_platform_closure_matrix"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cross = read_csv(PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv")
    closure = read_csv(PHASE12_DIR / "second_story_closure/phase12_second_story_closure_candidate_evidence.csv")
    direction = read_csv(PHASE12_DIR / "directionality_consensus/phase12_directionality_consensus_lr.csv")

    candidate = build_candidate_matrix(cross, closure, direction)
    dataset = summarize_dataset(candidate, cross)
    story = summarize_story(candidate, dataset)

    candidate.to_csv(OUT_DIR / "phase12_cross_platform_closure_candidate_matrix.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_cross_platform_closure_dataset_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_cross_platform_closure_story_summary.csv", index=False)
    write_status(dataset, candidate, story)
    write_manifest(candidate, dataset, story)
    print(dataset.to_string(index=False))
    print()
    print(candidate.sort_values(["closure_tier_rank", "dataset_key", "candidate_lr_pair"]).to_string(index=False))


def build_candidate_matrix(cross: pd.DataFrame, closure: pd.DataFrame, direction: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not closure.empty:
        for row in closure.itertuples(index=False):
            direction_row = pick(direction, dataset_key=getattr(row, "dataset_key", ""), method="spatialself", lr_pair=getattr(row, "lr_pair", ""))
            rows.append(
                {
                    "dataset_key": row.dataset_key,
                    "tissue": row.tissue,
                    "platform": row.platform,
                    "candidate_lr_pair": row.lr_pair,
                    "evidence_source": row.evidence_source,
                    "spatialself_rank": safe_float(row.spatialself_rank),
                    "spatialself_suppresses_negatives": safe_bool(row.spatialself_suppresses_negatives),
                    "best_baseline_rank": safe_float(row.best_baseline_rank),
                    "baseline_false_positive_mode": safe_bool(row.baseline_false_positive_mode),
                    "local_receiver_bh_fdr": safe_float(row.local_receiver_bh_fdr),
                    "lr_identity_bh_fdr": safe_float(row.lr_identity_bh_fdr),
                    "downstream_bh_fdr": safe_float(row.downstream_bh_fdr),
                    "downstream_negative_leakage": safe_bool(row.downstream_negative_leakage),
                    "direction_receiver_pairing_q": first_finite(
                        safe_float(row.direction_receiver_pairing_q),
                        safe_float(get(direction_row, "receiver_pairing_bh_fdr")),
                    ),
                    "direction_consensus_call": str(get(direction_row, "directionality_consensus_call", "")),
                    "prior_closure_class": row.closure_class,
                    "prior_main_blocker": row.main_blocker,
                    "rank_pass": rank_pass(row),
                    "fdr_pass": fdr_pass(row),
                    "downstream_pass": downstream_pass(row),
                    "negative_suppression_pass": negative_suppression_pass(row),
                    "direction_pass": direction_pass(row, direction_row),
                    "baseline_advantage_pass": baseline_advantage_pass(row),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["n_validation_axes_passed"] = out[
        [
            "rank_pass",
            "fdr_pass",
            "downstream_pass",
            "negative_suppression_pass",
            "direction_pass",
            "baseline_advantage_pass",
        ]
    ].sum(axis=1)
    out["closure_tier"] = [closure_tier(row) for row in out.itertuples(index=False)]
    out["closure_tier_rank"] = out["closure_tier"].map(tier_rank).fillna(99).astype(int)
    out["allowed_cross_platform_claim"] = [allowed_claim(row) for row in out.itertuples(index=False)]
    out["remaining_blocker"] = [remaining_blocker(row) for row in out.itertuples(index=False)]
    return out


def summarize_dataset(candidate: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_keys = sorted(set(candidate.get("dataset_key", pd.Series(dtype=str)).astype(str)) | set(cross.get("dataset_key", pd.Series(dtype=str)).astype(str)))
    for dataset_key in dataset_keys:
        cand = candidate[candidate["dataset_key"].astype(str).eq(dataset_key)] if not candidate.empty else pd.DataFrame()
        cross_row = pick(cross, dataset_key=dataset_key)
        if not cand.empty:
            best = cand.sort_values(["closure_tier_rank", "n_validation_axes_passed", "spatialself_rank"], ascending=[True, False, True]).head(1).iloc[0]
            n_a = int(cand["closure_tier"].astype(str).str.startswith("A").sum())
            n_b = int(cand["closure_tier"].astype(str).str.startswith("B").sum())
            n_supportive = int(cand["closure_tier"].astype(str).isin(["A_conservative_non_crc_story", "B_supportive_platform_validation"]).sum())
            n_not_clean = int(cand["closure_tier"].astype(str).isin(["D_not_clean_under_controls", "C_exploratory_or_partial"]).sum())
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "platform": best.platform,
                    "best_candidate": best.candidate_lr_pair,
                    "best_closure_tier": best.closure_tier,
                    "best_axes_passed": int(best.n_validation_axes_passed),
                    "n_candidates": int(len(cand)),
                    "n_A_candidates": n_a,
                    "n_B_candidates": n_b,
                    "n_supportive_or_better": n_supportive,
                    "n_not_clean_or_partial": n_not_clean,
                    "rank_validation_status": str(get(cross_row, "rank_validation_status", "")),
                    "cross_platform_story_class": str(get(cross_row, "story_class", "")),
                    "dataset_cross_platform_call": dataset_call(best.closure_tier, n_supportive, dataset_key),
                    "recommended_use": dataset_recommended_use(best, cross_row),
                }
            )
        elif len(cross_row):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "platform": get(cross_row, "platform", ""),
                    "best_candidate": get(cross_row, "best_positive_control", ""),
                    "best_closure_tier": "C_exploratory_or_partial",
                    "best_axes_passed": 0,
                    "n_candidates": 0,
                    "n_A_candidates": 0,
                    "n_B_candidates": 0,
                    "n_supportive_or_better": 0,
                    "n_not_clean_or_partial": 1,
                    "rank_validation_status": get(cross_row, "rank_validation_status", ""),
                    "cross_platform_story_class": get(cross_row, "story_class", ""),
                    "dataset_cross_platform_call": "exploratory_cross_platform_audit_only",
                    "recommended_use": get(cross_row, "recommended_use", ""),
                }
            )
    return pd.DataFrame(rows)


def summarize_story(candidate: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    n_dataset = int(len(dataset))
    n_non_crc = int(dataset[~dataset["dataset_key"].astype(str).str.contains("post_xenium|colon|crc", case=False, na=False)].shape[0]) if not dataset.empty else 0
    n_a = int(dataset["n_A_candidates"].sum()) if not dataset.empty else 0
    n_b = int(dataset["n_B_candidates"].sum()) if not dataset.empty else 0
    n_supportive_dataset = int(dataset["n_supportive_or_better"].gt(0).sum()) if not dataset.empty else 0
    n_not_clean = int(dataset["dataset_cross_platform_call"].astype(str).str.contains("not_clean|exploratory|caveat|partial", case=False, na=False).sum()) if not dataset.empty else 0
    if n_a >= 1 and n_supportive_dataset >= 2:
        overall = "cross_platform_validation_supported_with_caveats"
    elif n_a >= 1:
        overall = "second_story_supported_cross_platform_partial"
    else:
        overall = "cross_platform_validation_not_clean"
    return pd.DataFrame(
        [
            {
                "n_datasets_audited": n_dataset,
                "n_non_crc_datasets_audited": n_non_crc,
                "n_A_candidates": n_a,
                "n_B_candidates": n_b,
                "n_datasets_supportive_or_better": n_supportive_dataset,
                "n_datasets_not_clean_or_exploratory": n_not_clean,
                "overall_cross_platform_closure_call": overall,
                "claim_boundary": "Use lymph-node CXCL12-CXCR4 as conservative A candidate; use tonsil as supportive platform evidence; do not use melanoma as clean success under current controls.",
            }
        ]
    )


def closure_tier(row: object) -> str:
    if (
        safe_float(row.spatialself_rank) <= 5
        and row.fdr_pass
        and row.downstream_pass
        and row.negative_suppression_pass
        and not row.downstream_negative_leakage
    ):
        return "A_conservative_non_crc_story"
    if row.rank_pass and row.fdr_pass and row.negative_suppression_pass and row.n_validation_axes_passed >= 4:
        return "B_supportive_platform_validation"
    if row.n_validation_axes_passed >= 3:
        return "C_exploratory_or_partial"
    return "D_not_clean_under_controls"


def tier_rank(tier: str) -> int:
    return {
        "A_conservative_non_crc_story": 1,
        "B_supportive_platform_validation": 2,
        "C_exploratory_or_partial": 3,
        "D_not_clean_under_controls": 4,
    }.get(str(tier), 99)


def allowed_claim(row: object) -> str:
    if row.closure_tier == "A_conservative_non_crc_story":
        return "Conservative non-CRC biological story with rank, FDR, downstream support and negative suppression."
    if row.closure_tier == "B_supportive_platform_validation":
        return "Supportive cross-platform validation; disclose downstream or negative-control caveats."
    if row.closure_tier == "C_exploratory_or_partial":
        return "Exploratory or supplementary evidence only."
    return "Do not use as success story under current controls."


def remaining_blocker(row: object) -> str:
    blockers = []
    if not row.rank_pass:
        blockers.append("rank_not_clean")
    if not row.fdr_pass:
        blockers.append("fdr_not_clean")
    if not row.downstream_pass:
        blockers.append("downstream_not_clean")
    if bool(row.downstream_negative_leakage):
        blockers.append("downstream_negative_leakage")
    if not row.negative_suppression_pass:
        blockers.append("negative_suppression_not_clean")
    if not row.direction_pass:
        blockers.append("direction_not_supported")
    return ";".join(blockers) if blockers else "no_major_blocker_under_current_audit"


def dataset_call(best_tier: str, n_supportive: int, dataset_key: str) -> str:
    if best_tier == "A_conservative_non_crc_story":
        return "conservative_A_story_candidate"
    if best_tier == "B_supportive_platform_validation":
        return "supportive_cross_platform_validation"
    if "melanoma" in str(dataset_key):
        return "not_clean_native_real_sample"
    if n_supportive > 0:
        return "supportive_with_major_caveats"
    return "exploratory_cross_platform_audit_only"


def dataset_recommended_use(best: pd.Series, cross_row: pd.Series) -> str:
    if best.closure_tier == "A_conservative_non_crc_story":
        return "Use as the conservative second biological story, with observational/real-tissue caveats."
    if best.closure_tier == "B_supportive_platform_validation":
        return "Use as supportive platform validation, not as a clean main story."
    existing = str(get(cross_row, "recommended_use", ""))
    return existing if existing else "Use only as exploratory or limitation evidence."


def rank_pass(row: object) -> bool:
    rank = safe_float(getattr(row, "spatialself_rank", np.nan))
    neg = safe_float(getattr(row, "spatialself_negative_min_rank", np.nan))
    return (np.isfinite(rank) and rank <= 10) and (not np.isfinite(neg) or rank < neg)


def fdr_pass(row: object) -> bool:
    vals = [
        safe_float(getattr(row, "local_receiver_bh_fdr", np.nan)),
        safe_float(getattr(row, "lr_identity_bh_fdr", np.nan)),
    ]
    return any(np.isfinite(v) and v <= 0.1 for v in vals)


def downstream_pass(row: object) -> bool:
    q = safe_float(getattr(row, "downstream_bh_fdr", np.nan))
    return np.isfinite(q) and q <= 0.1 and not safe_bool(getattr(row, "downstream_negative_leakage", False))


def negative_suppression_pass(row: object) -> bool:
    suppress = safe_bool(getattr(row, "spatialself_suppresses_negatives", False))
    baseline_fp = safe_bool(getattr(row, "baseline_false_positive_mode", False))
    return suppress or baseline_fp


def direction_pass(row: object, direction_row: pd.Series) -> bool:
    q = first_finite(
        safe_float(getattr(row, "direction_receiver_pairing_q", np.nan)),
        safe_float(get(direction_row, "receiver_pairing_bh_fdr")),
    )
    call = str(get(direction_row, "directionality_consensus_call", ""))
    return (np.isfinite(q) and q <= 0.1) or "source_sink" in call or "receiver_pairing" in call


def baseline_advantage_pass(row: object) -> bool:
    rank = safe_float(getattr(row, "spatialself_rank", np.nan))
    baseline = safe_float(getattr(row, "best_baseline_rank", np.nan))
    baseline_fp = safe_bool(getattr(row, "baseline_false_positive_mode", False))
    return baseline_fp or (np.isfinite(rank) and np.isfinite(baseline) and rank < baseline)


def write_status(dataset: pd.DataFrame, candidate: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Cross-Platform Closure Matrix",
        "",
        "## Scope",
        "",
        "This matrix consolidates non-CRC candidate evidence. It separates conservative A candidates, supportive platform evidence, exploratory mixed results and not-clean cases.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_cross_platform_call}`: best={row.best_candidate} "
            f"({row.best_closure_tier}), A={int(row.n_A_candidates)}, B={int(row.n_B_candidates)}, "
            f"supportive+={int(row.n_supportive_or_better)}. {row.recommended_use}"
        )
    lines += ["", "## Best Candidates", ""]
    for row in candidate.sort_values(["closure_tier_rank", "dataset_key", "candidate_lr_pair"]).head(15).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.candidate_lr_pair}` `{row.closure_tier}`: axes={int(row.n_validation_axes_passed)}, "
            f"rank={fmt(row.spatialself_rank)}, downstream q={fmt(row.downstream_bh_fdr)}, "
            f"direction q={fmt(row.direction_receiver_pairing_q)}, blocker={row.remaining_blocker}"
        )
    if not story.empty:
        row = story.iloc[0]
        lines += [
            "",
            "## Overall",
            "",
            f"- `{row.overall_cross_platform_closure_call}`: datasets={int(row.n_datasets_audited)}, non-CRC={int(row.n_non_crc_datasets_audited)}, A={int(row.n_A_candidates)}, B={int(row.n_B_candidates)}, supportive datasets={int(row.n_datasets_supportive_or_better)}.",
            f"- Claim boundary: {row.claim_boundary}",
        ]
    (OUT_DIR / "phase12_cross_platform_closure_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(candidate: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_cross_platform_closure_matrix",
        "outputs": [
            rel(OUT_DIR / "phase12_cross_platform_closure_candidate_matrix.csv"),
            rel(OUT_DIR / "phase12_cross_platform_closure_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_cross_platform_closure_story_summary.csv"),
            rel(OUT_DIR / "phase12_cross_platform_closure_status.md"),
        ],
        "n_candidate_rows": int(len(candidate)),
        "n_dataset_rows": int(len(dataset)),
        "n_story_rows": int(len(story)),
        "claim_boundary": "Conservative cross-platform closure audit; does not upgrade caveated tonsil/melanoma cases into clean wins.",
    }
    (OUT_DIR / "phase12_cross_platform_closure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pick(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    sub = df
    for key, value in kwargs.items():
        if key not in sub.columns:
            return pd.Series(dtype=object)
        sub = sub[sub[key].astype(str).eq(str(value))]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def get(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    if row is None or len(row) == 0:
        return default
    return row[name] if name in row.index else default


def first_finite(*values: float) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return np.nan


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no", "nan", ""}:
            return False
    if value is None:
        return False
    try:
        return bool(value) and not pd.isna(value)
    except TypeError:
        return bool(value)


def fmt(value: Any) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3g}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
