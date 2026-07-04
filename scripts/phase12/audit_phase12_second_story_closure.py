#!/usr/bin/env python
"""Strict closure audit for Phase 12 non-CRC second-story candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
PHASE2R_DIR = PROJECT_ROOT / "outputs/phase2_revised"
OUT_DIR = PHASE12_DIR / "second_story_closure"

TONSIL_A1_RANK = PHASE12_DIR / "manuscript_figure_package/figure3_tonsil_A1_method_lr_rank.csv"
TONSIL_D1_RANK = PHASE12_DIR / "manuscript_figure_package/figure3_tonsil_D1_method_lr_rank.csv"
TONSIL_A1_LOCAL = PHASE12_DIR / "manuscript_figure_package/figure3_tonsil_A1_local_contrast_fdr.csv"
STRICT_STORY = PHASE12_DIR / "strict_story_candidates/phase12_strict_story_candidate_evidence.csv"
EXPANDED_DOWNSTREAM = PHASE12_DIR / "expanded_atlas_downstream_scan/phase12_expanded_atlas_downstream_lr_summary.csv"
CROSS_PLATFORM = PHASE12_DIR / "cross_platform_story/phase12_cross_platform_story_audit.csv"

CURATED_DOWNSTREAM_ROOT = PHASE2R_DIR / "goal1_curated_downstream_panel"
FORMAL_FDR_ROOT = PHASE2R_DIR / "goal5_real_background_fdr_100perm"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(tonsil_rows("spamosaic_tonsil_A1", TONSIL_A1_RANK, TONSIL_A1_LOCAL))
    rows.extend(tonsil_rows("spamosaic_tonsil_D1", TONSIL_D1_RANK, None))
    rows.extend(melanoma_rows())
    rows.extend(lymph_node_rows())

    evidence = pd.DataFrame(rows)
    if not evidence.empty:
        evidence["closure_score"] = evidence.apply(score_candidate, axis=1)
        evidence["closure_class"] = evidence.apply(classify_candidate, axis=1)
        evidence["main_blocker"] = evidence.apply(main_blocker, axis=1)
        evidence = evidence.sort_values(["closure_score", "dataset_key", "lr_pair"], ascending=[False, True, True])

    summary = summarize(evidence)
    evidence.to_csv(OUT_DIR / "phase12_second_story_closure_candidate_evidence.csv", index=False)
    summary.to_csv(OUT_DIR / "phase12_second_story_closure_summary.csv", index=False)
    write_status(evidence, summary)
    write_manifest()
    print(summary.to_string(index=False))
    print()
    print(evidence.to_string(index=False))


def tonsil_rows(dataset_key: str, rank_path: Path, local_path: Path | None) -> list[dict[str, Any]]:
    rank = read_csv(rank_path)
    if rank.empty:
        return []
    local = read_csv(local_path) if local_path else pd.DataFrame()
    curated = load_curated_downstream("real_background_v2_formal_tonsil", "spatialself_wavelet_edge_prior")
    fdr = load_lr_identity_fdr("real_background_v2_formal_tonsil", "human_tonsil_slide_tags_replicate_00")
    rows = []
    positives = rank[(rank["method"].astype(str).eq("spatialself_wavelet_edge_prior")) & (~rank["false_positive_high_expression"].astype(bool))]
    for row in positives.itertuples(index=False):
        lr_pair = f"{row.ligand}-{row.receptor}"
        ss_rank = safe_float(row.rank)
        negative_min_rank = min_negative_rank(rank, "spatialself_wavelet_edge_prior")
        baseline = baseline_ranks(rank, lr_pair)
        local_sub = local[(local["ligand"].astype(str).eq(row.ligand)) & (local["receptor"].astype(str).eq(row.receptor))] if not local.empty else pd.DataFrame()
        local_receiver_q = local_q(local_sub, "receiver_permutation")
        local_spatial_q = local_q(local_sub, "spatial_permutation")
        fdr_row = pick_lr_pair(fdr, lr_pair)
        down = pick_curated_downstream(curated, lr_pair)
        rows.append(
            {
                "dataset_key": dataset_key,
                "tissue": "tonsil",
                "platform": "SpaMosaic tonsil",
                "lr_pair": lr_pair,
                "preferred_goal4_axis": lr_pair in {"CXCL13-CXCR5", "CD40LG-CD40", "IL21-IL21R"},
                "spatialself_rank": ss_rank,
                "spatialself_negative_min_rank": negative_min_rank,
                "spatialself_suppresses_negatives": bool(np.isfinite(negative_min_rank) and ss_rank < negative_min_rank),
                "best_baseline_rank": baseline["best_baseline_rank"],
                "baseline_negative_top_rank": baseline["best_baseline_negative_rank"],
                "baseline_false_positive_mode": baseline["baseline_false_positive_mode"],
                "local_receiver_bh_fdr": local_receiver_q,
                "local_spatial_bh_fdr": local_spatial_q,
                "lr_identity_bh_fdr": get(fdr_row, "group_bh_fdr"),
                "lr_identity_significant": bool(get(fdr_row, "group_significant", False)),
                "downstream_bh_fdr": get(down, "bh_fdr"),
                "downstream_spearman_r": get(down, "spearman_r"),
                "downstream_status": get(down, "status", "missing"),
                "downstream_negative_leakage": curated_negative_leakage(curated),
                "evidence_source": "phase12_tonsil_figure3;phase2_revised_curated_downstream;phase2_revised_formal_fdr",
            }
        )
    return rows


def melanoma_rows() -> list[dict[str, Any]]:
    curated = load_curated_downstream("real_background_v2_formal_melanoma", "spatialself_wavelet_edge_prior")
    fdr = load_lr_identity_fdr("real_background_v2_formal_melanoma", "human_melanoma_multiome_replicate_00")
    candidates = ["CXCL9-CXCR3", "PDCD1-CD274", "SPP1-CD44"]
    rows = []
    for lr_pair in candidates:
        fdr_row = pick_lr_pair(fdr, lr_pair)
        down = pick_curated_downstream(curated, lr_pair)
        rows.append(
            {
                "dataset_key": "native_human_melanoma_multiome",
                "tissue": "melanoma",
                "platform": "native human melanoma multiome",
                "lr_pair": lr_pair,
                "preferred_goal4_axis": True,
                "spatialself_rank": np.nan,
                "spatialself_negative_min_rank": np.nan,
                "spatialself_suppresses_negatives": np.nan,
                "best_baseline_rank": np.nan,
                "baseline_negative_top_rank": np.nan,
                "baseline_false_positive_mode": True,
                "local_receiver_bh_fdr": np.nan,
                "local_spatial_bh_fdr": np.nan,
                "lr_identity_bh_fdr": get(fdr_row, "group_bh_fdr"),
                "lr_identity_significant": bool(get(fdr_row, "group_significant", False)),
                "downstream_bh_fdr": get(down, "bh_fdr"),
                "downstream_spearman_r": get(down, "spearman_r"),
                "downstream_status": get(down, "status", "missing"),
                "downstream_negative_leakage": curated_negative_leakage(curated),
                "evidence_source": "phase2_revised_curated_downstream;phase2_revised_formal_fdr",
            }
        )
    return rows


def lymph_node_rows() -> list[dict[str, Any]]:
    strict = read_csv(STRICT_STORY)
    expanded = read_csv(EXPANDED_DOWNSTREAM)
    if strict.empty:
        return []
    sub = strict[strict["dataset_key"].astype(str).eq("human_lymph_node_visium_raw_full")].copy()
    rows = []
    for row in sub.itertuples(index=False):
        lr_pair = str(row.lr_pair)
        down = expanded[
            (expanded["dataset_key"].astype(str).eq("human_lymph_node_visium_raw_full"))
            & (expanded["lr_pair"].astype(str).eq(lr_pair))
        ].head(1)
        downstream_fdr = float(down["target_top300_enrichment_bh_fdr"].iloc[0]) if len(down) and pd.notna(down["target_top300_enrichment_bh_fdr"].iloc[0]) else get(row, "downstream_target_fdr")
        rows.append(
            {
                "dataset_key": "human_lymph_node_visium_raw_full",
                "tissue": "lymph_node",
                "platform": "human lymph node Visium full-transcriptome",
                "lr_pair": lr_pair,
                "preferred_goal4_axis": lr_pair in {"CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5"},
                "spatialself_rank": get(row, "spatialself_rank"),
                "spatialself_negative_min_rank": np.nan,
                "spatialself_suppresses_negatives": bool(get(row, "high_expression_negative_suppressed", False)),
                "best_baseline_rank": get(row, "baseline_min_rank"),
                "baseline_negative_top_rank": np.nan,
                "baseline_false_positive_mode": bool(safe_float(get(row, "baseline_top10_count")) < safe_float(get(row, "baseline_after10_count"))),
                "local_receiver_bh_fdr": get(row, "targeted_panel_bh_q"),
                "local_spatial_bh_fdr": get(row, "coordinate_bh_fdr"),
                "lr_identity_bh_fdr": get(row, "hard_negative_bh_q"),
                "lr_identity_significant": safe_float(get(row, "hard_negative_bh_q")) <= 0.1,
                "downstream_bh_fdr": downstream_fdr,
                "downstream_spearman_r": np.nan,
                "downstream_status": "expanded_target_rank" if np.isfinite(safe_float(downstream_fdr)) else "missing",
                "downstream_negative_leakage": False,
                "direction_receiver_pairing_q": get(row, "direction_receiver_pairing_q"),
                "direction_vector_q": get(row, "direction_vector_q"),
                "prior_strict_story_class": get(row, "strict_story_class"),
                "evidence_source": "phase12_strict_story;phase12_expanded_downstream",
            }
        )
    return rows


def baseline_ranks(rank: pd.DataFrame, lr_pair: str) -> dict[str, Any]:
    baseline = rank[~rank["method"].astype(str).eq("spatialself_wavelet_edge_prior")].copy()
    baseline["lr_pair"] = baseline["ligand"].astype(str) + "-" + baseline["receptor"].astype(str)
    pair_rows = baseline[baseline["lr_pair"].eq(lr_pair)]
    neg_rows = baseline[baseline["false_positive_high_expression"].astype(bool)]
    best_pair = safe_float(pair_rows["rank"].min()) if len(pair_rows) else np.nan
    best_negative = safe_float(neg_rows["rank"].min()) if len(neg_rows) else np.nan
    return {
        "best_baseline_rank": best_pair,
        "best_baseline_negative_rank": best_negative,
        "baseline_false_positive_mode": bool(np.isfinite(best_negative) and np.isfinite(best_pair) and best_negative <= best_pair),
    }


def min_negative_rank(rank: pd.DataFrame, method: str) -> float:
    sub = rank[(rank["method"].astype(str).eq(method)) & (rank["false_positive_high_expression"].astype(bool))]
    return safe_float(sub["rank"].min()) if len(sub) else np.nan


def local_q(local: pd.DataFrame, null_model: str) -> float:
    if local.empty:
        return np.nan
    sub = local[local["null_model"].astype(str).eq(null_model)]
    return safe_float(sub["local_contrast_bh_fdr"].min()) if len(sub) else np.nan


def load_curated_downstream(dataset: str, method: str) -> pd.DataFrame:
    stem = {
        "real_background_v2_formal_tonsil": "human_tonsil_slide_tags_replicate_00",
        "real_background_v2_formal_melanoma": "human_melanoma_multiome_replicate_00",
    }.get(dataset)
    if not stem:
        return pd.DataFrame()
    path = (
        CURATED_DOWNSTREAM_ROOT
        / dataset
        / method
        / f"{stem}__spatialself_wavelet_predictions__downstream_target_correlation.csv"
    )
    return read_csv(path)


def load_lr_identity_fdr(dataset: str, replicate: str) -> pd.DataFrame:
    path = (
        FORMAL_FDR_ROOT
        / dataset
        / f"{replicate}__spatialself_wavelet_predictions__lr_identity_permutation__lr_pair_direct_fdr.csv"
    )
    return read_csv(path)


def pick_curated_downstream(df: pd.DataFrame, lr_pair: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    ligand, receptor = lr_pair.split("-", 1)
    sub = df[(df["ligand"].astype(str).eq(ligand)) & (df["receptor"].astype(str).eq(receptor))]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.sort_values("bh_fdr", na_position="last").iloc[0]


def curated_negative_leakage(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    neg = df[df["mechanism"].astype(str).str.contains("false_positive", na=False)]
    return bool(pd.to_numeric(neg.get("bh_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).any())


def pick_lr_pair(df: pd.DataFrame, lr_pair: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    ligand, receptor = lr_pair.split("-", 1)
    sub = df[(df["ligand"].astype(str).eq(ligand)) & (df["receptor"].astype(str).eq(receptor))]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def score_candidate(row: pd.Series) -> int:
    score = 0
    rank = safe_float(row.get("spatialself_rank"))
    if np.isfinite(rank) and rank <= 3:
        score += 1
    if bool(row.get("spatialself_suppresses_negatives")):
        score += 1
    baseline_rank = safe_float(row.get("best_baseline_rank"))
    if np.isfinite(rank) and np.isfinite(baseline_rank) and rank < baseline_rank:
        score += 1
    if bool(row.get("baseline_false_positive_mode")):
        score += 1
    if min_finite(row.get("local_receiver_bh_fdr"), row.get("local_spatial_bh_fdr")) <= 0.1:
        score += 1
    if safe_float(row.get("lr_identity_bh_fdr")) <= 0.1:
        score += 1
    if safe_float(row.get("downstream_bh_fdr")) <= 0.1:
        score += 1
    if bool(row.get("downstream_negative_leakage")):
        score -= 1
    if safe_float(row.get("direction_receiver_pairing_q")) <= 0.1:
        score += 1
    if safe_float(row.get("direction_vector_q")) <= 0.1:
        score += 1
    return int(score)


def classify_candidate(row: pd.Series) -> str:
    has_rank = safe_float(row.get("spatialself_rank")) <= 5 or row.get("tissue") == "melanoma"
    has_fdr = min_finite(row.get("local_receiver_bh_fdr"), row.get("local_spatial_bh_fdr"), row.get("lr_identity_bh_fdr")) <= 0.1
    has_downstream = safe_float(row.get("downstream_bh_fdr")) <= 0.1
    negatives_clean = not bool(row.get("downstream_negative_leakage")) and bool(row.get("spatialself_suppresses_negatives", True))
    baseline_ok = bool(row.get("baseline_false_positive_mode")) or safe_float(row.get("spatialself_rank")) < safe_float(row.get("best_baseline_rank"))
    direction_ok = safe_float(row.get("direction_receiver_pairing_q")) <= 0.1

    if has_rank and has_fdr and has_downstream and negatives_clean and (baseline_ok or direction_ok):
        return "A_candidate_if_framed_conservatively"
    if has_rank and has_fdr and (has_downstream or direction_ok):
        return "B_supportive_second_story"
    if has_rank or has_fdr or has_downstream:
        return "C_partial_story"
    return "D_not_story_ready"


def main_blocker(row: pd.Series) -> str:
    blockers = []
    rank = safe_float(row.get("spatialself_rank"))
    if (not np.isfinite(rank) or rank > 5) and row.get("tissue") != "melanoma":
        blockers.append("rank_not_top5")
    if min_finite(row.get("local_receiver_bh_fdr"), row.get("local_spatial_bh_fdr"), row.get("lr_identity_bh_fdr")) > 0.1:
        blockers.append("no_clean_fdr")
    downstream_q = safe_float(row.get("downstream_bh_fdr"))
    if not np.isfinite(downstream_q) or downstream_q > 0.1:
        blockers.append("no_downstream_target_support")
    if bool(row.get("downstream_negative_leakage")):
        blockers.append("downstream_negative_leakage")
    if not bool(row.get("spatialself_suppresses_negatives", True)):
        blockers.append("spatialself_negative_not_suppressed")
    if safe_float(row.get("best_baseline_rank")) <= safe_float(row.get("spatialself_rank")) and not bool(row.get("baseline_false_positive_mode")):
        blockers.append("baseline_equally_strong")
    return ";".join(blockers) if blockers else "no_major_blocker_under_current_audit"


def summarize(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame()
    rows = []
    for dataset_key, sub in evidence.groupby("dataset_key"):
        top = sub.sort_values(["closure_score", "lr_pair"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "best_candidate": top["lr_pair"],
                "best_closure_class": top["closure_class"],
                "best_closure_score": int(top["closure_score"]),
                "n_A_candidates": int(sub["closure_class"].astype(str).eq("A_candidate_if_framed_conservatively").sum()),
                "n_B_candidates": int(sub["closure_class"].astype(str).eq("B_supportive_second_story").sum()),
                "main_blocker": top["main_blocker"],
            }
        )
    return pd.DataFrame(rows).sort_values(["n_A_candidates", "best_closure_score"], ascending=[False, False])


def write_status(evidence: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Second-Story Closure Audit",
        "",
        "## Scope",
        "",
        "This audit asks whether any non-CRC candidate is strong enough to become the second main biological story after post-Xenium SPP1-CD44.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: best={row.best_candidate}; class={row.best_closure_class}; score={int(row.best_closure_score)}; A={int(row.n_A_candidates)}; B={int(row.n_B_candidates)}; blocker={row.main_blocker}"
        )
    if not evidence.empty:
        lines += ["", "## Top Candidates", ""]
        top = evidence.sort_values(["closure_score", "dataset_key"], ascending=[False, True]).head(12)
        for row in top.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` {row.lr_pair}: {row.closure_class}; score={int(row.closure_score)}; blocker={row.main_blocker}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "A candidate labeled A here is still observational. It can be used as a conservative second story only if the manuscript clearly states that real tissue has no native CCC ground truth and that this is not perturbational causality.",
    ]
    (OUT_DIR / "phase12_second_story_closure_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_second_story_closure_audit",
        "outputs": [
            rel(OUT_DIR / "phase12_second_story_closure_candidate_evidence.csv"),
            rel(OUT_DIR / "phase12_second_story_closure_summary.csv"),
            rel(OUT_DIR / "phase12_second_story_closure_status.md"),
        ],
        "claim_boundary": "Candidate-level story closure audit; observational and conservative.",
    }
    (OUT_DIR / "phase12_second_story_closure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get(row: pd.Series | Any, key: str, default: Any = np.nan) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return getattr(row, key, default)


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def min_finite(*values: Any) -> float:
    arr = [safe_float(v) for v in values]
    arr = [v for v in arr if np.isfinite(v)]
    return min(arr) if arr else np.inf


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
