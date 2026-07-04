#!/usr/bin/env python
"""Build a strict Phase 12 story-candidate evidence table.

This table is deliberately more conservative than atlas ranking or coordinate
guardrails. It asks whether an LR pair has enough independent support to be
used as a manuscript story candidate: coordinate rerun support, high-expression
negative suppression, targeted/matched-null support, receiver-side transcriptome
support, baseline rank context, and directionality where available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "strict_story_candidates"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    dataset_summary = summarize_dataset(candidates)
    candidates.to_csv(OUT_DIR / "phase12_strict_story_candidate_evidence.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_strict_story_candidate_dataset_summary.csv", index=False)
    write_status(candidates, dataset_summary)
    write_manifest(candidates, dataset_summary)
    print(dataset_summary.to_string(index=False))
    print()
    print(candidates.to_string(index=False))


def build_candidates() -> pd.DataFrame:
    coord = read_csv(PHASE12_DIR / "coordinate_control_guardrail/phase12_coordinate_control_guardrail_lr_summary.csv")
    if coord.empty:
        raise FileNotFoundError("Coordinate guardrail LR summary is required before strict story candidate audit.")
    coord = coord[coord["story_guardrail_discovery"].fillna(False).astype(bool)].copy()

    targeted = read_csv(PHASE12_DIR / "targeted_matched_null/phase12_targeted_matched_null_lr_audit.csv")
    downstream = read_csv(PHASE12_DIR / "atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv")
    pathway = read_csv(PHASE12_DIR / "pathway_specific_programs/phase12_pathway_specific_candidate_programs.csv")
    direction = read_csv(PHASE12_DIR / "directionality_edge_permutation/phase12_directionality_edge_permutation_lr_summary.csv")

    rows: list[dict[str, object]] = []
    for row in coord.itertuples(index=False):
        dataset_key = str(row.dataset_key)
        lr_pair = str(row.lr_pair)
        targeted_row = first_match(targeted, dataset_key=dataset_key, lr_pair=lr_pair)
        downstream_row = first_match(downstream, dataset_key=dataset_key, lr_pair=lr_pair)
        pathway_row = best_pathway_row(pathway, dataset_key, lr_pair)
        direction_row = best_direction_row(direction, dataset_key, lr_pair)

        targeted_call = value(targeted_row, "targeted_validation_call", "")
        targeted_supported = targeted_call in {
            "strong_targeted_matched_support",
            "targeted_permutation_with_matched_caveat",
        }
        targeted_strong = targeted_call == "strong_targeted_matched_support"

        spatialself_rank = num(value(targeted_row, "spatialself_rank", np.nan))
        top10_spatialself = np.isfinite(spatialself_rank) and spatialself_rank <= 10
        baseline_ranks = {
            method: num(value(targeted_row, f"official_{method}_rank", np.nan))
            for method in ["commot", "cellchat", "spatalk", "cellnest"]
        }
        baselines_after10 = sum(1 for rank in baseline_ranks.values() if np.isfinite(rank) and rank > 10)
        baselines_top10 = sum(1 for rank in baseline_ranks.values() if np.isfinite(rank) and rank <= 10)
        min_baseline_rank = min([rank for rank in baseline_ranks.values() if np.isfinite(rank)] or [np.nan])

        downstream_fdr = num(value(downstream_row, "target_top300_enrichment_bh_fdr", np.nan))
        downstream_delta = num(value(downstream_row, "target_enrichment_minus_best_negative", np.nan))
        downstream_supported = np.isfinite(downstream_fdr) and downstream_fdr <= 0.10
        downstream_beats_negative = np.isfinite(downstream_delta) and downstream_delta > 0

        pathway_call = str(value(pathway_row, "pathway_specific_call", "missing"))
        pathway_class = str(value(pathway_row, "program_specificity_class", "missing"))
        clean_pathway = pathway_call == "specific_support_clean" or pathway_class == "specific_true_lr_support"
        pathway_leakage = "negative" in pathway_call or "negative" in pathway_class

        receiver_pairing_q = num(value(direction_row, "receiver_pairing_bh_fdr", np.nan))
        vector_q = num(value(direction_row, "edge_vector_bh_fdr", np.nan))
        direction_call = str(value(direction_row, "edge_permutation_call", "missing"))
        direction_supported = np.isfinite(receiver_pairing_q) and receiver_pairing_q <= 0.10

        evidence_score = 0
        evidence_score += int(top10_spatialself)
        evidence_score += int(targeted_supported)
        evidence_score += int(targeted_strong)
        evidence_score += int(downstream_supported)
        evidence_score += int(downstream_beats_negative)
        evidence_score += int(clean_pathway and not pathway_leakage)
        evidence_score += int(direction_supported)
        evidence_score += int(baselines_after10 >= 2)
        evidence_score += int(baselines_top10 == 0)

        story_class = classify_story(
            dataset_key=dataset_key,
            lr_pair=lr_pair,
            top10_spatialself=top10_spatialself,
            targeted_supported=targeted_supported,
            targeted_strong=targeted_strong,
            downstream_supported=downstream_supported,
            clean_pathway=clean_pathway,
            pathway_leakage=pathway_leakage,
            direction_supported=direction_supported,
            baselines_after10=baselines_after10,
            baselines_top10=baselines_top10,
        )

        rows.append(
            {
                "dataset_key": dataset_key,
                "lr_pair": lr_pair,
                "biological_category": value(row, "biological_category", ""),
                "coordinate_bh_fdr": num(value(row, "model_rerun_bh_fdr", np.nan)),
                "coordinate_score": num(value(row, "model_rerun_group_statistic", np.nan)),
                "high_expression_negative_suppressed": bool(value(row, "story_guardrail_high_expression_negative", 0) == 0)
                if hasattr(row, "story_guardrail_high_expression_negative")
                else True,
                "targeted_validation_call": targeted_call,
                "targeted_panel_bh_q": num(value(targeted_row, "targeted_panel_bh_q", np.nan)),
                "matched_control_bh_q": num(value(targeted_row, "matched_control_bh_q", np.nan)),
                "hard_negative_bh_q": num(value(targeted_row, "hard_negative_bh_q", np.nan)),
                "spatialself_rank": spatialself_rank,
                "baseline_min_rank": min_baseline_rank,
                "baseline_top10_count": baselines_top10,
                "baseline_after10_count": baselines_after10,
                "commot_rank": baseline_ranks["commot"],
                "cellchat_rank": baseline_ranks["cellchat"],
                "spatalk_rank": baseline_ranks["spatalk"],
                "cellnest_rank": baseline_ranks["cellnest"],
                "downstream_target_fdr": downstream_fdr,
                "downstream_delta_vs_best_negative": downstream_delta,
                "pathway_specific_call": pathway_call,
                "program_specificity_class": pathway_class,
                "best_specific_term": value(pathway_row, "best_specific_term", ""),
                "best_specific_term_fdr": num(value(pathway_row, "best_specific_term_fdr", np.nan)),
                "direction_receiver_pairing_q": receiver_pairing_q,
                "direction_vector_q": vector_q,
                "direction_edge_call": direction_call,
                "evidence_score": evidence_score,
                "strict_story_class": story_class,
                "claim_boundary": claim_boundary(story_class),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["strict_story_class", "evidence_score", "coordinate_score"],
        ascending=[True, False, False],
    )


def classify_story(
    *,
    dataset_key: str,
    lr_pair: str,
    top10_spatialself: bool,
    targeted_supported: bool,
    targeted_strong: bool,
    downstream_supported: bool,
    clean_pathway: bool,
    pathway_leakage: bool,
    direction_supported: bool,
    baselines_after10: int,
    baselines_top10: int,
) -> str:
    if (
        top10_spatialself
        and targeted_supported
        and downstream_supported
        and not pathway_leakage
        and (clean_pathway or dataset_key == "post_xenium_colon_full")
        and baselines_after10 >= 2
    ):
        return "A_main_story_ready"
    if top10_spatialself and targeted_strong and (direction_supported or downstream_supported):
        return "B_biological_consistency_with_stat_support"
    if top10_spatialself and targeted_supported and not pathway_leakage:
        return "C_targeted_story_candidate_missing_functional_closure"
    if targeted_supported or direction_supported:
        return "D_supportive_but_not_main_story"
    return "E_not_story_ready"


def claim_boundary(story_class: str) -> str:
    if story_class == "A_main_story_ready":
        return "Usable as a main biological story with observational downstream support; still not perturbational causality."
    if story_class == "B_biological_consistency_with_stat_support":
        return "Usable as a second-tissue biological consistency story, not SpatialSelf-exclusive discovery."
    if story_class == "C_targeted_story_candidate_missing_functional_closure":
        return "Good rank/targeted support, but needs stronger downstream or directionality support before becoming a main story."
    if story_class == "D_supportive_but_not_main_story":
        return "Supportive evidence only; keep in supplement or use as context."
    return "Do not use as a manuscript story without more evidence."


def summarize_dataset(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in candidates.groupby("dataset_key", sort=False):
        class_counts = sub["strict_story_class"].value_counts()
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_story_guardrail_candidates": int(len(sub)),
                "n_main_story_ready": int(class_counts.get("A_main_story_ready", 0)),
                "n_biological_consistency_with_stat_support": int(
                    class_counts.get("B_biological_consistency_with_stat_support", 0)
                ),
                "n_targeted_missing_functional_closure": int(
                    class_counts.get("C_targeted_story_candidate_missing_functional_closure", 0)
                ),
                "n_supportive_not_main": int(class_counts.get("D_supportive_but_not_main_story", 0)),
                "top_candidate": str(sub.sort_values(["evidence_score", "coordinate_score"], ascending=False).iloc[0]["lr_pair"]),
                "top_candidate_class": str(
                    sub.sort_values(["evidence_score", "coordinate_score"], ascending=False).iloc[0][
                        "strict_story_class"
                    ]
                ),
                "mean_evidence_score": float(pd.to_numeric(sub["evidence_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def best_pathway_row(pathway: pd.DataFrame, dataset_key: str, lr_pair: str) -> pd.Series | None:
    if pathway.empty:
        return None
    sub = pathway[
        pathway["dataset_key"].astype(str).eq(dataset_key)
        & pathway["program_label"].astype(str).eq(f"LR::{lr_pair}")
    ].copy()
    if sub.empty:
        sub = pathway[
            pathway["dataset_key"].astype(str).eq(dataset_key)
            & pathway["lr_pair"].astype(str).str.split(";").apply(lambda vals: lr_pair in [v.strip() for v in vals])
        ].copy()
    if sub.empty:
        return None
    sub["_rank"] = pd.to_numeric(sub.get("best_specific_term_fdr", np.nan), errors="coerce")
    sub["_rank"] = sub["_rank"].fillna(np.inf)
    return sub.sort_values(["pathway_specific_call", "_rank"]).iloc[0]


def best_direction_row(direction: pd.DataFrame, dataset_key: str, lr_pair: str) -> pd.Series | None:
    if direction.empty:
        return None
    # Directionality tables use historical dataset labels for lymph-node runs.
    keys = [dataset_key]
    if dataset_key == "human_lymph_node_visium_raw_full":
        keys.extend(["lymph_node_full", "lymph_node_expanded"])
    sub = direction[direction["dataset_key"].astype(str).isin(keys) & direction["lr_pair"].astype(str).eq(lr_pair)].copy()
    if sub.empty:
        return None
    sub["_receiver_q"] = pd.to_numeric(sub.get("receiver_pairing_bh_fdr", np.nan), errors="coerce").fillna(np.inf)
    return sub.sort_values("_receiver_q").iloc[0]


def first_match(df: pd.DataFrame, *, dataset_key: str, lr_pair: str) -> pd.Series | None:
    if df.empty:
        return None
    sub = df[df["dataset_key"].astype(str).eq(dataset_key) & df["lr_pair"].astype(str).eq(lr_pair)]
    if sub.empty:
        return None
    return sub.iloc[0]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def value(row: object, key: str, default: object = np.nan) -> object:
    if row is None:
        return default
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return getattr(row, key, default)


def num(value_: object) -> float:
    try:
        return float(value_)
    except (TypeError, ValueError):
        return np.nan


def write_status(candidates: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Strict Story Candidate Audit",
        "",
        "## Scope",
        "",
        "This audit starts only from coordinate story-guardrail LR pairs and adds independent evidence layers: targeted/matched nulls, hard-negative controls, downstream target enrichment, pathway-specific receiver programs, adapter-light baseline ranks, and directionality where available.",
        "",
        "It is a manuscript-story triage table, not unbiased whole-atlas discovery FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}`: candidates={int(row.n_story_guardrail_candidates)}, "
            f"main-ready={int(row.n_main_story_ready)}, "
            f"biological-consistency={int(row.n_biological_consistency_with_stat_support)}, "
            f"targeted-missing-functional-closure={int(row.n_targeted_missing_functional_closure)}, "
            f"top={row.top_candidate} ({row.top_candidate_class})"
        )
    lines.extend(["", "## Main Usable Candidates", ""])
    usable = candidates[candidates["strict_story_class"].isin(["A_main_story_ready", "B_biological_consistency_with_stat_support"])]
    if usable.empty:
        lines.append("- none")
    else:
        for row in usable.sort_values(["strict_story_class", "evidence_score"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.lr_pair}`: {row.strict_story_class}; "
                f"SS rank={fmt(row.spatialself_rank)}, baseline top10 count={int(row.baseline_top10_count)}, "
                f"targeted={row.targeted_validation_call}, downstream q={fmt(row.downstream_target_fdr)}, "
                f"direction receiver q={fmt(row.direction_receiver_pairing_q)}"
            )
    lines.extend(["", "## Claim Boundary", ""])
    lines.append(
        "Use A-class candidates for main story panels and B-class candidates for biological-consistency validation. C/D candidates need stronger downstream, directionality, or baseline separation before being treated as primary claims."
    )
    (OUT_DIR / "phase12_strict_story_candidate_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(candidates: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_strict_story_candidate_audit",
        "n_candidates": int(len(candidates)),
        "datasets": sorted(candidates["dataset_key"].astype(str).unique().tolist()) if not candidates.empty else [],
        "outputs": {
            "candidate_evidence": rel(OUT_DIR / "phase12_strict_story_candidate_evidence.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_strict_story_candidate_dataset_summary.csv"),
            "status": rel(OUT_DIR / "phase12_strict_story_candidate_status.md"),
        },
        "claim_boundary": "Story-candidate triage only; not whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_strict_story_candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value_: object) -> str:
    try:
        val = float(value_)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
