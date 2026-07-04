#!/usr/bin/env python
"""Synthesize revised Phase 2 evidence into manuscript-facing result scaffolds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_SECTIONS = [
    {
        "result_id": "R1",
        "title": "Mechanism-layered benchmarks establish auditable CCC ground truth",
        "claim_ids": ["C1_simulator_v2_mechanism_layered_truth", "C2_real_background_v2_preserves_st_properties"],
        "main_message": "The benchmark layer now combines controlled mechanism-specific truth with real ST-background semi-synthetic datasets, including the generated 18-replicate dataset layer for the next tuned baseline panel.",
        "must_include_caveat": "Do not claim that the simulator reproduces all real CCC biology or that unlabeled real tissue has ground truth.",
        "suggested_figures": "Fig1B;Fig1C",
    },
    {
        "result_id": "R2",
        "title": "SpatialSelf improves the FP/FDR-constrained weak-signal tradeoff",
        "claim_ids": ["C3_spatialself_fp_suppression_tradeoff"],
        "main_message": "SpatialSelf variants retain an FP/FDR-constrained weak-signal tradeoff signal in formal stress summaries and in the completed 18-replicate tuned real-background baseline panel. The frozen endpoint favors SpatialSelf at aggregate level and paired mean endpoint deltas are positive against tested baselines, but tissue-level winners are mixed and paired CIs still cross zero for SpaTalk and CellNEST.",
        "must_include_caveat": "Do not write that SpatialSelf has the highest weak recall, wins every tissue, dominates guardrail sensitivity, has paired significance against every baseline, has a power proof, or has final global superiority across all baselines; disclose endpoint rules and tissue-specific winners.",
        "suggested_figures": "Fig2A",
    },
    {
        "result_id": "R3",
        "title": "Baseline-matched injected controls support false-positive suppression",
        "claim_ids": ["C4_baseline_matched_control_ranking"],
        "main_message": "SpatialSelf keeps high-expression negative controls out of top ranks across formal lymph node, tonsil, and melanoma injected controls.",
        "must_include_caveat": "Downstream receiver-response proxy is mixed and does not yet support a downstream-superiority claim.",
        "suggested_figures": "Fig2B",
    },
    {
        "result_id": "R4",
        "title": "Permutation FDR converts scores into calibrated discovery tables",
        "claim_ids": ["C5_formal_fdr_schema"],
        "main_message": "The FDR module writes candidate-edge, LR-pair, cell-type-pair, and significant-result tables under three fixed null models.",
        "must_include_caveat": "Do not claim final alpha control or 500-permutation manuscript calibration; candidate-edge discoveries remain absent.",
        "suggested_figures": "Fig2C",
    },
    {
        "result_id": "R5",
        "title": "The evidence chain is auditable but not manuscript-final",
        "claim_ids": ["C6_auditability_reproducibility", "C6_runtime_scalability_support"],
        "main_message": "Source tables, claims, limitations, figure exports, and audits are synchronized into a reproducible evidence bundle.",
        "must_include_caveat": "Do not describe the package as a final complete manuscript or full 100k-cell high-resolution scalability proof.",
        "suggested_figures": "Fig1A;runtime supplementary panel",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default="outputs/phase2_revised/goal6_manuscript_assembly")
    args = parser.parse_args()

    assembly_dir = PROJECT_ROOT / args.assembly_dir
    claim_matrix = pd.read_csv(assembly_dir / "claim_evidence_matrix.csv")
    limitations = pd.read_csv(assembly_dir / "manuscript_limitations.csv")
    source_index = pd.read_csv(assembly_dir / "source_table_index.csv")
    figure_plan = pd.read_csv(assembly_dir / "figure_panel_plan.csv")

    result_outline = _result_outline(claim_matrix)
    gap_table = _gap_table(limitations)
    readiness = _readiness_summary(claim_matrix, source_index, figure_plan, gap_table)
    wording_guardrails = _wording_guardrails(claim_matrix, result_outline)

    result_outline.to_csv(assembly_dir / "manuscript_results_outline.csv", index=False)
    gap_table.to_csv(assembly_dir / "manuscript_gap_priority.csv", index=False)
    readiness.to_csv(assembly_dir / "manuscript_readiness_summary.csv", index=False)
    wording_guardrails.to_csv(assembly_dir / "manuscript_wording_guardrails.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal6_results_synthesis",
        "assembly_dir": str(assembly_dir.relative_to(PROJECT_ROOT)),
        "claim_level": "manuscript_scaffold_not_final_text",
        "outputs": {
            "results_outline": str((assembly_dir / "manuscript_results_outline.csv").relative_to(PROJECT_ROOT)),
            "gap_priority": str((assembly_dir / "manuscript_gap_priority.csv").relative_to(PROJECT_ROOT)),
            "readiness_summary": str((assembly_dir / "manuscript_readiness_summary.csv").relative_to(PROJECT_ROOT)),
            "wording_guardrails": str((assembly_dir / "manuscript_wording_guardrails.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (assembly_dir / "manuscript_results_synthesis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result_outline[["result_id", "title", "evidence_status", "ready_for_main_text"]].to_string(index=False))
    print(readiness.to_string(index=False))
    print(f"Wrote manuscript results synthesis to {assembly_dir}")


def _result_outline(claim_matrix: pd.DataFrame) -> pd.DataFrame:
    claims = claim_matrix.set_index("claim_id")
    rows = []
    for section in RESULT_SECTIONS:
        claim_ids = section["claim_ids"]
        statuses = [str(claims.loc[claim_id, "evidence_status"]) for claim_id in claim_ids if claim_id in claims.index]
        allowed = [str(claims.loc[claim_id, "allowed_wording"]) for claim_id in claim_ids if claim_id in claims.index]
        disallowed = [str(claims.loc[claim_id, "disallowed_wording"]) for claim_id in claim_ids if claim_id in claims.index]
        rows.append(
            {
                **section,
                "claim_ids": ";".join(claim_ids),
                "evidence_status": "source_ready" if statuses and set(statuses) == {"source_ready"} else "partial",
                "ready_for_main_text": bool(statuses and set(statuses) == {"source_ready"}),
                "allowed_wording": " | ".join(allowed),
                "disallowed_wording": " | ".join(disallowed),
            }
        )
    return pd.DataFrame(rows)


def _gap_table(limitations: pd.DataFrame) -> pd.DataFrame:
    priority = {
        "L3_downstream_proxy": 1,
        "L2_baseline_panel_incomplete": 2,
        "L4_fdr_power": 3,
        "L1_smoke_scale": 4,
        "L6_runtime_scope": 5,
        "L5_no_final_figures": 6,
    }
    severity = {
        "L3_downstream_proxy": "major",
        "L2_baseline_panel_incomplete": "major",
        "L4_fdr_power": "major",
        "L1_smoke_scale": "moderate",
        "L6_runtime_scope": "moderate",
        "L5_no_final_figures": "polish",
    }
    table = limitations.copy()
    table["priority_rank"] = table["limitation_id"].map(priority).fillna(99).astype(int)
    table["severity"] = table["limitation_id"].map(severity).fillna("moderate")
    table["blocks_final_manuscript_claim"] = table["severity"].isin(["major", "moderate"])
    return table.sort_values(["priority_rank", "limitation_id"]).reset_index(drop=True)


def _readiness_summary(
    claim_matrix: pd.DataFrame,
    source_index: pd.DataFrame,
    figure_plan: pd.DataFrame,
    gap_table: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_level": "phase2_source_ready_not_manuscript_final",
                "source_ready_claims": int((claim_matrix["evidence_status"].astype(str) == "source_ready").sum()),
                "total_claims": int(len(claim_matrix)),
                "ready_source_tables": int(source_index["ready_for_plotting"].astype(bool).sum()),
                "total_source_tables": int(len(source_index)),
                "figure_plan_panels": int(len(figure_plan)),
                "major_or_moderate_gaps": int(gap_table["blocks_final_manuscript_claim"].astype(bool).sum()),
                "main_text_scaffold_ready": bool(
                    (claim_matrix["evidence_status"].astype(str) == "source_ready").all()
                    and source_index["ready_for_plotting"].astype(bool).all()
                ),
                "manuscript_final_ready": False,
            }
        ]
    )


def _wording_guardrails(claim_matrix: pd.DataFrame, result_outline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in claim_matrix.itertuples(index=False):
        rows.append(
            {
                "scope": getattr(row, "claim_id"),
                "allowed_wording": getattr(row, "allowed_wording"),
                "disallowed_wording": getattr(row, "disallowed_wording"),
                "reason": getattr(row, "claim_scope"),
            }
        )
    for row in result_outline.itertuples(index=False):
        rows.append(
            {
                "scope": row.result_id,
                "allowed_wording": row.main_message,
                "disallowed_wording": row.must_include_caveat,
                "reason": row.title,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
