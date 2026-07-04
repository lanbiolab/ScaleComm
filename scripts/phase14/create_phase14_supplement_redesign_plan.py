#!/usr/bin/env python
"""Create Phase 14 supplementary figure redesign plan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase14"


ROWS = [
    {
        "supplementary_figure": "Supplementary Fig. 1",
        "title": "Benchmark realism statistical audit",
        "supports_main": "Figure 1",
        "phase13_source": "phase13_supplementary_figure01_benchmark_realism",
        "phase14_layout": "clean statistical audit grid",
        "must_show": "library size, sparsity, mean-variance, Moran's I, NN distance/PCA evidence",
        "avoid": "overloaded heatmap annotations",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 2",
        "title": "Full formal baseline metrics",
        "supports_main": "Figure 2",
        "phase13_source": "phase13_supplementary_figure02_full_baseline_metrics",
        "phase14_layout": "metric matrix plus small multiple bars",
        "must_show": "AUROC, AUPR, top-k precision, weak recall, empirical FDR, high-expression FP",
        "avoid": "main-figure style claim inflation",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 3",
        "title": "Weak-signal tradeoff sensitivity",
        "supports_main": "Figure 2",
        "phase13_source": "phase13_supplementary_figure03_weak_signal_sensitivity",
        "phase14_layout": "tradeoff sensitivity plate",
        "must_show": "weak recall vs FDR, guardrail thresholds, operating-point sensitivity",
        "avoid": "claiming highest raw weak recall",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 4",
        "title": "Architecture and LR-identity ablation",
        "supports_main": "Figure 1",
        "phase13_source": "phase13_supplementary_figure04_architecture_ablation",
        "phase14_layout": "delta plots grouped by component",
        "must_show": "LR identity, graph/wavelet, expression, distance and pseudo-label components",
        "avoid": "dense horizontal bar dumps",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 5",
        "title": "Deep baseline closure and method status",
        "supports_main": "Figure 2",
        "phase13_source": "phase13_supplementary_figure05_deep_baseline_status",
        "phase14_layout": "status matrix",
        "must_show": "DeepTalk, HoloNet, stMLnet, CytoSignal, NICHES code/dependency/output/ranking status",
        "avoid": "treating status-only methods as scored baselines",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 6",
        "title": "Runtime and scalability",
        "supports_main": "Figure 2",
        "phase13_source": "phase13_supplementary_figure06_runtime_scalability",
        "phase14_layout": "runtime scaling plate",
        "must_show": "wall time, memory, candidate edges, cells/spots",
        "avoid": "unclear log scales",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 7",
        "title": "Expanded atlas and top-k contamination",
        "supports_main": "Figure 5",
        "phase13_source": "phase13_supplementary_figure07_expanded_atlas_topk",
        "phase14_layout": "top-k contamination curves plus compact summary",
        "must_show": "negative/risk fraction by top-k and high-expression contamination",
        "avoid": "burying top-k interpretation",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 8",
        "title": "Target-decoy and matched-null FDR",
        "supports_main": "Figure 5",
        "phase13_source": "phase13_supplementary_figure08_target_decoy_fdr",
        "phase14_layout": "FDR method detail plate",
        "must_show": "targeted q, matched-control q, control-prefix FDR",
        "avoid": "presenting targeted FDR as unbiased whole-atlas FDR",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 9",
        "title": "Post-Xenium colon source evidence",
        "supports_main": "Figure 3",
        "phase13_source": "phase13_supplementary_figure09_post_xenium_colon_sources",
        "phase14_layout": "source evidence plate",
        "must_show": "rank table, local contrast nulls, downstream target enrichment, negative control guardrail",
        "avoid": "repeating the main spatial hero",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 10",
        "title": "Lymph node and tonsil source evidence",
        "supports_main": "Figure 4",
        "phase13_source": "phase13_supplementary_figure10_lymph_tonsil_sources",
        "phase14_layout": "cross-platform source evidence plate",
        "must_show": "tonsil A1/D1 rank, tonsil local contrast, lymph receiver pairing, program specificity",
        "avoid": "overstating supportive datasets",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 11",
        "title": "Hard-negative and coordinate-control guardrails",
        "supports_main": "Figure 5",
        "phase13_source": "phase13_supplementary_figure11_hard_negative_coordinate_guardrails",
        "phase14_layout": "guardrail stress-test plate",
        "must_show": "hard-negative stress, edge leakage, coordinate control, coordinate rerun smoke",
        "avoid": "calling smoke tests final proof",
        "status": "planned_redesign",
    },
    {
        "supplementary_figure": "Supplementary Fig. 12",
        "title": "Claim matrix and forbidden extensions",
        "supports_main": "all figures",
        "phase13_source": "phase13_supplementary_figure12_claim_matrix",
        "phase14_layout": "claim boundary table",
        "must_show": "supported statements, caveats, forbidden extensions, evidence file links",
        "avoid": "dense unwrapped table text",
        "status": "planned_redesign",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = pd.DataFrame(ROWS)
    plan.to_csv(OUT / "phase14_supplement_redesign_plan.csv", index=False)
    (OUT / "phase14_supplement_redesign_plan.md").write_text(render_markdown(plan), encoding="utf-8")
    print(plan.to_string(index=False))


def render_markdown(plan: pd.DataFrame) -> str:
    lines = [
        "# Phase 14 Supplement Redesign Plan",
        "",
        "The supplement should become a supporting evidence system, not an internal audit dump.",
        "",
    ]
    for row in plan.itertuples(index=False):
        lines.extend(
            [
                f"## {row.supplementary_figure} | {row.title}",
                "",
                f"- Supports: {row.supports_main}",
                f"- Phase 13 source: `{row.phase13_source}`",
                f"- Phase 14 layout: {row.phase14_layout}",
                f"- Must show: {row.must_show}",
                f"- Avoid: {row.avoid}",
                f"- Status: `{row.status}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
