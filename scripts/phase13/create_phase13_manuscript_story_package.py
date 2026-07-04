#!/usr/bin/env python
"""Create Phase 13 manuscript story and figure legend package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase13"
FIG = OUT / "figures"


MAIN_FIGURES = [
    {
        "figure": "Figure 1",
        "title": "SpatialSelf framework and realistic benchmark recovery",
        "headline": "SpatialSelf models LR identity and multi-scale spatial context to recover realistic CCC signals.",
        "plain_story": "第一张图解决“方法到底新在哪里、benchmark 是不是更真实”。它先讲模型有 LR identity、图小波/多尺度空间、hard-negative 自监督，再讲 benchmark 里有近距离、旁分泌、长距离、弱信号和高表达假阳性，最后用性能和消融说明这些设计不是装饰。",
        "legend_intro": "SpatialSelf integrates ligand-receptor identity, multi-scale graph-wavelet spatial context and hard-negative self-supervision for spatial CCC inference.",
        "safe_claim": "SpatialSelf improves recovery in realistic controlled and semi-synthetic benchmark settings.",
        "claim_boundary": "The component audit is controlled evidence; it should not be written as final unbiased whole-atlas FDR proof.",
        "manifest": "phase13_figure1_method_benchmark_manifest.csv",
        "panels": [
            ("a", "Architecture schematic showing LR identity, spatial graph/wavelet context and hard-negative self-supervision."),
            ("b", "Benchmark design covering contact-dependent, paracrine, chemokine-like, weak-signal and high-expression false-positive settings."),
            ("c", "Benchmark recovery summary for weak-signal and mechanism-aware settings."),
            ("d", "Component ablation separating contributions from LR identity, spatial graph/wavelet context and feature components."),
        ],
    },
    {
        "figure": "Figure 2",
        "title": "Baseline comparison and hard-negative suppression",
        "headline": "SpatialSelf gives the strongest practical tradeoff between weak/context signal recovery and high-expression false-positive control.",
        "plain_story": "第二张图专门回应审稿人对 baseline 的质疑。它不是只说我们 AUPR 高，而是把 COMMOT、CellChat、SpaTalk、CellNEST 等放在同一个评价框架里，同时把 DeepTalk/HoloNet/stMLnet/CytoSignal/NICHES 这类方法能不能正式比较说清楚。核心点是：有些方法能把 positive 排前，但 negative 也容易混进来；SpatialSelf 的卖点是 tradeoff 更稳。",
        "legend_intro": "SpatialSelf was compared with retained CCC baselines under formal benchmark, weak-signal, hard-negative and runtime criteria.",
        "safe_claim": "SpatialSelf supports a conservative weak-signal/FDR tradeoff and suppresses high-expression false positives in the retained benchmark panels.",
        "claim_boundary": "Status-only or adapter-light methods should not be described as fully comparable scored baselines unless LR-pair ranking output exists.",
        "manifest": "phase13_figure2_baseline_hardnegative_manifest.csv",
        "panels": [
            ("a", "Baseline comparability and evidence-level contract."),
            ("b", "Formal benchmark performance across controlled and real-background benchmark settings."),
            ("c", "Weak-signal recall versus empirical FDR tradeoff."),
            ("d", "High-expression false-positive suppression."),
            ("e", "Runtime profile for retained runnable methods."),
            ("f", "Deep baseline closure/status table."),
        ],
    },
    {
        "figure": "Figure 3",
        "title": "Post-Xenium colon context-specific communication",
        "headline": "The strongest biological case is SPP1-CD44 in post-Xenium colon, supported by spatial localization, local contrast and receiver response.",
        "plain_story": "第三张图是主生物故事。不要讲成“发现了因果通讯”，而是讲 SpatialSelf 在真实背景里优先找到了一个 context-specific LR，其他方法不容易排前或更容易受高表达负控干扰。SPP1-CD44 有空间局部富集、有 local-contrast FDR、有 receiver target response，所以是目前最强的 case。",
        "legend_intro": "In post-Xenium colon, SpatialSelf prioritized SPP1-CD44 and related context-specific communication supported by spatial and transcriptional evidence.",
        "safe_claim": "SPP1-CD44 is supported as an observational, context-specific CCC candidate with local enrichment and receiver-response consistency.",
        "claim_boundary": "This figure supports biological plausibility and prioritization, not causal signaling or exhaustive whole-atlas FDR.",
        "manifest": "phase13_figure3_post_xenium_colon_manifest.csv",
        "panels": [
            ("a", "Spatial map of prioritized SPP1-CD44 communication edges."),
            ("b", "Control-panel ranking comparison across retained methods."),
            ("c", "Local-contrast FDR separating focal positives from high-expression controls."),
            ("d", "Receiver target-gene and gene-program support."),
            ("e", "Rank and local-contrast separation of high-expression negative controls."),
            ("f", "Audit of local-contrast null models for the colon story."),
        ],
    },
    {
        "figure": "Figure 4",
        "title": "Cross-platform immune communication validation",
        "headline": "Lymph node and tonsil provide supportive cross-platform immune communication evidence, while remaining caveated.",
        "plain_story": "第四张图讲“不是只有一个 colon case”。但这里要克制：lymph node 和 tonsil 是 supportive，不是第二个完美闭环。它们支持 CCL19/CCL21-CCR7、CXCL13-CXCR5 等免疫区域通信在不同组织/平台里能被 SpatialSelf 捕捉到，同时也明确有些 downstream 或 leakage 还没完全闭环。",
        "legend_intro": "SpatialSelf-supported immune communication programs were examined across lymph-node and tonsil contexts.",
        "safe_claim": "SpatialSelf recovers biologically consistent immune communication patterns across tissue and platform contexts.",
        "claim_boundary": "These panels should be written as cross-platform support, not as SpatialSelf-exclusive causal discoveries.",
        "manifest": "phase13_figure4_cross_platform_validation_manifest.csv",
        "panels": [
            ("a", "Story triage across post-Xenium colon, lymph node and tonsil evidence classes."),
            ("b", "Tonsil positive and hard-negative ranking."),
            ("c", "Lymph-node receiver-pairing directionality support."),
            ("d", "Tonsil local-contrast FDR."),
            ("e", "Receiver-program specificity in lymph node."),
            ("f", "Cross-platform audit summary."),
        ],
    },
    {
        "figure": "Figure 5",
        "title": "False-positive guardrails and claim boundaries",
        "headline": "SpatialSelf reduces major false-positive modes, but strict unbiased whole-atlas FDR remains a bounded limitation.",
        "plain_story": "第五张图是给审稿人看的诚实边界。它告诉别人：我们不是把 FDR 问题藏起来，而是分层审计。targeted/matched null、target-decoy、hard-negative、coordinate control 都做了；但 whole-atlas 严格 FDR 还不能吹成完全解决。这个图反而让文章显得成熟。",
        "legend_intro": "False-positive modes were audited using targeted, matched-null, target-decoy, hard-negative, coordinate-control and claim-boundary analyses.",
        "safe_claim": "SpatialSelf reduces major hard-negative and coordinate-control false-positive modes in the audited settings.",
        "claim_boundary": "Do not upgrade targeted or local null evidence into strict unbiased whole-atlas FDR control.",
        "manifest": "phase13_figure5_fdr_guardrails_manifest.csv",
        "panels": [
            ("a", "FDR evidence layers and their claim strength."),
            ("b", "Targeted/matched-null support."),
            ("c", "Control-prefix FDR across top-k thresholds."),
            ("d", "Hard-negative stress testing."),
            ("e", "Edge guardrail leakage analysis."),
            ("f", "Coordinate-control guardrail."),
            ("g", "Pathway receiver-program guardrail."),
            ("h", "Claim matrix separating supported statements from forbidden extensions."),
        ],
    },
]


SUPPLEMENTARY = [
    ("Supplementary Fig. 1", "Benchmark realism audit", "Shows that semi-synthetic benchmarks preserve key real ST background distributions; supports benchmark plausibility, not ground truth."),
    ("Supplementary Fig. 2", "Full formal baseline metrics", "Provides complete retained baseline metrics behind Fig. 2."),
    ("Supplementary Fig. 3", "Weak-signal tradeoff and sensitivity", "Shows weak-signal/FDR guardrail sensitivity and operating-point behavior."),
    ("Supplementary Fig. 4", "Architecture and LR-identity ablation", "Reports broader controlled ablations for LR identity, graph, expression and distance components."),
    ("Supplementary Fig. 5", "Deep baseline closure and method status", "Documents DeepTalk/HoloNet/stMLnet/CytoSignal/NICHES compatibility and scoring limitations."),
    ("Supplementary Fig. 6", "Runtime and scalability", "Reports runtime and high-resolution scaling source evidence."),
    ("Supplementary Fig. 7", "Expanded atlas and top-k contamination", "Audits top-k contamination and expanded-atlas behavior beyond selected LR cases."),
    ("Supplementary Fig. 8", "Target-decoy and matched-null FDR", "Reports targeted/matched-null and target-decoy evidence for story-level FDR support."),
    ("Supplementary Fig. 9", "Post-Xenium colon source evidence", "Provides source evidence behind the SPP1-CD44 colon story."),
    ("Supplementary Fig. 10", "Lymph node and tonsil source evidence", "Provides source evidence behind cross-platform immune communication support."),
    ("Supplementary Fig. 11", "Hard-negative and coordinate-control guardrails", "Reports hard-negative, edge and coordinate-control guardrails."),
    ("Supplementary Fig. 12", "Claim matrix and forbidden extensions", "Defines what the manuscript can and cannot claim from current evidence."),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_rows = []
    for fig in MAIN_FIGURES:
        manifest = read_manifest(fig["manifest"])
        source_map = manifest_source_map(manifest)
        for panel, message in fig["panels"]:
            panel_rows.append(
                {
                    "figure": fig["figure"],
                    "panel": panel,
                    "panel_message": message,
                    "manifest": fig["manifest"],
                    "source_data": source_map.get(panel, ""),
                    "safe_claim": fig["safe_claim"],
                    "claim_boundary": fig["claim_boundary"],
                }
            )
    pd.DataFrame(panel_rows).to_csv(OUT / "phase13_main_figure_panel_story_map.csv", index=False)
    pd.DataFrame(MAIN_FIGURES).drop(columns=["panels"]).to_csv(OUT / "phase13_main_figure_story_summary.csv", index=False)
    pd.DataFrame(SUPPLEMENTARY, columns=["figure", "title", "purpose"]).to_csv(OUT / "phase13_supplementary_story_summary.csv", index=False)

    (OUT / "phase13_manuscript_story_guide_zh.md").write_text(render_zh_story(), encoding="utf-8")
    (OUT / "phase13_figure_legends_draft_en.md").write_text(render_en_legends(), encoding="utf-8")
    print("wrote Phase 13 manuscript story package")


def read_manifest(name: str) -> pd.DataFrame:
    path = FIG / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def manifest_source_map(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "panel" not in df.columns:
        return {}
    source_col = "source_data" if "source_data" in df.columns else "source_key"
    if source_col not in df.columns:
        return {}
    return {str(row["panel"]): str(row[source_col]) for _, row in df.iterrows()}


def render_zh_story() -> str:
    lines = [
        "# Phase 13 Manuscript Story Guide",
        "",
        "## 一句话主线",
        "",
        "SpatialSelf 的故事不是“又一个 CCC 打分器”，而是：用 LR identity、多尺度空间图/图小波和 hard-negative 自监督，把弱的、context-specific 的细胞通讯从高表达假阳性里分出来，并且用真实背景 benchmark、baseline、真实样本和 FDR guardrail 分层证明。",
        "",
        "## 主文图叙事",
        "",
    ]
    for fig in MAIN_FIGURES:
        lines.extend(
            [
                f"### {fig['figure']}：{fig['title']}",
                "",
                f"- 这一图要讲：{fig['headline']}",
                f"- 大白话：{fig['plain_story']}",
                f"- 能说：{fig['safe_claim']}",
                f"- 不能说过头：{fig['claim_boundary']}",
                "",
            ]
        )
        for panel, message in fig["panels"]:
            lines.append(f"  - {panel}: {message}")
        lines.append("")
    lines.extend(
        [
            "## 补充图定位",
            "",
            "补充图不是堆表，而是把主文不适合展开的复杂证据放进去：benchmark realism、完整 baseline、ablation、DeepTalk/status table、runtime、expanded atlas、target-decoy/matched-null FDR、hard-negative/coordinate guardrail、claim matrix。",
            "",
        ]
    )
    for fig, title, purpose in SUPPLEMENTARY:
        lines.append(f"- **{fig}: {title}** — {purpose}")
    lines.extend(
        [
            "",
            "## 写作边界",
            "",
            "- 可以强调：weak/context signal 与 hard-negative suppression 的 tradeoff。",
            "- 可以强调：post-Xenium colon SPP1-CD44 是当前最强 biological case。",
            "- 可以强调：lymph node/tonsil 是 cross-platform supportive validation。",
            "- 不要写：SpatialSelf 已经解决 strict whole-atlas FDR。",
            "- 不要写：DeepTalk 是正式同等评分 baseline，除非后面拿到 comparable LR-pair ranking output。",
            "- 不要写：真实样本结果证明因果通讯；应该写 observational support / receiver-response consistency。",
            "",
        ]
    )
    return "\n".join(lines)


def render_en_legends() -> str:
    lines = [
        "# Phase 13 Draft Figure Legends",
        "",
        "These legends are manuscript drafts. Exact n definitions, test names and correction procedures should be finalized after the Methods text is locked.",
        "",
    ]
    for fig in MAIN_FIGURES:
        lines.extend(
            [
                f"## {fig['figure']} | {fig['title']}",
                "",
                f"{fig['legend_intro']} {fig['safe_claim']} {fig['claim_boundary']}",
                "",
            ]
        )
        panel_text = []
        for panel, message in fig["panels"]:
            panel_text.append(f"({panel}) {message}")
        lines.append(" ".join(panel_text))
        lines.append("")
        lines.append("Source data are provided in the Phase 13 source-data package and panel manifest.")
        lines.append("")
    lines.extend(
        [
            "## Supplementary Figures",
            "",
        ]
    )
    for fig, title, purpose in SUPPLEMENTARY:
        lines.extend([f"### {fig} | {title}", "", purpose, ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
