#!/usr/bin/env python
"""Render Phase 13 supplementary figure drafts.

These are compact evidence plates, not exhaustive table dumps. Full rows are
copied into Phase 13 source_data so each panel remains traceable.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from phase13_figure_style import PALETTE, apply_style, panel_label, save_figure, soften_axes


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "phase13" / "figures"
SOURCE_OUT = ROOT / "outputs" / "phase13" / "source_data"

P2 = ROOT / "outputs/phase2_revised"
P7 = ROOT / "outputs/phase7/functional_cross_platform_validation"
P8 = ROOT / "outputs/phase8/manuscript_grade_evidence_consolidation"
P12 = ROOT / "outputs/phase12/manuscript_figure_package"

DATASET_LABEL = {
    "lymph_node": "Lymph\nnode",
    "melanoma": "Melanoma",
    "tonsil": "Tonsil",
    "human_lymph_node_visium_raw_full": "Lymph node\nVisium",
    "post_xenium_colon_full": "Post-Xenium\ncolon",
    "visium_hd_crc_full": "Visium HD\nCRC",
}
METHOD_LABEL = {
    "spatialself_wavelet": "SpatialSelf",
    "spatialself_wavelet_edge_prior": "SpatialSelf\n+ edge prior",
    "spatialself_calibrated": "SpatialSelf\ncalibrated",
    "spatialself": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
}
METHOD_COLOR = {
    "SpatialSelf": PALETTE["spatialself"],
    "SpatialSelf\n+ edge prior": PALETTE["wavelet"],
    "SpatialSelf\ncalibrated": PALETTE["spatialself"],
    "COMMOT": "#626262",
    "CellChat": "#9A9A9A",
    "SpaTalk": "#C58E3C",
    "CellNEST": "#8A6AAE",
    "DeepTalk": PALETTE["neutral"],
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []

    for renderer in [
        render_supp1_realism,
        render_supp2_baseline,
        render_supp3_weak_sensitivity,
        render_supp4_ablation,
        render_supp5_deep_status,
        render_supp6_runtime,
        render_supp7_expanded_atlas,
        render_supp8_fdr,
        render_supp9_colon_sources,
        render_supp10_cross_platform_sources,
        render_supp11_guardrails,
        render_supp12_claim_matrix,
    ]:
        manifests.extend(renderer())

    manifest = pd.DataFrame(manifests)
    manifest.to_csv(OUT_DIR / "phase13_supplementary_figures_manifest.csv", index=False)
    print(f"wrote {len(manifest)} supplementary panel manifest rows")


def render_supp1_realism() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 1"
    stem = OUT_DIR / "phase13_supplementary_figure01_benchmark_realism"
    summary = read_csv(P7 / "goal3_benchmark_realism_figure/phase7_benchmark_realism_summary.csv")
    metrics = read_csv(P7 / "goal3_benchmark_realism_figure/phase7_benchmark_realism_metrics.csv")
    export_source("supplementary_figure01_realism_summary", summary)
    export_source("supplementary_figure01_realism_metrics", metrics)

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, left=0.075, right=0.985, bottom=0.09, top=0.84, wspace=0.48, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])

    cols = [
        "selected_library_size_spearman",
        "selected_sparsity_spearman",
        "background_gene_mean_spearman",
        "background_gene_variance_spearman",
        "background_morans_i_spearman",
    ]
    heatmap(
        ax_a,
        summary.set_index("dataset_key")[cols],
        row_labels=[DATASET_LABEL.get(x, x) for x in summary["dataset_key"]],
        col_labels=["library", "sparsity", "gene mean", "gene var", "Moran's I"],
        title="Distributional preservation",
        panel="a",
        vmin=0.95,
        vmax=1.0,
    )
    bar_panel(
        ax_b,
        [DATASET_LABEL.get(x, x).replace("\n", " ") for x in summary["dataset_key"]],
        summary["pca_matched_distance_norm"],
        title="PCA distance remains small",
        ylabel="Matched PCA distance",
        panel="b",
        color=PALETTE["spatialself"],
    )
    bar_panel(
        ax_c,
        [DATASET_LABEL.get(x, x).replace("\n", " ") for x in summary["dataset_key"]],
        summary["median_nn_distance"],
        title="Real coordinate geometry retained",
        ylabel="Median NN distance",
        panel="c",
        color=PALETTE["wavelet"],
    )
    comp = summary[["dataset_key", "n_regions", "n_cell_types"]].melt("dataset_key", var_name="quantity", value_name="count")
    grouped_bar(ax_d, comp, "dataset_key", "quantity", "count", panel="d", title="Composition diversity in real-background benchmarks")
    fig_title(fig, "Supplementary Fig. 1 | Benchmark realism audit", "Semi-synthetic benchmarks preserve real ST background distributions; this supports benchmark plausibility, not biological ground truth.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, ["a", "b", "c", "d"], ["realism_summary", "realism_summary", "realism_summary", "realism_summary"])


def render_supp2_baseline() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 2"
    stem = OUT_DIR / "phase13_supplementary_figure02_full_baseline_metrics"
    summary = read_csv(P2 / "goal2_formal_baseline_panel_all/goal2_formal_baseline_summary.csv")
    metrics = read_csv(P2 / "goal2_formal_baseline_panel_all/goal2_formal_baseline_metrics.csv")
    export_source("supplementary_figure02_formal_baseline_summary", summary)
    export_source("supplementary_figure02_formal_baseline_metrics", metrics)

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.075, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    datasets = ["controlled_v2_formal", "real_background_v2_formal"]
    metric_sets = [
        ("AUPR", "aupr_mean", 0, 0.45),
        ("Top-k precision", "top_k_precision_mean", 0, 0.75),
        ("Weak-signal recall", "weak_signal_recall_at_threshold_mean", 0, 1.0),
        ("High-expression FP rate", "false_positive_high_expression_rate_at_threshold_mean", 0, 1.0),
    ]
    for ax, (title, metric, vmin, vmax), panel in zip(axes, metric_sets, list("abcd")):
        sub = summary[summary["dataset"].isin(datasets)].copy()
        sub["method_label"] = sub["method"].map(METHOD_LABEL).fillna(sub["method"])
        pivot = sub.pivot_table(index="method_label", columns="dataset", values=metric, aggfunc="mean")
        ordered = [METHOD_LABEL.get(x, x) for x in ["spatialself_wavelet_edge_prior", "spatialself_wavelet", "commot", "cellchat", "spatalk", "cellnest"]]
        pivot = pivot.reindex([x for x in ordered if x in pivot.index])
        heatmap(ax, pivot, [x.replace("\n", " ") for x in pivot.index], ["controlled", "real-bg"], title, panel, vmin=vmin, vmax=vmax)
    fig_title(fig, "Supplementary Fig. 2 | Full formal baseline metrics", "Retained baselines are reported across accuracy, weak-signal and hard-negative metrics; adapter-light methods remain labeled as such.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["formal_baseline_summary"] * 4)


def render_supp3_weak_sensitivity() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 3"
    stem = OUT_DIR / "phase13_supplementary_figure03_weak_signal_sensitivity"
    gate = read_csv(P2 / "goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_method_gate.csv")
    sens = read_csv(P2 / "goal2_tuned_guardrail_sensitivity/goal2_tuned_guardrail_sensitivity.csv")
    sens_summary = read_csv(P2 / "goal2_tuned_guardrail_sensitivity/goal2_tuned_guardrail_sensitivity_summary.csv")
    export_source("supplementary_figure03_weak_tradeoff_gate", gate)
    export_source("supplementary_figure03_guardrail_sensitivity", sens)
    export_source("supplementary_figure03_guardrail_sensitivity_summary", sens_summary)

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
    gs = GridSpec(1, 3, figure=fig, left=0.075, right=0.985, bottom=0.12, top=0.82, wspace=0.42)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    scatter_tradeoff(ax_a, gate, panel="a", title="Weak recall vs empirical FDR")
    agg = sens[sens["scope"].eq("aggregate_real_background_v2_formal")].copy()
    mat = agg.pivot_table(index="max_high_expression_fp_rate", columns="max_empirical_fdr", values="spatialself_winner", aggfunc="max").fillna(False).astype(int)
    heatmap(ax_b, mat, [f"{x:.2f}" for x in mat.index], [f"{x:.2f}" for x in mat.columns], "SpatialSelf wins under guardrails", "b", vmin=0, vmax=1)
    bar_panel(
        ax_c,
        [x.replace("human_", "").replace("_replicate_00", "").replace("_", "\n") for x in sens_summary["scope"]],
        sens_summary["spatialself_win_fraction"],
        title="Guardrail-win fraction",
        ylabel="Fraction",
        panel="c",
        color=PALETTE["spatialself"],
    )
    fig_title(fig, "Supplementary Fig. 3 | Weak-signal tradeoff and sensitivity", "SpatialSelf provides a practical weak-signal/FDR tradeoff; this does not claim the maximum raw weak recall.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abc"), ["weak_tradeoff_gate", "guardrail_sensitivity", "guardrail_sensitivity_summary"])


def render_supp4_ablation() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 4"
    stem = OUT_DIR / "phase13_supplementary_figure04_architecture_ablation"
    summary = read_csv(ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_summary.csv")
    metrics = read_csv(ROOT / "outputs/phase1/exp04_model_ablation/full_architecture_identity_component_ablation_epoch10/spatialself_ablation_metrics.csv")
    export_source("supplementary_figure04_ablation_summary", summary)
    export_source("supplementary_figure04_ablation_metrics", metrics)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.18, right=0.985, bottom=0.11, top=0.84, wspace=0.38, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    top = summary.sort_values("aupr_mean", ascending=False).head(10)
    hbar(ax_a, clean_ablation(top["ablation"]), top["aupr_mean"], "AUPR by ablation", "AUPR", "a", PALETTE["spatialself"])
    hbar(ax_b, clean_ablation(top["ablation"]), top["weak_signal_recall_at_threshold_mean"], "Weak-signal recall", "Recall", "b", PALETTE["weak"])
    hbar(ax_c, clean_ablation(top["ablation"]), top["empirical_fdr_at_threshold_mean"], "Empirical FDR", "FDR", "c", PALETTE["hard_negative_soft"])
    hbar(ax_d, clean_ablation(top["ablation"]), top["false_positive_high_expression_rate_at_threshold_mean"], "High-expression FP rate", "FP rate", "d", PALETTE["hard_negative"])
    fig_title(fig, "Supplementary Fig. 4 | Architecture and LR-identity ablation", "Controlled ablations quantify LR identity, graph, expression and distance effects; align final claims with the final model variant.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["ablation_summary"] * 4)


def render_supp5_deep_status() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 5"
    stem = OUT_DIR / "phase13_supplementary_figure05_deep_baseline_status"
    deep = read_csv(P8 / "phase8_deep_baseline_closure_table.csv")
    status = read_csv(P2 / "goal2_formal_baseline_panel_all/goal2_formal_baseline_status.csv")
    export_source("supplementary_figure05_deep_baseline_closure", deep)
    export_source("supplementary_figure05_formal_baseline_status", status)

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig, left=0.055, right=0.985, bottom=0.08, top=0.82, width_ratios=[1.1, 1.25], wspace=0.2)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    table_panel(ax_a, deep[["method", "phase7_final_status", "lr_pair_ranking_support", "manuscript_use"]], "Deep baseline closure", "a", max_rows=5)
    compact = status[["method", "dataset", "baseline_evidence_level", "status"]].drop_duplicates().head(12)
    table_panel(ax_b, compact, "Formal baseline evidence level", "b", max_rows=12)
    fig_title(fig, "Supplementary Fig. 5 | Deep baseline closure and method status", "Deep baseline compatibility and LR-rank support are documented transparently; status-only methods are not treated as scored baselines.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, ["a", "b"], ["deep_baseline_closure", "formal_baseline_status"])


def render_supp6_runtime() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 6"
    stem = OUT_DIR / "phase13_supplementary_figure06_runtime_scalability"
    runtime = read_csv(P2 / "goal7_runtime_scalability/goal7_method_runtime_profile.csv")
    scaling = read_csv(P2 / "goal7_runtime_scalability/goal7_highres_scaling_runs.csv")
    scaling_summary = read_csv(P2 / "goal7_runtime_scalability/goal7_highres_scaling_summary.csv")
    export_source("supplementary_figure06_runtime_profile", runtime)
    export_source("supplementary_figure06_highres_scaling_runs", scaling)
    export_source("supplementary_figure06_highres_scaling_summary", scaling_summary)

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
    gs = GridSpec(1, 3, figure=fig, left=0.075, right=0.985, bottom=0.14, top=0.82, wspace=0.46)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    comp = runtime[runtime["status"].eq("completed")].copy()
    comp["label"] = comp["method"].map(METHOD_LABEL).fillna(comp["method"])
    hbar(ax_a, comp["label"], comp["wall_clock_seconds"], "Runnable adapter runtime", "Seconds", "a", PALETTE["spatialself"], log=True)
    ax_b.plot(scaling["n_cells"], scaling["n_candidate_edges"], marker="o", color=PALETTE["wavelet"], lw=1.4)
    panel_label(ax_b, "b")
    ax_b.set_title("Candidate-edge scaling", loc="left", pad=8)
    ax_b.set_xlabel("Cells/spots")
    ax_b.set_ylabel("Candidate edges")
    soften_axes(ax_b)
    ax_c.plot(scaling["n_cells"], scaling["wall_clock_seconds"], marker="o", color=PALETTE["success"], lw=1.4)
    panel_label(ax_c, "c")
    ax_c.set_title("Wall-clock scaling", loc="left", pad=8)
    ax_c.set_xlabel("Cells/spots")
    ax_c.set_ylabel("Seconds")
    soften_axes(ax_c)
    fig_title(fig, "Supplementary Fig. 6 | Runtime and scalability", "SpatialSelf runtime is practical in retained benchmark runs; runtime adapters do not imply identical biological comparability.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abc"), ["runtime_profile", "highres_scaling_runs", "highres_scaling_runs"])


def render_supp7_expanded_atlas() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 7"
    stem = OUT_DIR / "phase13_supplementary_figure07_expanded_atlas_topk"
    ds = read_csv(P12 / "figure1_expanded_calibrated_dataset_summary.csv")
    topk = read_csv(P12 / "figure1_expanded_calibrated_topk_summary.csv")
    simple = read_csv(P12 / "figure1_expanded_same_universe_baseline_method_summary.csv")
    official = read_csv(P12 / "figure1_expanded_official_adapter_method_summary.csv")
    export_source("supplementary_figure07_expanded_dataset_summary", ds)
    export_source("supplementary_figure07_expanded_topk_summary", topk)
    export_source("supplementary_figure07_same_universe_baselines", simple)
    export_source("supplementary_figure07_official_adapter_baselines", official)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    grouped_bar(ax_a, ds.melt("dataset_key", value_vars=["top10_true_lr", "top10_high_expression_negative"], var_name="metric", value_name="count"), "dataset_key", "metric", "count", "a", "Top10 true LR vs high-expression controls")
    for dataset, sub in topk.groupby("dataset_key"):
        ax_b.plot(sub["top_k"], sub["negative_or_risk_fraction"], marker="o", lw=1.2, label=DATASET_LABEL.get(dataset, dataset).replace("\n", " "))
    panel_label(ax_b, "b")
    ax_b.set_title("Negative/risk fraction by top-k", loc="left", pad=8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("top-k")
    ax_b.set_ylabel("Fraction")
    soften_axes(ax_b)
    ax_b.legend(frameon=False, fontsize=5.3)
    hbar(ax_c, simple["method_label"].head(9), simple["mean_top10_high_expression_negative"].head(9), "Same-universe high-expression contamination", "Mean top10 controls", "c", PALETTE["hard_negative_soft"])
    hbar(ax_d, official["method_label"], official["mean_top10_high_expression_negative"], "Official adapter-light high-expression controls", "Mean top10 controls", "d", PALETTE["hard_negative"])
    fig_title(fig, "Supplementary Fig. 7 | Expanded atlas and top-k contamination", "Expanded atlas scoring and top-k contamination are quantified beyond selected LR panels.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["expanded_dataset_summary", "expanded_topk_summary", "same_universe_baselines", "official_adapter_baselines"])


def render_supp8_fdr() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 8"
    stem = OUT_DIR / "phase13_supplementary_figure08_target_decoy_fdr"
    td_ds = read_csv(P12 / "figure1_expanded_target_decoy_dataset_summary.csv")
    td_topk = read_csv(P12 / "figure1_expanded_target_decoy_topk_summary.csv")
    targeted = read_csv(P12 / "figure1_targeted_matched_null_dataset_summary.csv")
    supported = read_csv(P12 / "figure1_targeted_matched_null_supported_lr.csv")
    export_source("supplementary_figure08_target_decoy_dataset", td_ds)
    export_source("supplementary_figure08_target_decoy_topk", td_topk)
    export_source("supplementary_figure08_targeted_dataset", targeted)
    export_source("supplementary_figure08_supported_lr", supported)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    grouped_bar(ax_a, targeted.melt("dataset_key", value_vars=["n_targeted_panel_q_0_1", "n_matched_control_q_0_1", "n_strong_targeted_matched_support"], var_name="metric", value_name="count"), "dataset_key", "metric", "count", "a", "Targeted and matched-null support")
    sub = td_topk[td_topk["top_k"].isin([10, 100, 200])].copy()
    for dataset, sdf in sub.groupby("dataset_key"):
        ax_b.plot(sdf["top_k"], sdf["primary_control_prefix_fdr"], marker="o", lw=1.2, label=DATASET_LABEL.get(dataset, dataset).replace("\n", " "))
    panel_label(ax_b, "b")
    ax_b.set_title("Target-decoy control prefix FDR", loc="left", pad=8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("top-k")
    ax_b.set_ylabel("Control FDR")
    soften_axes(ax_b)
    ax_b.legend(frameon=False, fontsize=5.3)
    top = supported.sort_values(["dataset_key", "spatialself_rank"]).head(10)
    hbar(ax_c, top["dataset_key"].map(DATASET_LABEL).str.replace("\n", " ") + "\n" + top["lr_pair"], top["targeted_panel_bh_q"], "Supported LR targeted q", "BH q", "c", PALETTE["spatialself"])
    hbar(ax_d, top["dataset_key"].map(DATASET_LABEL).str.replace("\n", " ") + "\n" + top["lr_pair"], top["matched_control_bh_q"], "Supported LR matched-control q", "BH q", "d", PALETTE["success"])
    fig_title(fig, "Supplementary Fig. 8 | Targeted, target-decoy and matched-null FDR", "Control and targeted nulls quantify story-level support; they are not unbiased whole-atlas discovery FDR.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["targeted_dataset", "target_decoy_topk", "supported_lr", "supported_lr"])


def render_supp9_colon_sources() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 9"
    stem = OUT_DIR / "phase13_supplementary_figure09_post_xenium_colon_sources"
    rank = read_csv(P12 / "figure2_crc_method_lr_rank_table.csv")
    fdr = read_csv(P12 / "figure2_crc_local_contrast_fdr.csv")
    down = read_csv(P12 / "figure2_crc_whole_transcriptome_downstream.csv")
    impact = read_csv(P12 / "figure2_crc_commot_style_impact.csv")
    export_source("supplementary_figure09_crc_rank", rank)
    export_source("supplementary_figure09_crc_local_fdr", fdr)
    export_source("supplementary_figure09_crc_downstream", down)
    export_source("supplementary_figure09_crc_commot_style_impact", impact)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    rank["lr_pair"] = rank["ligand"].astype(str) + "-" + rank["receptor"].astype(str)
    focal = rank[rank["lr_pair"].isin(["SPP1-CD44", "TGFB1-TGFBR2", "CXCL12-CXCR4", "ACTB-KRT8", "COL1A1-EPCAM"])].copy()
    heat = focal.pivot_table(index="lr_pair", columns="method", values="rank", aggfunc="min")
    heatmap(ax_a, heat, list(heat.index), [METHOD_LABEL.get(c, c) for c in heat.columns], "Control-panel ranks", "a", vmin=1, vmax=max(8, np.nanmax(heat.values)))
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    rx = fdr[fdr["null_model"].eq("receiver_permutation")]
    hbar(ax_b, rx["lr_pair"], -np.log10(rx["local_contrast_bh_fdr"].clip(lower=1e-4)), "Local contrast FDR", "-log10 BH q", "b", PALETTE["spatialself"])
    hbar(ax_c, down["lr_pair"], down["target_top300_enrichment"], "Receiver target enrichment", "Top300 enrichment", "c", PALETTE["success"])
    hbar(ax_d, impact["method"], impact["max_negative_rf_importance"], "Negative impact guardrail", "Max negative importance", "d", PALETTE["hard_negative_soft"])
    fig_title(fig, "Supplementary Fig. 9 | Post-Xenium colon source evidence", "The SPP1-CD44 story is supported by ranking, local contrast and receiver-response source tables; validation remains observational.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["crc_rank", "crc_local_fdr", "crc_downstream", "crc_commot_style_impact"])


def render_supp10_cross_platform_sources() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 10"
    stem = OUT_DIR / "phase13_supplementary_figure10_lymph_tonsil_sources"
    lymph = read_csv(P12 / "figure3_lymph_node_directionality_edge_permutation.csv")
    program = read_csv(P12 / "figure3_lymph_node_program_specificity.csv")
    tonsil_rank = read_csv(P12 / "figure3_tonsil_A1_method_lr_rank.csv")
    tonsil_fdr = read_csv(P12 / "figure3_tonsil_A1_local_contrast_fdr.csv")
    tonsil_d1 = read_csv(P12 / "figure3_tonsil_D1_method_lr_rank.csv")
    export_source("supplementary_figure10_lymph_directionality_edge", lymph)
    export_source("supplementary_figure10_lymph_program_specificity", program)
    export_source("supplementary_figure10_tonsil_A1_rank", tonsil_rank)
    export_source("supplementary_figure10_tonsil_A1_fdr", tonsil_fdr)
    export_source("supplementary_figure10_tonsil_D1_rank", tonsil_d1)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    sub = lymph[(lymph["dataset_key"].eq("lymph_node_full")) & (lymph["method"].isin(["spatialself", "commot"]))].copy()
    heat = sub.pivot_table(index="lr_pair", columns="method", values="receiver_pairing_bh_fdr", aggfunc="min")
    heatmap(ax_a, -np.log10(heat.clip(lower=1e-4)), list(heat.index), [METHOD_LABEL.get(c, c) for c in heat.columns], "Lymph-node receiver-pairing q", "a", vmin=0, vmax=2)
    hbar(ax_b, program["lr_pair"], program["specific_fraction_of_significant"].fillna(0), "Lymph-node program specificity", "Specific fraction", "b", PALETTE["weak"])
    tonsil_rank["lr_pair"] = tonsil_rank["ligand"].astype(str) + "-" + tonsil_rank["receptor"].astype(str)
    heat2 = tonsil_rank.pivot_table(index="lr_pair", columns="method_label", values="rank", aggfunc="min")
    heat2 = heat2.reindex(columns=[c for c in ["SpatialSelf", "COMMOT", "CellChat", "SpaTalk", "CellNEST"] if c in heat2.columns])
    heatmap(ax_c, heat2, list(heat2.index), list(heat2.columns), "Tonsil A1 LR ranks", "c", vmin=1, vmax=7)
    tonsil_fdr["lr_pair"] = tonsil_fdr["ligand"].astype(str) + "-" + tonsil_fdr["receptor"].astype(str)
    rx = tonsil_fdr[tonsil_fdr["null_model"].eq("receiver_permutation")]
    hbar(ax_d, rx["lr_pair"], -np.log10(rx["local_contrast_bh_fdr"].clip(lower=1e-4)), "Tonsil local contrast", "-log10 BH q", "d", PALETTE["spatialself"])
    fig_title(fig, "Supplementary Fig. 10 | Lymph node and tonsil supportive validation", "Lymph node and tonsil provide supportive cross-platform evidence with explicit caveats about mixed downstream closure and leakage.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["lymph_directionality_edge", "lymph_program_specificity", "tonsil_A1_rank", "tonsil_A1_fdr"])


def render_supp11_guardrails() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 11"
    stem = OUT_DIR / "phase13_supplementary_figure11_hard_negative_coordinate_guardrails"
    hard = read_csv(P12 / "figure4_hard_negative_dataset_summary.csv")
    edge = read_csv(P12 / "figure4_edge_hard_negative_dataset_summary.csv")
    coord = read_csv(P12 / "figure4_coordinate_control_guardrail_dataset_summary.csv")
    smoke = read_csv(P12 / "figure4_coordinate_model_rerun_smoke_summary.csv")
    export_source("supplementary_figure11_hard_negative_dataset", hard)
    export_source("supplementary_figure11_edge_guardrail", edge)
    export_source("supplementary_figure11_coordinate_guardrail", coord)
    export_source("supplementary_figure11_coordinate_smoke", smoke)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.985, bottom=0.12, top=0.84, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    hbar(ax_a, hard["dataset_key"].map(DATASET_LABEL), hard["negative_comparison_beat_fraction"], "Hard-negative stress", "Beat fraction", "a", PALETTE["hard_negative_soft"])
    grouped_bar(ax_b, edge.melt(["dataset_key", "score_mode"], value_vars=["mean_leakage_fraction"], var_name="metric", value_name="value"), "dataset_key", "score_mode", "value", "b", "Edge guardrail leakage")
    grouped_bar(ax_c, coord.melt("dataset_key", value_vars=["n_story_guardrail_true_lr", "n_story_guardrail_high_expression_negative"], var_name="metric", value_name="count"), "dataset_key", "metric", "count", "c", "Coordinate-control story guardrail")
    grouped_bar(ax_d, smoke.melt("dataset_key", value_vars=["n_positive_p_le_0_1", "n_negative_p_le_0_1"], var_name="metric", value_name="count"), "dataset_key", "metric", "count", "d", "Coordinate rerun smoke")
    fig_title(fig, "Supplementary Fig. 11 | Hard-negative and coordinate-control guardrails", "False-positive and coordinate-control behavior are quantified; coordinate reruns remain smoke/guardrail evidence.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, list("abcd"), ["hard_negative_dataset", "edge_guardrail", "coordinate_guardrail", "coordinate_smoke"])


def render_supp12_claim_matrix() -> list[dict[str, object]]:
    fig_id = "Supplementary Fig. 12"
    stem = OUT_DIR / "phase13_supplementary_figure12_claim_matrix"
    claims = read_csv(P12 / "figure4_claim_matrix_guardrails.csv")
    index = read_csv(P12 / "phase12_manuscript_figure_panel_index.csv")
    export_source("supplementary_figure12_claim_matrix", claims)
    export_source("supplementary_figure12_panel_index", index)

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig, left=0.045, right=0.985, bottom=0.06, top=0.84, width_ratios=[1.35, 1.0], wspace=0.18)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    keep = claims[["topic", "support_status", "caveat_or_forbidden_extension"]].head(18)
    table_panel(ax_a, keep, "Claim guardrail matrix", "a", max_rows=18, fontsize=4.5)
    panel_index = index[["figure_id", "panel", "primary_message", "review_risk"]].head(16)
    table_panel(ax_b, panel_index, "Panel evidence map", "b", max_rows=16, fontsize=4.3)
    fig_title(fig, "Supplementary Fig. 12 | Claim matrix and forbidden extensions", "Each major claim is linked to an evidence file and a forbidden extension; this is a writing guardrail, not new evidence.")
    save_figure(fig, stem)
    plt.close(fig)
    return manifest_rows(fig_id, stem, ["a", "b"], ["claim_matrix", "panel_index"])


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def export_source(name: str, df: pd.DataFrame) -> Path:
    path = SOURCE_OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def fig_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, 0.94, subtitle, ha="left", va="top", fontsize=6.2, color=PALETTE["muted"])


def heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    panel: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    panel_label(ax, panel, x=-0.12, y=1.03)
    arr = df.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2g}" if val < 10 else f"{val:.0f}", ha="center", va="center", fontsize=5.2, color="white" if val > np.nanmean(arr) else PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5.2, length=0)


def bar_panel(ax: plt.Axes, labels, values, title: str, ylabel: str, panel: str, color: str) -> None:
    panel_label(ax, panel, x=-0.1, y=1.03)
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylabel(ylabel)
    soften_axes(ax)


def hbar(ax: plt.Axes, labels, values, title: str, xlabel: str, panel: str, color: str, *, log: bool = False) -> None:
    labels = [str(x) for x in labels]
    values = pd.Series(values).astype(float).fillna(0).to_numpy()
    y = np.arange(len(labels))[::-1]
    panel_label(ax, panel, x=-0.1, y=1.03)
    ax.barh(y, values, color=color, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])


def grouped_bar(ax: plt.Axes, df: pd.DataFrame, x_col: str, group_col: str, y_col: str, panel: str, title: str) -> None:
    panel_label(ax, panel, x=-0.1, y=1.03)
    pivot = df.pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc="mean").fillna(0)
    x = np.arange(len(pivot.index))
    width = 0.78 / max(len(pivot.columns), 1)
    colors = [PALETTE["spatialself"], PALETTE["success"], PALETTE["weak"], PALETTE["hard_negative_soft"], PALETTE["neutral"]]
    for i, col in enumerate(pivot.columns):
        ax.bar(x - 0.39 + width / 2 + i * width, pivot[col].to_numpy(float), width=width, color=colors[i % len(colors)], edgecolor="white", linewidth=0.5, label=str(col).replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(str(v), str(v)).replace("\n", " ") for v in pivot.index], rotation=25, ha="right")
    ax.set_title(title, loc="left", pad=8)
    soften_axes(ax)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right")


def scatter_tradeoff(ax: plt.Axes, gate: pd.DataFrame, panel: str, title: str) -> None:
    panel_label(ax, panel, x=-0.1, y=1.03)
    df = gate.copy()
    for _, row in df.iterrows():
        label = METHOD_LABEL.get(row["method"], row["method"])
        ax.scatter(
            row["empirical_fdr_at_threshold_mean"],
            row["weak_signal_recall_at_threshold_mean"],
            s=65 + 500 * float(row["aupr_mean"]),
            color=METHOD_COLOR.get(label, PALETTE["neutral"]),
            edgecolor=PALETTE["ink"] if "SpatialSelf" in label else "white",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.text(row["empirical_fdr_at_threshold_mean"] + 0.015, row["weak_signal_recall_at_threshold_mean"] + 0.018, label.replace("\n", " "), fontsize=5.4)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("Empirical FDR")
    ax.set_ylabel("Weak-signal recall")
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.04, 1.08)
    soften_axes(ax)


def table_panel(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str, *, max_rows: int = 10, fontsize: float = 5.0) -> None:
    panel_label(ax, panel, x=-0.03, y=1.02)
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=8)
    sub = df.head(max_rows).copy()
    wrapped = sub.astype(str).map(lambda x: "\n".join(textwrap.wrap(x, width=28)) if len(x) > 28 else x)
    table = ax.table(cellText=wrapped.values, colLabels=wrapped.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["neutral_soft"])
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor(PALETTE["soft_bg"])
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor("white")


def clean_ablation(series: pd.Series) -> list[str]:
    return [
        str(x)
        .replace("wavelet_", "")
        .replace("hard_expression_distance", "hard expr+dist")
        .replace("fixed_random_lr_embedding", "random LR emb.")
        .replace("one_hot_lr_identity", "one-hot LR")
        .replace("no_expression_features", "no expression")
        .replace("no_spatial_graph", "no spatial graph")
        .replace("no_lr_embedding", "no LR emb.")
        .replace("no_distance", "no distance")
        .replace("_", " ")
        for x in series
    ]


def manifest_rows(fig_id: str, stem: Path, panels: list[str], source_keys: list[str]) -> list[dict[str, object]]:
    return [
        {
            "supplementary_figure": fig_id,
            "panel": panel,
            "figure_stem": stem.name,
            "svg": f"outputs/phase13/figures/{stem.name}.svg",
            "pdf": f"outputs/phase13/figures/{stem.name}.pdf",
            "png": f"outputs/phase13/figures/{stem.name}.png",
            "source_key": source,
            "status": "draft_rendered",
        }
        for panel, source in zip(panels, source_keys)
    ]


if __name__ == "__main__":
    main()
