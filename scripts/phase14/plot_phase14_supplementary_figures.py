#!/usr/bin/env python
"""Render Phase 14 redesigned supplementary figure system.

These figures are cleaner support plates derived from Phase 13 source data.
They intentionally avoid dense table screenshots while keeping traceability.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from phase14_style import PALETTE, METHOD_COLORS, apply_style, panel_label, clean_axes, save_figure


ROOT = Path(__file__).resolve().parents[2]
P13 = ROOT / "outputs" / "phase13" / "source_data"
OUT = ROOT / "outputs" / "phase14"
FIG = OUT / "figures"
SOURCE = OUT / "source_data"


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest = []
    renderers = [
        supp1_realism,
        supp2_baseline,
        supp3_weak_sensitivity,
        supp4_ablation,
        supp5_deep_status,
        supp6_runtime,
        supp7_topk,
        supp8_fdr,
        supp9_colon,
        supp10_cross_platform,
        supp11_guardrails,
        supp12_claim_matrix,
    ]
    for renderer in renderers:
        manifest.extend(renderer())
    pd.DataFrame(manifest).to_csv(FIG / "phase14_supplementary_figures_manifest.csv", index=False)
    print(f"wrote {len(manifest)} Phase 14 supplementary panel records")


def supp1_realism() -> list[dict[str, str]]:
    summary = read("supplementary_figure01_realism_summary.csv")
    metrics = read("supplementary_figure01_realism_metrics.csv")
    export("supplementary_figure01_realism_summary", summary)
    export("supplementary_figure01_realism_metrics", metrics)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"wspace": 0.45})
    cols = ["selected_library_size_spearman", "selected_sparsity_spearman", "background_morans_i_spearman"]
    heat(axs[0], summary.set_index("dataset_key")[cols], "Realism correlations", "a")
    bar(axs[1], summary["dataset_key"], summary["pca_matched_distance_norm"], "PCA distance", "b", PALETTE["spatialself"])
    bar(axs[2], summary["dataset_key"], summary["median_nn_distance"], "NN distance", "c", PALETTE["wavelet"])
    title(fig, "Supplementary Fig. 1 | Benchmark realism audit", "Real-background simulations preserve key ST distributional features.")
    return save_supp(fig, "phase14_supplementary_figure01_benchmark_realism", "Supplementary Fig. 1", "benchmark realism")


def supp2_baseline() -> list[dict[str, str]]:
    summary = read("supplementary_figure02_formal_baseline_summary.csv")
    export("supplementary_figure02_formal_baseline_summary", summary)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.5})
    summary["method_clean"] = summary["method"].map(clean_method)
    real = summary[summary["dataset"].eq("real_background_v2_formal")]
    for ax, metric, name, panel in [
        (axs[0], "aupr_mean", "AUPR", "a"),
        (axs[1], "weak_signal_recall_at_threshold_mean", "weak recall", "b"),
        (axs[2], "false_positive_high_expression_rate_at_threshold_mean", "high-expr FP", "c"),
    ]:
        ordered_bar(ax, real, "method_clean", metric, name, panel)
    title(fig, "Supplementary Fig. 2 | Full formal baseline metrics", "Complete baseline metrics are kept in supplement, separated from the main tradeoff figure.")
    return save_supp(fig, "phase14_supplementary_figure02_full_baseline_metrics", "Supplementary Fig. 2", "formal baseline metrics")


def supp3_weak_sensitivity() -> list[dict[str, str]]:
    gate = read("supplementary_figure03_weak_tradeoff_gate.csv")
    sens = read("supplementary_figure03_guardrail_sensitivity_summary.csv")
    export("supplementary_figure03_weak_tradeoff_gate", gate)
    export("supplementary_figure03_guardrail_sensitivity_summary", sens)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.42})
    tradeoff(axs[0], gate, "a")
    bar(axs[1], sens["scope"], sens["spatialself_win_fraction"], "guardrail win fraction", "b", PALETTE["spatialself"])
    title(fig, "Supplementary Fig. 3 | Weak-signal sensitivity", "Weak-signal gains are interpreted jointly with empirical FDR and guardrail thresholds.")
    return save_supp(fig, "phase14_supplementary_figure03_weak_signal_sensitivity", "Supplementary Fig. 3", "weak sensitivity")


def supp4_ablation() -> list[dict[str, str]]:
    summary = read("supplementary_figure04_ablation_summary.csv")
    export("supplementary_figure04_ablation_summary", summary)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.48})
    top = summary.sort_values("aupr_mean", ascending=False).head(8).copy()
    top["label"] = top["ablation"].map(lambda x: str(x).replace("wavelet_", "").replace("_", " "))
    hbar(axs[0], top["label"], top["aupr_mean"], "AUPR", "a", PALETTE["spatialself"])
    hbar(axs[1], top["label"], top["weak_signal_recall_at_threshold_mean"], "weak recall", "b", PALETTE["weak"])
    hbar(axs[2], top["label"], top["false_positive_high_expression_rate_at_threshold_mean"], "high-expr FP", "c", PALETTE["risk_soft"])
    title(fig, "Supplementary Fig. 4 | Component ablation", "Ablations remain component-level evidence and are kept out of the main figure density.")
    return save_supp(fig, "phase14_supplementary_figure04_architecture_ablation", "Supplementary Fig. 4", "ablation")


def supp5_deep_status() -> list[dict[str, str]]:
    deep = read("supplementary_figure05_deep_baseline_closure.csv")
    export("supplementary_figure05_deep_baseline_closure", deep)
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    status_matrix(ax, deep, "Deep baseline status matrix", "a")
    title(fig, "Supplementary Fig. 5 | Deep baseline closure", "Deep baseline compatibility is shown transparently without counting status-only methods as scored baselines.")
    return save_supp(fig, "phase14_supplementary_figure05_deep_baseline_status", "Supplementary Fig. 5", "deep baseline status")


def supp6_runtime() -> list[dict[str, str]]:
    runtime = read("supplementary_figure06_runtime_profile.csv")
    scaling = read("supplementary_figure06_highres_scaling_runs.csv")
    export("supplementary_figure06_runtime_profile", runtime)
    export("supplementary_figure06_highres_scaling_runs", scaling)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.42})
    r = runtime[runtime["status"].eq("completed")].copy()
    method_col = "method_label" if "method_label" in r.columns else "method"
    r["method_clean"] = r[method_col].map(clean_method)
    ordered_bar(axs[0], r, "method_clean", "wall_clock_seconds", "runtime (s)", "a", log=True)
    axs[1].plot(scaling["n_cells"], scaling["wall_clock_seconds"], marker="o", color=PALETTE["spatialself"], lw=1.4)
    panel_label(axs[1], "b")
    axs[1].set_title("scaling smoke test", loc="left", pad=8)
    axs[1].set_xlabel("cells/spots")
    axs[1].set_ylabel("seconds")
    clean_axes(axs[1])
    title(fig, "Supplementary Fig. 6 | Runtime and scalability", "Runtime evidence is shown as supporting feasibility rather than biological validation.")
    return save_supp(fig, "phase14_supplementary_figure06_runtime_scalability", "Supplementary Fig. 6", "runtime")


def supp7_topk() -> list[dict[str, str]]:
    ds = read("supplementary_figure07_expanded_dataset_summary.csv")
    topk = read("supplementary_figure07_expanded_topk_summary.csv")
    export("supplementary_figure07_expanded_dataset_summary", ds)
    export("supplementary_figure07_expanded_topk_summary", topk)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.42})
    for dataset, sub in topk.groupby("dataset_key"):
        axs[0].plot(sub["top_k"], sub["negative_or_risk_fraction"], marker="o", lw=1.2, label=short_dataset(dataset))
    panel_label(axs[0], "a")
    axs[0].set_title("top-k contamination", loc="left", pad=8)
    axs[0].set_xscale("log")
    axs[0].set_ylabel("negative/risk fraction")
    clean_axes(axs[0])
    axs[0].legend(frameon=False, fontsize=5.4)
    grouped(axs[1], ds, ["top10_true_lr", "top10_high_expression_negative"], "top10 content", "b")
    title(fig, "Supplementary Fig. 7 | Expanded atlas top-k contamination", "Top-k contamination is audited beyond selected LR examples.")
    return save_supp(fig, "phase14_supplementary_figure07_expanded_atlas_topk", "Supplementary Fig. 7", "top-k contamination")


def supp8_fdr() -> list[dict[str, str]]:
    targeted = read("supplementary_figure08_targeted_dataset.csv")
    topk = read("supplementary_figure08_target_decoy_topk.csv")
    supported = read("supplementary_figure08_supported_lr.csv")
    export("supplementary_figure08_targeted_dataset", targeted)
    export("supplementary_figure08_target_decoy_topk", topk)
    export("supplementary_figure08_supported_lr", supported)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})
    grouped(axs[0], targeted, ["n_targeted_panel_q_0_1", "n_matched_control_q_0_1"], "targeted vs matched", "a")
    for dataset, sub in topk[topk["top_k"].isin([10, 100])].groupby("dataset_key"):
        axs[1].plot(sub["top_k"], sub["primary_control_prefix_fdr"], marker="o", lw=1.2, label=short_dataset(dataset))
    panel_label(axs[1], "b")
    axs[1].set_title("target-decoy FDR", loc="left", pad=8)
    axs[1].set_xscale("log")
    axs[1].set_ylabel("control FDR")
    clean_axes(axs[1])
    hbar(axs[2], supported["lr_pair"].head(8), supported["targeted_panel_bh_q"].head(8), "supported LR q", "c", PALETTE["spatialself"])
    title(fig, "Supplementary Fig. 8 | Target-decoy and matched-null FDR", "Targeted and matched-null FDR are detailed without overclaiming whole-atlas control.")
    return save_supp(fig, "phase14_supplementary_figure08_target_decoy_fdr", "Supplementary Fig. 8", "FDR detail")


def supp9_colon() -> list[dict[str, str]]:
    rank = read("supplementary_figure09_crc_rank.csv")
    fdr = read("supplementary_figure09_crc_local_fdr.csv")
    down = read("supplementary_figure09_crc_downstream.csv")
    export("supplementary_figure09_crc_rank", rank)
    export("supplementary_figure09_crc_local_fdr", fdr)
    export("supplementary_figure09_crc_downstream", down)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})
    rank["lr_pair"] = rank["ligand"].astype(str) + "-" + rank["receptor"].astype(str)
    ss = rank[rank["method_label"].astype(str).str.contains("SpatialSelf", na=False)].head(8)
    hbar(axs[0], ss["lr_pair"], ss["rank"], "SpatialSelf rank", "a", PALETTE["signal"])
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    sub = fdr[fdr["null_model"].eq("lr_identity_permutation")].copy()
    hbar(axs[1], sub["lr_pair"], -np.log10(sub["local_contrast_bh_fdr"].clip(lower=1e-4)), "local contrast", "b", PALETTE["spatialself"])
    hbar(axs[2], down["lr_pair"], down["target_top300_enrichment"], "target response", "c", PALETTE["weak"])
    title(fig, "Supplementary Fig. 9 | Post-Xenium colon source evidence", "Source evidence for the SPP1-CD44 biological anchor is kept traceable.")
    return save_supp(fig, "phase14_supplementary_figure09_post_xenium_colon_sources", "Supplementary Fig. 9", "colon source")


def supp10_cross_platform() -> list[dict[str, str]]:
    lymph = read("supplementary_figure10_lymph_directionality_edge.csv")
    tonsil = read("supplementary_figure10_tonsil_A1_rank.csv")
    fdr = read("supplementary_figure10_tonsil_A1_fdr.csv")
    export("supplementary_figure10_lymph_directionality_edge", lymph)
    export("supplementary_figure10_tonsil_A1_rank", tonsil)
    export("supplementary_figure10_tonsil_A1_fdr", fdr)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})
    lymph_method_col = "method_label" if "method_label" in lymph.columns else "method"
    lymph_ss = lymph[(lymph[lymph_method_col].map(clean_method).eq("SpatialSelf")) & (~lymph["high_expression_negative"].astype(bool))].head(6)
    hbar(axs[0], lymph_ss["lr_pair"], -np.log10(lymph_ss["receiver_pairing_bh_fdr"].clip(lower=1e-4)), "lymph pairing", "a", PALETTE["signal"])
    tonsil["lr_pair"] = tonsil["ligand"].astype(str) + "-" + tonsil["receptor"].astype(str)
    tonsil_ss = tonsil[tonsil["method_label"].astype(str).str.contains("SpatialSelf", na=False)].head(6)
    hbar(axs[1], tonsil_ss["lr_pair"], tonsil_ss["rank"], "tonsil rank", "b", PALETTE["spatialself"])
    fdr["lr_pair"] = fdr["ligand"].astype(str) + "-" + fdr["receptor"].astype(str)
    rx = fdr[fdr["null_model"].eq("receiver_permutation")].head(6)
    hbar(axs[2], rx["lr_pair"], -np.log10(rx["local_contrast_bh_fdr"].clip(lower=1e-4)), "tonsil local q", "c", PALETTE["weak"])
    title(fig, "Supplementary Fig. 10 | Lymph node and tonsil source evidence", "Cross-platform evidence remains supportive and caveated.")
    return save_supp(fig, "phase14_supplementary_figure10_lymph_tonsil_sources", "Supplementary Fig. 10", "cross-platform source")


def supp11_guardrails() -> list[dict[str, str]]:
    hard = read("supplementary_figure11_hard_negative_dataset.csv")
    coord = read("supplementary_figure11_coordinate_guardrail.csv")
    edge = read("supplementary_figure11_edge_guardrail.csv")
    export("supplementary_figure11_hard_negative_dataset", hard)
    export("supplementary_figure11_coordinate_guardrail", coord)
    export("supplementary_figure11_edge_guardrail", edge)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})
    bar(axs[0], hard["dataset_key"], hard["negative_comparison_beat_fraction"], "hard negative stress", "a", PALETTE["risk_soft"])
    grouped(axs[1], coord, ["n_story_guardrail_true_lr", "n_story_guardrail_high_expression_negative"], "coordinate guardrail", "b")
    grouped(axs[2], edge, ["mean_leakage_fraction"], "edge leakage", "c")
    title(fig, "Supplementary Fig. 11 | Hard-negative and coordinate-control guardrails", "Guardrails are shown as stress tests, not final proof.")
    return save_supp(fig, "phase14_supplementary_figure11_hard_negative_coordinate_guardrails", "Supplementary Fig. 11", "guardrails")


def supp12_claim_matrix() -> list[dict[str, str]]:
    claims = read("supplementary_figure12_claim_matrix.csv")
    export("supplementary_figure12_claim_matrix", claims)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    claim_table(ax, claims.head(10))
    title(fig, "Supplementary Fig. 12 | Claim matrix and forbidden extensions", "Supported statements, caveats and forbidden extensions are separated for writing control.")
    return save_supp(fig, "phase14_supplementary_figure12_claim_matrix", "Supplementary Fig. 12", "claim matrix")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(P13 / name)


def export(name: str, df: pd.DataFrame) -> None:
    df.to_csv(SOURCE / f"{name}.csv", index=False)


def title(fig: plt.Figure, head: str, sub: str) -> None:
    fig.suptitle(head, x=0.055, y=0.985, ha="left", fontsize=9.8, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, 0.91, sub, ha="left", va="top", fontsize=6.1, color=PALETTE["muted"])
    fig.subplots_adjust(top=0.75, bottom=0.18, left=0.08, right=0.985)


def save_supp(fig: plt.Figure, stem_name: str, fig_id: str, source_key: str) -> list[dict[str, str]]:
    stem = FIG / stem_name
    save_figure(fig, stem, dpi=450)
    plt.close(fig)
    return [{"supplementary_figure": fig_id, "figure_stem": stem_name, "source_key": source_key, "status": "draft_rendered"}]


def clean_method(x: object) -> str:
    text = str(x).replace("\n", " ")
    key_map = {
        "spatialself_wavelet": "SpatialSelf",
        "spatialself_wavelet_edge_prior": "SpatialSelf",
        "spatialself_calibrated": "SpatialSelf",
        "commot": "COMMOT",
        "cellchat": "CellChat",
        "spatalk": "SpaTalk",
        "cellnest": "CellNEST",
        "deeptalk": "DeepTalk",
    }
    if text in key_map:
        return key_map[text]
    return "SpatialSelf" if text.startswith("SpatialSelf") else text


def short_dataset(x: object) -> str:
    text = str(x)
    return text.replace("human_lymph_node_visium_raw_full", "lymph node").replace("post_xenium_colon_full", "post-Xenium").replace("visium_hd_crc_full", "Visium HD").replace("_", " ")


def heat(ax: plt.Axes, df: pd.DataFrame, title_text: str, panel: str) -> None:
    panel_label(ax, panel, x=-0.12)
    arr = df.to_numpy(float)
    im = ax.imshow(arr, aspect="auto", cmap="Blues", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_title(title_text, loc="left", pad=8)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([str(c).replace("_spearman", "").replace("selected_", "").replace("background_", "") for c in df.columns], rotation=35, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([short_dataset(i) for i in df.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def bar(ax: plt.Axes, labels, values, title_text: str, panel: str, color: str) -> None:
    panel_label(ax, panel, x=-0.12)
    labels = [short_dataset(x) for x in labels]
    x = np.arange(len(labels))
    ax.bar(x, pd.Series(values).astype(float), color=color, edgecolor="white", linewidth=0.6)
    ax.set_title(title_text, loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    clean_axes(ax)


def hbar(ax: plt.Axes, labels, values, title_text: str, panel: str, color: str) -> None:
    panel_label(ax, panel, x=-0.12)
    labels = [str(x) for x in labels]
    vals = pd.Series(values).astype(float).fillna(0).to_numpy()
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=color, edgecolor="white", linewidth=0.6)
    ax.set_title(title_text, loc="left", pad=8)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.shorten(x, width=20, placeholder="...") for x in labels])
    clean_axes(ax, grid="x")


def grouped(ax: plt.Axes, df: pd.DataFrame, cols: list[str], title_text: str, panel: str) -> None:
    panel_label(ax, panel, x=-0.12)
    x = np.arange(len(df))
    width = 0.74 / max(len(cols), 1)
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        ax.bar(x - 0.37 + width / 2 + i * width, df[col].astype(float), width=width, label=col.replace("_", " "), color=[PALETTE["spatialself"], PALETTE["signal"], PALETTE["risk_soft"]][i % 3], edgecolor="white", linewidth=0.5)
    ax.set_title(title_text, loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_dataset(x) for x in df["dataset_key"]] if "dataset_key" in df.columns else range(len(df)), rotation=25, ha="right")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=5.0)


def ordered_bar(ax: plt.Axes, df: pd.DataFrame, label_col: str, value_col: str, title_text: str, panel: str, *, log: bool = False) -> None:
    panel_label(ax, panel, x=-0.12)
    sub = df.copy()
    x = np.arange(len(sub))
    ax.bar(x, sub[value_col].astype(float), color=[METHOD_COLORS.get(str(m), PALETTE["neutral"]) for m in sub[label_col]], edgecolor="white", linewidth=0.6)
    ax.set_title(title_text, loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub[label_col], rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
    clean_axes(ax)


def tradeoff(ax: plt.Axes, df: pd.DataFrame, panel: str) -> None:
    panel_label(ax, panel, x=-0.08)
    df = df.copy()
    method_col = "method_label" if "method_label" in df.columns else "method"
    df["method_clean"] = df[method_col].map(clean_method)
    for _, row in df.iterrows():
        method = row["method_clean"]
        ax.scatter(row["empirical_fdr_at_threshold_mean"], row["weak_signal_recall_at_threshold_mean"], s=90, color=METHOD_COLORS.get(method, PALETTE["neutral"]), edgecolor="white", linewidth=0.6)
        ax.text(row["empirical_fdr_at_threshold_mean"] + 0.015, row["weak_signal_recall_at_threshold_mean"] + 0.015, method, fontsize=5.4)
    ax.set_title("weak recall vs FDR", loc="left", pad=8)
    ax.set_xlabel("empirical FDR")
    ax.set_ylabel("weak recall")
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.04, 1.08)
    clean_axes(ax)


def status_matrix(ax: plt.Axes, deep: pd.DataFrame, title_text: str, panel: str) -> None:
    panel_label(ax, panel, x=-0.04)
    ax.set_axis_off()
    ax.set_title(title_text, loc="left", pad=8)
    cols = ["code_availability", "dependency_reproducibility", "lr_pair_ranking_support", "runtime_memory_status"]
    rows = []
    for _, row in deep.iterrows():
        vals = [str(row.get(c, "not audited")) for c in cols]
        rows.append([row["method"]] + vals)
    table = ax.table(cellText=rows, colLabels=["method", "code", "dependency", "LR rank", "runtime"], loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(5.2)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["neutral_soft"])
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor(PALETTE["soft_bg"])
            cell.set_text_props(weight="bold")


def claim_table(ax: plt.Axes, claims: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.04)
    ax.set_axis_off()
    cols = [c for c in ["topic", "support_status", "caveat_or_forbidden_extension"] if c in claims.columns]
    sub = claims[cols].copy()
    for col in sub.columns:
        sub[col] = sub[col].astype(str).map(lambda x: "\n".join(textwrap.wrap(x, 32)))
    table = ax.table(cellText=sub.values, colLabels=sub.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(4.8)
    table.scale(1.0, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["neutral_soft"])
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor(PALETTE["soft_bg"])
            cell.set_text_props(weight="bold")


if __name__ == "__main__":
    main()
