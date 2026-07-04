#!/usr/bin/env python
"""Render Phase 13 Figure 2: baseline comparison and hard-negative suppression."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from phase13_figure_style import PALETTE, apply_style, panel_label, save_figure, soften_axes


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "phase13" / "figures"
SOURCE_OUT = ROOT / "outputs" / "phase13" / "source_data"

SOURCE_FILES = {
    "formal_summary": ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_summary.csv",
    "formal_metrics": ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_metrics.csv",
    "formal_status": ROOT / "outputs/phase2_revised/goal2_formal_baseline_panel_all/goal2_formal_baseline_status.csv",
    "weak_tradeoff_gate": ROOT
    / "outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_method_gate.csv",
    "weak_tradeoff_metrics": ROOT / "outputs/phase2_revised/goal2_weak_signal_tradeoff/goal2_weak_signal_tradeoff_metrics.csv",
    "runtime": ROOT / "outputs/phase2_revised/goal7_runtime_scalability/goal7_method_runtime_profile.csv",
    "deep_status": ROOT / "outputs/phase8/manuscript_grade_evidence_consolidation/phase8_deep_baseline_closure_table.csv",
}

METHOD_ORDER = [
    "spatialself_wavelet",
    "spatialself_wavelet_edge_prior",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
    "deeptalk",
]
METHOD_LABEL = {
    "spatialself_wavelet": "SpatialSelf",
    "spatialself_wavelet_edge_prior": "SpatialSelf\n+ edge prior",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
    "nichenet": "NicheNet",
    "niches": "NICHES",
    "holonet": "HoloNet",
    "stmlnet": "stMLnet",
    "cytosignal": "CytoSignal",
}
METHOD_COLOR = {
    "spatialself_wavelet": PALETTE["spatialself"],
    "spatialself_wavelet_edge_prior": PALETTE["wavelet"],
    "commot": "#626262",
    "cellchat": "#9A9A9A",
    "spatalk": "#C58E3C",
    "cellnest": "#8A6AAE",
    "deeptalk": "#B8BDC4",
}
DATASET_LABEL = {
    "controlled_v2_formal": "Controlled\nsimulation",
    "real_background_v2_formal": "Real-background\nsemi-synthetic",
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    source_tables = build_source_tables(data)
    export_source_tables(source_tables)

    fig = plt.figure(figsize=(7.2, 7.15), constrained_layout=False)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.92, 1.0, 0.92],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.07,
        right=0.985,
        top=0.895,
        bottom=0.075,
        wspace=0.42,
        hspace=0.62,
    )

    ax_design = fig.add_subplot(gs[0, 0])
    ax_aupr = fig.add_subplot(gs[0, 1:])
    ax_trade = fig.add_subplot(gs[1, 0:2])
    ax_fp = fig.add_subplot(gs[1, 2])
    ax_runtime = fig.add_subplot(gs[2, 0:2])
    ax_deep = fig.add_subplot(gs[2, 2])

    draw_method_design(ax_design, source_tables["status_compact"])
    draw_formal_aupr(ax_aupr, source_tables["formal_summary_compact"])
    draw_weak_fdr_tradeoff(ax_trade, source_tables["weak_tradeoff_gate"])
    draw_hard_negative(ax_fp, source_tables["formal_summary_compact"])
    draw_runtime(ax_runtime, source_tables["runtime_compact"])
    draw_deep_status(ax_deep, source_tables["deep_status_compact"])

    fig.suptitle(
        "SpatialSelf improves benchmark recovery while suppressing hard false-positive modes",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=10.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.07,
        0.952,
        "Main baseline figure draft. CellChat/SpaTalk are projected adapter-light baselines; DeepTalk is status-only for LR-pair ranking unless a comparable predict_ccc output exists.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    out_stem = OUT_DIR / "phase13_figure2_baseline_hardnegative"
    save_figure(fig, out_stem)
    write_manifest(out_stem)
    plt.close(fig)
    print(f"wrote {out_stem}.svg/.pdf/.png")


def load_data() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 source files: " + "; ".join(missing))
    return {key: pd.read_csv(path) for key, path in SOURCE_FILES.items()}


def build_source_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    formal = data["formal_summary"].copy()
    formal = formal[formal["method"].isin(METHOD_ORDER)].copy()
    formal["method_label"] = formal["method"].map(METHOD_LABEL).fillna(formal["method"])
    formal["dataset_label"] = formal["dataset"].map(DATASET_LABEL).fillna(formal["dataset"])
    formal["is_spatialself"] = formal["method"].str.startswith("spatialself")

    status = data["formal_status"].copy()
    status_compact = summarize_status(status)

    weak_gate = data["weak_tradeoff_gate"].copy()
    weak_gate = weak_gate[weak_gate["method"].isin(METHOD_ORDER)].copy()
    weak_gate["method_label"] = weak_gate["method"].map(METHOD_LABEL).fillna(weak_gate["method"])
    weak_gate["is_spatialself"] = weak_gate["method"].str.startswith("spatialself")

    runtime = data["runtime"].copy()
    runtime_compact = runtime[runtime["method"].isin(METHOD_ORDER + ["nichenet", "niches"])].copy()
    runtime_compact["method_label"] = runtime_compact["method"].map(METHOD_LABEL).fillna(runtime_compact["method"])

    deep = data["deep_status"].copy()
    deep["method_key"] = deep["method"].str.lower()
    deep["method_label"] = deep["method_key"].map(METHOD_LABEL).fillna(deep["method"])
    deep_compact = deep[
        [
            "method",
            "method_label",
            "phase7_final_status",
            "code_availability",
            "dependency_reproducibility",
            "lr_pair_ranking_support",
            "runtime_memory_status",
            "manuscript_use",
            "limitation",
        ]
    ].copy()

    return {
        "formal_summary_compact": formal,
        "formal_metrics": data["formal_metrics"],
        "status_compact": status_compact,
        "weak_tradeoff_gate": weak_gate,
        "weak_tradeoff_metrics": data["weak_tradeoff_metrics"],
        "runtime_compact": runtime_compact,
        "deep_status_compact": deep_compact,
    }


def summarize_status(status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = status[status["method"].eq(method)]
        if sub.empty:
            level = "status-only" if method == "deeptalk" else "not audited"
            detail = "No formal comparable LR-edge score in current panel."
            n_dataset = 0
        else:
            levels = sub["baseline_evidence_level"].fillna("unknown").astype(str)
            if (levels == "candidate_edge_metric").any():
                level = "candidate-edge"
            elif levels.str.contains("projected", case=False, na=False).any():
                level = "adapter-light"
            elif levels.str.contains("adapter_light", case=False, na=False).any():
                level = "adapter-light"
            elif levels.str.contains("model_required|status", case=False, na=False).any():
                level = "status-only"
            else:
                level = "status-only"
            detail = "; ".join(sorted(sub["status"].astype(str).unique())[:2])
            n_dataset = sub["dataset"].nunique()
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "evidence_level": level,
                "n_datasets": n_dataset,
                "status_detail": detail,
            }
        )
    return pd.DataFrame(rows)


def export_source_tables(source_tables: dict[str, pd.DataFrame]) -> None:
    for key, table in source_tables.items():
        table.to_csv(SOURCE_OUT / f"figure2_{key}.csv", index=False)


def draw_method_design(ax: plt.Axes, status: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.08, y=1.03)
    ax.set_axis_off()
    ax.set_title("Baseline comparison contract", loc="left", pad=8)
    level_color = {
        "candidate-edge": PALETTE["spatialself"],
        "adapter-light": PALETTE["weak"],
        "status-only": PALETTE["neutral"],
        "not audited": PALETTE["neutral"],
    }
    y0 = 0.82
    for i, row in enumerate(status.itertuples(index=False)):
        y = y0 - i * 0.105
        color = level_color.get(row.evidence_level, PALETTE["neutral"])
        ax.add_patch(
            patches.Circle((0.06, y), 0.025, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.7)
        )
        ax.text(
            0.11,
            y,
            str(row.method_label).replace("\n", " "),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.6,
            color=PALETTE["ink"],
            fontweight="bold" if str(row.method).startswith("spatialself") else "normal",
        )
        ax.text(
            0.62,
            y,
            row.evidence_level,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color=PALETTE["muted"],
        )
    legend_y = 0.045
    for x, label, color in [
        (0.02, "comparable edge scores", PALETTE["spatialself"]),
        (0.02, "adapter-light", PALETTE["weak"]),
        (0.02, "status-only", PALETTE["neutral"]),
    ]:
        ax.add_patch(patches.Circle((x, legend_y), 0.015, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(x + 0.04, legend_y, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=PALETTE["muted"])
        legend_y -= 0.06


def draw_formal_aupr(ax: plt.Axes, formal: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.045, y=1.03)
    ax.set_title("Formal benchmark performance", loc="left", pad=8)
    df = formal[formal["method"].isin(METHOD_ORDER[:-1])].copy()
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER[:-1], ordered=True)
    df = df.sort_values(["dataset", "method"])
    datasets = ["controlled_v2_formal", "real_background_v2_formal"]
    x = np.arange(len(METHOD_ORDER[:-1]))
    width = 0.34
    offsets = [-width / 2, width / 2]
    for j, dataset in enumerate(datasets):
        sub = df[df["dataset"].eq(dataset)].set_index("method").reindex(METHOD_ORDER[:-1])
        vals = sub["aupr_mean"].to_numpy(float)
        err = sub["aupr_std"].fillna(0).to_numpy(float)
        bars = ax.bar(
            x + offsets[j],
            vals,
            width=width,
            yerr=err,
            color=["#EAF2FA" if j == 0 else "#D4E5F5"] * len(x),
            edgecolor=[METHOD_COLOR.get(m, PALETTE["neutral"]) for m in METHOD_ORDER[:-1]],
            linewidth=1.0,
            error_kw=dict(lw=0.6, capsize=2, capthick=0.6, ecolor=PALETTE["muted"]),
            label=DATASET_LABEL[dataset].replace("\n", " "),
        )
        for bar, method in zip(bars, METHOD_ORDER[:-1]):
            if method.startswith("spatialself"):
                bar.set_facecolor(METHOD_COLOR[method])
                bar.set_alpha(0.95)
            else:
                bar.set_alpha(0.78)
    ax.set_ylabel("AUPR")
    ax.set_ylim(0, max(0.48, df["aupr_mean"].max() * 1.22))
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m].replace("\n", " ") for m in METHOD_ORDER[:-1]], rotation=35, ha="right")
    soften_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=5.8)
    ax.text(
        0.01,
        0.97,
        "mean ± s.d., n=3 per benchmark",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=PALETTE["muted"],
    )


def draw_weak_fdr_tradeoff(ax: plt.Axes, weak_gate: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.045, y=1.03)
    ax.set_title("Weak-signal recovery must be read with FDR", loc="left", pad=8)
    df = weak_gate[weak_gate["method"].isin(METHOD_ORDER[:-1])].copy()
    ax.axvspan(0, 0.55, color="#F2F7FC", zorder=0)
    ax.axhspan(0.2, 1.02, color="#FBF7EC", zorder=0, alpha=0.65)
    for _, row in df.iterrows():
        method = row["method"]
        x = row["empirical_fdr_at_threshold_mean"]
        y = row["weak_signal_recall_at_threshold_mean"]
        size = 180 + 520 * max(float(row["aupr_mean"]), 0)
        color = METHOD_COLOR.get(method, PALETTE["neutral"])
        edge = PALETTE["ink"] if str(method).startswith("spatialself") else "white"
        ax.scatter(x, y, s=size, color=color, edgecolor=edge, linewidth=0.8, alpha=0.9, zorder=3)
        dx = -0.035 if x > 0.75 else 0.018
        dy = -0.055 if y > 0.85 else 0.025
        ax.text(
            x + dx,
            y + dy,
            METHOD_LABEL[method].replace("\n", " "),
            fontsize=5.8,
            ha="right" if x > 0.75 else "left",
            va="center",
            color=PALETTE["ink"],
        )
    ax.annotate(
        "high raw recall\nbut high FDR",
        xy=(0.82, 0.97),
        xytext=(0.57, 0.76),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=0.7),
        fontsize=5.9,
        ha="left",
        va="center",
        color=PALETTE["muted"],
    )
    ax.annotate(
        "better practical zone:\nnonzero weak recovery,\nlower FDR / FP",
        xy=(0.52, 0.21),
        xytext=(0.08, 0.46),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["spatialself"], lw=0.8),
        fontsize=5.9,
        ha="left",
        va="center",
        color=PALETTE["spatialself_dark"],
    )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Empirical FDR at threshold")
    ax.set_ylabel("Weak-signal recall at threshold")
    soften_axes(ax)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.65)
    size_handle = Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=PALETTE["muted"], markersize=8)
    ax.legend([size_handle], ["circle area scales with AUPR"], frameon=False, loc="lower right", fontsize=5.6)


def draw_hard_negative(ax: plt.Axes, formal: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.09, y=1.03)
    ax.set_title("High-expression false-positive suppression", loc="left", pad=8)
    df = formal[formal["dataset"].eq("real_background_v2_formal") & formal["method"].isin(METHOD_ORDER[:-1])].copy()
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER[:-1], ordered=True)
    df = df.sort_values("method")
    y = np.arange(len(df))[::-1]
    vals = df["false_positive_high_expression_rate_at_threshold_mean"].to_numpy(float)
    errs = df["false_positive_high_expression_rate_at_threshold_std"].fillna(0).to_numpy(float)
    colors = [METHOD_COLOR.get(m, PALETTE["neutral"]) if str(m).startswith("spatialself") else "#C8C8C8" for m in df["method"]]
    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="white", linewidth=0.6, error_kw=dict(lw=0.6, capsize=2, ecolor=PALETTE["muted"]))
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[m].replace("\n", " ") for m in df["method"]])
    ax.set_xlabel("FP rate")
    ax.set_xlim(0, max(0.22, vals.max() * 1.25))
    soften_axes(ax)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", visible=False)
    for yy, v in zip(y, vals):
        ax.text(v + 0.006, yy, f"{v:.3f}", ha="left", va="center", fontsize=5.7, color=PALETTE["muted"])
    ax.text(
        0.02,
        0.02,
        "Real-background benchmark",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["muted"],
    )


def draw_runtime(ax: plt.Axes, runtime: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.045, y=1.03)
    ax.set_title("Runtime profile on retained high-resolution benchmark", loc="left", pad=8)
    order = ["spatialself_wavelet", "spatialself_wavelet_edge_prior", "deeptalk", "cellchat", "cellnest", "spatalk", "commot"]
    df = runtime[runtime["method"].isin(order)].copy()
    df["method"] = pd.Categorical(df["method"], order, ordered=True)
    df = df.sort_values("method")
    x = np.arange(len(df))
    vals = df["wall_clock_seconds"].astype(float).to_numpy()
    colors = [METHOD_COLOR.get(m, PALETTE["neutral"]) if str(m).startswith("spatialself") else "#BFC3C8" for m in df["method"]]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (s, log)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m].replace("\n", " ") for m in df["method"]], rotation=35, ha="right")
    soften_axes(ax)
    ax.text(
        0.01,
        0.98,
        "n=4,035 cells/spots; 332,328 candidate edges; runtime adapters are not equivalent to formal LR-pair scoring for DeepTalk.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=PALETTE["muted"],
    )
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.12, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5, color=PALETTE["muted"])


def draw_deep_status(ax: plt.Axes, deep: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.09, y=1.03)
    ax.set_axis_off()
    ax.set_title("Deep baseline closure", loc="left", pad=8)
    rows = deep.copy()
    rows["is_usable"] = rows["lr_pair_ranking_support"].astype(str).str.contains("usable", case=False, na=False) & ~rows[
        "lr_pair_ranking_support"
    ].astype(str).str.contains("not_usable", case=False, na=False)
    y0 = 0.82
    for i, row in enumerate(rows.itertuples(index=False)):
        y = y0 - i * 0.15
        support = str(row.lr_pair_ranking_support)
        color = PALETTE["success"] if bool(row.is_usable) else PALETTE["neutral"]
        ax.add_patch(patches.Circle((0.06, y), 0.025, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.7))
        ax.text(0.11, y + 0.025, row.method_label, transform=ax.transAxes, ha="left", va="center", fontsize=6.5, fontweight="bold", color=PALETTE["ink"])
        wrapped = "\n".join(textwrap.wrap(short_status(support), width=31))
        ax.text(0.11, y - 0.026, wrapped, transform=ax.transAxes, ha="left", va="center", fontsize=5.35, color=PALETTE["muted"], linespacing=1.05)
    ax.text(
        0.02,
        0.02,
        "Reported as reproducibility/status evidence,\nnot as scored LR-pair baseline when output is absent.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=PALETTE["muted"],
    )


def short_status(text: str) -> str:
    text = text.replace("_", " ")
    replacements = {
        "not usable currently": "not usable for LR-pair ranking",
        "not audited": "not audited in current panel",
        "unknown or not formal candidate edge": "unknown/not formal edge output",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_manifest(out_stem: Path) -> None:
    rows = [
        {
            "figure": "Figure 2",
            "panel": "a",
            "title": "Baseline comparison contract",
            "source_data": "outputs/phase13/source_data/figure2_status_compact.csv",
            "claim_supported": "Comparable and adapter-light/status-only baselines are explicitly separated.",
        },
        {
            "figure": "Figure 2",
            "panel": "b",
            "title": "Formal benchmark performance",
            "source_data": "outputs/phase13/source_data/figure2_formal_summary_compact.csv",
            "claim_supported": "SpatialSelf has strong AUPR across controlled and real-background benchmarks.",
        },
        {
            "figure": "Figure 2",
            "panel": "c",
            "title": "Weak-signal/FDR tradeoff",
            "source_data": "outputs/phase13/source_data/figure2_weak_tradeoff_gate.csv",
            "claim_supported": "High raw weak recall in some baselines coincides with high empirical FDR; SpatialSelf supports a more conservative tradeoff.",
        },
        {
            "figure": "Figure 2",
            "panel": "d",
            "title": "High-expression false-positive suppression",
            "source_data": "outputs/phase13/source_data/figure2_formal_summary_compact.csv",
            "claim_supported": "SpatialSelf suppresses high-expression false positives in the real-background formal benchmark.",
        },
        {
            "figure": "Figure 2",
            "panel": "e",
            "title": "Runtime profile",
            "source_data": "outputs/phase13/source_data/figure2_runtime_compact.csv",
            "claim_supported": "SpatialSelf runtime is practical relative to retained runnable baselines on the retained high-resolution benchmark.",
        },
        {
            "figure": "Figure 2",
            "panel": "f",
            "title": "Deep baseline closure",
            "source_data": "outputs/phase13/source_data/figure2_deep_status_compact.csv",
            "claim_supported": "DeepTalk/HoloNet/stMLnet/CytoSignal/NICHES status and limitations are reported transparently.",
        },
    ]
    pd.DataFrame(rows).to_csv(f"{out_stem}_manifest.csv", index=False)


if __name__ == "__main__":
    main()
