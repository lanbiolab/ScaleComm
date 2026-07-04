#!/usr/bin/env python
"""Build Fig. 5 draft: LR-specific scale/range/context mechanism validation.

This script uses existing scDesign3 mechanism-benchmark outputs and SpatialSelf
tuning summaries. It does not rerun any baseline model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
BENCH_200 = ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
TUNING = ROOT / "outputs/reference_based_benchmark/spatialself_mechanism_tuning"
OUT = ROOT / "outputs/current_fig5_lr_scale_mechanism"
SOURCE = OUT / "source_data"
FIGDIR = OUT / "figures"


RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABEL = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}
MECH_LABEL = {
    "juxtacrine_contact": "Contact",
    "short_range_paracrine": "Short",
    "medium_range_paracrine": "Medium",
    "long_range_chemokine_niche": "Long niche",
}
METHOD_LABEL = {
    "spatialself_range_context": "SpatialSelf",
    "SpatialSelf": "SpatialSelf",
    "commot": "COMMOT",
    "COMMOT": "COMMOT",
    "cellchat": "CellChat",
    "CellChat": "CellChat",
    "cellnest": "CellNEST",
    "CellNEST": "CellNEST",
    "deeptalk": "DeepTalk",
    "DeepTalk": "DeepTalk",
    "spatalk": "SpaTalk",
    "SpaTalk": "SpaTalk",
    "cellphonedb": "CellPhoneDB",
    "CellPhoneDB": "CellPhoneDB",
    "niches": "NICHES",
    "NICHES": "NICHES",
    "cytosignal": "CytoSignal",
    "CytoSignal": "CytoSignal",
    "spaotsc": "SpaOTsc",
    "SpaOTsc": "SpaOTsc",
}
METHOD_ORDER = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellChat",
    "CellPhoneDB",
    "NICHES",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
    "SpaTalk",
]

COL = {
    "ink": "#2B2826",
    "muted": "#6F6860",
    "grid": "#E5E0D7",
    "light": "#F5F2EB",
    "panel": "#FFFFFF",
    "ours": "#237A78",
    "ours_dark": "#155C5B",
    "baseline": "#8A8177",
    "baseline_light": "#C8C1B8",
    "contact": "#8A8177",
    "short": "#D9992B",
    "medium": "#C94F48",
    "long_niche": "#237A78",
    "decoy": "#B85450",
    "blue": "#2F78A8",
    "sand": "#E6C06A",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "text.color": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "axes.edgecolor": COL["ink"],
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str, x: float = -0.06, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COL["ink"],
    )


def clean_axis(ax, grid_y: bool = True) -> None:
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
    if grid_y:
        ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)
    else:
        ax.grid(False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_distance_profiles() -> pd.DataFrame:
    frames = []
    for f in sorted((BENCH_200 / "method_runs/range_mechanism").glob("*/*_distance_bin_summary.csv")):
        df = read_csv(f)
        dataset_id = f.parent.name
        df["dataset_id"] = dataset_id
        frames.append(df)
    distance = pd.concat(frames, ignore_index=True)
    distance = distance[distance["range_class"].isin(RANGE_ORDER)].copy()
    distance = distance[distance["method"].map(METHOD_LABEL).eq("SpatialSelf")].copy()
    distance = distance.sort_values(["dataset_id", "lr_id", "median_distance"])
    distance["distance_bin_index"] = distance.groupby(["dataset_id", "lr_id"]).cumcount() + 1

    def scale01(s: pd.Series) -> pd.Series:
        vmax = float(s.max())
        if vmax <= 0 or np.isnan(vmax):
            return s * 0.0
        return s / vmax

    distance["score_norm"] = distance.groupby(["dataset_id", "lr_id"])["mean_score"].transform(scale01)
    distance["truth_norm"] = distance.groupby(["dataset_id", "lr_id"])["truth_rate"].transform(scale01)
    agg = (
        distance.groupby(["range_class", "distance_bin_index"], as_index=False)
        .agg(
            predicted_activity=("score_norm", "mean"),
            truth_enrichment=("truth_norm", "mean"),
            median_distance=("median_distance", "mean"),
            n_lr=("lr_id", "nunique"),
        )
        .sort_values(["range_class", "distance_bin_index"])
    )
    agg["range_label"] = agg["range_class"].map(RANGE_LABEL)
    return agg


def build_mechanism_heatmap() -> pd.DataFrame:
    mech = read_csv(BENCH_200 / "method_runs/scdesign3_200lr_mechanism_summary.csv")
    mech = mech[mech["range_class"].isin(RANGE_ORDER)].copy()
    mech["method_label"] = mech["method"].map(METHOD_LABEL).fillna(mech["method"])
    table = (
        mech.groupby(["method_label", "range_class"], as_index=False)
        .agg(aupr=("aupr", "mean"), top100_precision=("top100_precision", "mean"), fdr_top5pct=("fdr_top5pct", "mean"))
    )
    table["range_label"] = table["range_class"].map(RANGE_LABEL)
    table["method_label"] = pd.Categorical(table["method_label"], METHOD_ORDER, ordered=True)
    table["range_class"] = pd.Categorical(table["range_class"], RANGE_ORDER, ordered=True)
    return table.sort_values(["method_label", "range_class"])


def build_top_call_composition() -> pd.DataFrame:
    datasets = {
        "lymph_scdesign3_200lr": BENCH_200
        / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00/candidate_interactions.csv",
        "melanoma_scdesign3_200lr": BENCH_200
        / "datasets/reference_melanoma_scdesign3_200lr_replicate_00/candidate_interactions.csv",
    }
    rows = []
    for dataset_id, cand_path in datasets.items():
        cand = read_csv(cand_path)
        run_dir = BENCH_200 / "method_runs" / dataset_id
        for pred_path in sorted(run_dir.glob("*/*_predictions.csv")):
            pred = read_csv(pred_path)
            if pred.empty:
                continue
            method = METHOD_LABEL.get(str(pred["method"].iloc[0]), str(pred["method"].iloc[0]))
            dat = cand.merge(pred[["candidate_id", "score"]], on="candidate_id", how="inner")
            if dat.empty:
                continue
            threshold = dat["score"].quantile(0.95)
            selected = dat[dat["score"] >= threshold].copy()
            n = max(len(selected), 1)
            truth = selected["ground_truth"].astype(bool)
            wrong_range = selected["wrong_range_false_positive"].astype(bool)
            wrong_niche = (~selected["niche_rule_satisfied"].astype(bool)) & (~truth)
            high_expr = selected["false_positive_high_expression"].astype(bool) & (~truth)
            other_fp = (~truth) & (~wrong_range) & (~wrong_niche) & (~high_expr)
            rows.extend(
                [
                    {
                        "dataset_id": dataset_id,
                        "method_label": method,
                        "component": "True",
                        "fraction": truth.sum() / n,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_label": method,
                        "component": "Wrong range",
                        "fraction": wrong_range.sum() / n,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_label": method,
                        "component": "Wrong niche",
                        "fraction": wrong_niche.sum() / n,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_label": method,
                        "component": "High-expression decoy",
                        "fraction": high_expr.sum() / n,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_label": method,
                        "component": "Other false",
                        "fraction": other_fp.sum() / n,
                    },
                ]
            )
    comp = pd.DataFrame(rows)
    comp = comp.groupby(["method_label", "component"], as_index=False).agg(fraction=("fraction", "mean"))
    comp["method_label"] = pd.Categorical(comp["method_label"], METHOD_ORDER, ordered=True)
    return comp.sort_values(["method_label", "component"])


def build_decoy_summary() -> pd.DataFrame:
    mech = read_csv(BENCH_200 / "method_runs/scdesign3_200lr_mechanism_summary.csv")
    mech = mech[mech["range_class"].isin(RANGE_ORDER)].copy()
    mech["method_label"] = mech["method"].map(METHOD_LABEL).fillna(mech["method"])
    wrong = (
        mech.groupby("method_label", as_index=False)
        .agg(
            wrong_range_rate_top5pct=("wrong_range_rate_top5pct", "mean"),
            fdr_top5pct=("fdr_top5pct", "mean"),
            aupr=("aupr", "mean"),
        )
        .sort_values("wrong_range_rate_top5pct")
    )
    long_close = (
        mech[mech["range_class"].eq("long_niche")]
        .groupby("method_label", as_index=False)
        .agg(too_close_long_range_rate_top5pct=("too_close_long_range_rate_top5pct", "mean"))
    )
    wrong = wrong.merge(long_close, on="method_label", how="left")
    wrong["method_label"] = pd.Categorical(wrong["method_label"], METHOD_ORDER, ordered=True)
    return wrong.sort_values("wrong_range_rate_top5pct")


def build_effective_range_alignment() -> pd.DataFrame:
    by_mech = read_csv(BENCH_200 / "method_runs/scdesign3_200lr_by_mechanism.csv")
    by_mech = by_mech[by_mech["range_class"].isin(RANGE_ORDER)].copy()
    by_mech["method_label"] = by_mech["method"].map(METHOD_LABEL).fillna(by_mech["method"])
    subset = by_mech[by_mech["method_label"].isin(["SpatialSelf", "COMMOT", "SpaOTsc", "CellNEST"])].copy()
    subset["range_label"] = subset["range_class"].map(RANGE_LABEL)
    return subset[
        [
            "dataset_id",
            "tissue",
            "method_label",
            "range_class",
            "range_label",
            "true_positive_median_distance",
            "predicted_top_median_distance",
            "distance_median_error",
            "distance_profile_emd",
        ]
    ].sort_values(["method_label", "dataset_id", "range_class"])


def build_ablation_tradeoff() -> pd.DataFrame:
    tuning = read_csv(TUNING / "spatialself_tuning_summary.csv")
    wrong_cols = [c for c in tuning.columns if c.startswith("wrong_range_rate_top5pct__")]
    fdr_cols = [c for c in tuning.columns if c.startswith("fdr_top5pct__")]
    aupr_cols = [c for c in tuning.columns if c.startswith("aupr__")]
    tuning = tuning.copy()
    tuning["mean_wrong_range_top5pct"] = tuning[wrong_cols].mean(axis=1, skipna=True)
    tuning["mean_fdr_top5pct"] = tuning[fdr_cols].mean(axis=1, skipna=True)
    tuning["mean_mechanism_aupr"] = tuning[aupr_cols].mean(axis=1, skipna=True)
    label = {
        "range_context_guard": "Final gate",
        "range_context_spatial_null": "Spatial-null",
        "adaptive_wavelet_scale": "Wavelet scale",
        "range_context_mechanism_prior_fixed": "Fixed prior",
        "adaptive_wavelet_scale_contact": "Wavelet+contact",
        "range_context_contact_kernel": "Range+contact",
        "adaptive_contact_kernel": "Contact kernel",
    }
    tuning["variant_label"] = tuning["run_id"].map(label).fillna(tuning["run_id"])
    return tuning[
        [
            "run_id",
            "variant_label",
            "aupr",
            "top_k_precision",
            "empirical_fdr_at_threshold",
            "mean_mechanism_aupr",
            "mean_wrong_range_top5pct",
            "mean_fdr_top5pct",
            "selection_score",
        ]
    ].sort_values("selection_score", ascending=False)


def write_source_data() -> dict[str, pd.DataFrame]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    tables = {
        "panel_b_truth_design": pd.DataFrame(
            [
                {
                    "range_class": "contact",
                    "range_label": "Contact",
                    "valid_range_start": 0.00,
                    "valid_range_end": 0.12,
                    "context_required": "adjacent receiver",
                    "decoys": "far high-expression contact pairs",
                },
                {
                    "range_class": "short",
                    "range_label": "Short",
                    "valid_range_start": 0.06,
                    "valid_range_end": 0.22,
                    "context_required": "permissive local tissue",
                    "decoys": "too-far paracrine pairs",
                },
                {
                    "range_class": "medium",
                    "range_label": "Medium",
                    "valid_range_start": 0.12,
                    "valid_range_end": 0.34,
                    "context_required": "permissive local tissue",
                    "decoys": "wrong-range high-expression pairs",
                },
                {
                    "range_class": "long_niche",
                    "range_label": "Long niche",
                    "valid_range_start": 0.22,
                    "valid_range_end": 0.55,
                    "context_required": "specific spatial niche",
                    "decoys": "too-close or wrong-niche chemokine pairs",
                },
            ]
        ),
        "panel_c_score_distance_profiles": build_distance_profiles(),
        "panel_d_top_call_composition": build_top_call_composition(),
        "panel_e_mechanism_aupr": build_mechanism_heatmap(),
        "panel_f_wrong_range_suppression": build_decoy_summary(),
        "panel_g_effective_range_alignment": build_effective_range_alignment(),
        "panel_h_scale_context_ablation": build_ablation_tradeoff(),
    }
    for name, df in tables.items():
        df.to_csv(SOURCE / f"{name}.csv", index=False)
    manifest = {
        "core_conclusion": (
            "SpatialSelf models LR-specific spatial scale and context: predicted activities align "
            "with mechanism-specific ranges and suppress wrong-range/wrong-niche decoys."
        ),
        "source": "Existing scDesign3 200-LR mechanism benchmark and SpatialSelf tuning outputs.",
        "claim_boundary": (
            "The panels support alignment with simulated LR range/context rules; they do not prove "
            "that learned gates are identical to real biophysical signaling ranges."
        ),
        "panels": {name: f"source_data/{name}.csv" for name in tables},
    }
    with open(OUT / "fig5_lr_scale_mechanism_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return tables


def draw_panel_a(ax) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "a", x=-0.04, y=1.02)
    ax.text(0.02, 0.96, "LR-specific scale and context gating", fontsize=8.2, fontweight="bold", ha="left")

    # Left: graph-wavelet bank.
    bank_x = 0.03
    for i, (lab, alpha) in enumerate(zip(["S0", "S1", "S2", "S4", "S8"], [0.35, 0.45, 0.58, 0.72, 0.86], strict=False)):
        y = 0.72 - i * 0.095
        rect = FancyBboxPatch(
            (bank_x + i * 0.012, y),
            0.16,
            0.055,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.45,
            edgecolor=COL["baseline"],
            facecolor=mpl.colors.to_rgba(COL["baseline_light"], alpha),
        )
        ax.add_patch(rect)
        ax.text(bank_x + 0.08 + i * 0.012, y + 0.028, lab, fontsize=6.1, ha="center", va="center")
    ax.text(bank_x + 0.085, 0.80, "wavelet\nscale bank", fontsize=6.1, ha="center", va="bottom", color=COL["muted"], linespacing=0.82)

    # LR identity block.
    lr_box = FancyBboxPatch(
        (0.35, 0.62),
        0.17,
        0.105,
        boxstyle="round,pad=0.01,rounding_size=0.016",
        linewidth=0.6,
        edgecolor=COL["ours_dark"],
        facecolor="#E3F1EF",
    )
    ax.add_patch(lr_box)
    ax.text(0.435, 0.674, "LR identity", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COL["ours_dark"])
    ax.text(0.435, 0.636, "e.g. CCL19-CCR7", ha="center", va="center", fontsize=5.5, color=COL["muted"])

    gate_box = FancyBboxPatch(
        (0.33, 0.36),
        0.22,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.6,
        edgecolor=COL["ours_dark"],
        facecolor="#F6FBFA",
    )
    ax.add_patch(gate_box)
    ax.text(0.44, 0.455, "scale / range gate", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COL["ours_dark"])
    xs = np.linspace(0.365, 0.515, 5)
    weights = [0.10, 0.18, 0.35, 0.25, 0.12]
    for x, w in zip(xs, weights, strict=False):
        ax.add_line(Line2D([x, x], [0.382, 0.382 + 0.06 * w / max(weights)], lw=2.0, color=COL["ours"]))
    ax.text(0.44, 0.374, "LR-dependent weights", ha="center", va="top", fontsize=5.3, color=COL["muted"])

    context_box = FancyBboxPatch(
        (0.665, 0.48),
        0.245,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.6,
        edgecolor=COL["blue"],
        facecolor="#E9F2F6",
    )
    ax.add_patch(context_box)
    ax.text(0.787, 0.585, "context gate", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COL["blue"])
    ax.text(0.787, 0.528, "sender/receiver\n+ distance/niche", ha="center", va="center", fontsize=5.4, color=COL["muted"], linespacing=0.88)

    out = Circle((0.79, 0.25), 0.055, facecolor=COL["ours"], edgecolor=COL["ours_dark"], linewidth=0.6)
    ax.add_patch(out)
    ax.text(0.79, 0.25, "score", ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")

    for xy1, xy2 in [((0.22, 0.54), (0.33, 0.43)), ((0.435, 0.62), (0.44, 0.50)), ((0.55, 0.43), (0.665, 0.57)), ((0.787, 0.48), (0.79, 0.31))]:
        ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=7, linewidth=0.55, color=COL["muted"]))

    # Mini learned scale examples.
    y0 = 0.08
    for i, rc in enumerate(RANGE_ORDER):
        x = 0.07 + i * 0.205
        ax.text(x, y0 + 0.115, RANGE_LABEL[rc], fontsize=5.9, ha="center", color=COL[rc], fontweight="bold")
        if rc == "contact":
            vals = [1.0, 0.42, 0.05, 0.02]
        elif rc == "short":
            vals = [0.65, 1.0, 0.35, 0.04]
        elif rc == "medium":
            vals = [0.35, 0.78, 1.0, 0.25]
        else:
            vals = [0.18, 0.58, 0.92, 1.0]
        for j, v in enumerate(vals):
            ax.add_patch(Rectangle((x - 0.045 + j * 0.028, y0), 0.02, 0.085 * v, facecolor=COL[rc], edgecolor="none", alpha=0.85))
        ax.add_line(Line2D([x - 0.05, x + 0.055], [y0, y0], lw=0.45, color=COL["muted"]))
    ax.text(0.43, 0.005, "local  ->  broader spatial context", fontsize=5.8, ha="center", color=COL["muted"])


def draw_panel_b(ax, truth_design: pd.DataFrame) -> None:
    add_panel_label(ax, "b", x=-0.1, y=1.02)
    ax.set_xlim(0, 0.62)
    ax.set_ylim(-0.1, 4.2)
    ax.set_yticks([])
    ax.set_xlabel("sender-receiver distance", fontsize=6.8, labelpad=1.5)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    clean_axis(ax, grid_y=False)
    ax.spines["left"].set_visible(False)
    ax.text(0.0, 4.02, "Mechanism-aware benchmark design", fontsize=8.0, fontweight="bold", ha="left")

    for i, rc in enumerate(RANGE_ORDER):
        row = truth_design[truth_design["range_class"].eq(rc)].iloc[0]
        y = 3.35 - i * 0.85
        ax.hlines(y, 0.0, 0.60, color=COL["grid"], lw=1.2, zorder=0)
        ax.add_patch(
            FancyBboxPatch(
                (row["valid_range_start"], y - 0.09),
                row["valid_range_end"] - row["valid_range_start"],
                0.18,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                linewidth=0,
                facecolor=COL[rc],
                alpha=0.86,
            )
        )
        ax.text(-0.012, y, row["range_label"], ha="right", va="center", fontsize=6.7, color=COL[rc], fontweight="bold")
        # Decoy marks.
        if rc == "contact":
            xs = [0.30, 0.47]
        elif rc == "short":
            xs = [0.42, 0.53]
        elif rc == "medium":
            xs = [0.05, 0.53]
        else:
            xs = [0.06, 0.17, 0.43]
        for x in xs:
            ax.plot(x, y, marker="x", color=COL["decoy"], ms=4.5, mew=0.9, zorder=3)
        if rc == "long_niche":
            ax.text(0.47, y + 0.18, "niche rule", ha="center", va="center", fontsize=5.4, color=COL["muted"])
    ax.text(0.115, 0.13, "true range", fontsize=5.8, ha="center", color=COL["muted"])
    ax.plot(0.29, 0.13, marker="x", color=COL["decoy"], ms=4.3, mew=0.9)
    ax.text(0.34, 0.13, "decoy", fontsize=5.8, va="center", color=COL["muted"])


def draw_panel_c(fig, outer_spec, profile: pd.DataFrame) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_spec, wspace=0.28)
    axes = [fig.add_subplot(inner[0, i]) for i in range(4)]
    for i, (ax, rc) in enumerate(zip(axes, RANGE_ORDER, strict=False)):
        sub = profile[profile["range_class"].eq(rc)]
        x = sub["distance_bin_index"].to_numpy()
        ax.plot(
            x,
            sub["truth_enrichment"],
            color=COL[rc],
            lw=1.05,
            linestyle=(0, (2, 1.5)),
            alpha=0.65,
        )
        ax.plot(
            x,
            sub["predicted_activity"],
            color=COL[rc],
            lw=1.65,
            marker="o",
            ms=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
        ax.fill_between(x, 0, sub["truth_enrichment"], color=COL[rc], alpha=0.10, linewidth=0)
        ax.set_title(RANGE_LABEL[rc], fontsize=7.2, color=COL[rc], pad=2.0, fontweight="bold")
        ax.set_xlim(0.75, 6.25)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xticks([1, 3, 6])
        if i == 0:
            ax.set_ylabel("scaled activity", fontsize=6.8, labelpad=2)
            add_panel_label(ax, "c", x=-0.28, y=1.19)
            ax.text(-0.28, 1.13, "Predicted activity follows LR range", transform=ax.transAxes, fontsize=8.0, fontweight="bold", ha="left")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("distance bin", fontsize=6.5, labelpad=1.5)
        clean_axis(ax, grid_y=True)
    axes[-1].legend(
        [
            Line2D([0], [0], color=COL["muted"], lw=1.0, linestyle=(0, (2, 1.5))),
            Line2D([0], [0], color=COL["muted"], lw=1.6, marker="o", ms=3.0),
        ],
        ["truth", "SpatialSelf"],
        loc="upper right",
        bbox_to_anchor=(1.05, 1.04),
        fontsize=5.8,
        handlelength=1.8,
        borderpad=0.1,
    )


def draw_panel_d(ax, comp: pd.DataFrame) -> None:
    add_panel_label(ax, "d", x=-0.08, y=1.03)
    components = ["True", "Wrong range", "Wrong niche", "High-expression decoy", "Other false"]
    colors = {
        "True": COL["ours"],
        "Wrong range": COL["sand"],
        "Wrong niche": COL["decoy"],
        "High-expression decoy": "#C9A3B8",
        "Other false": "#D6D1C9",
    }
    pivot = comp.pivot(index="method_label", columns="component", values="fraction").reindex(METHOD_ORDER).fillna(0)
    y = np.arange(len(METHOD_ORDER))
    left = np.zeros(len(METHOD_ORDER))
    for comp_name in components:
        vals = pivot[comp_name].to_numpy() if comp_name in pivot.columns else np.zeros(len(METHOD_ORDER))
        ax.barh(y, vals, left=left, height=0.55, color=colors[comp_name], edgecolor="white", linewidth=0.35, label=comp_name)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction among top 5% calls", fontsize=6.8)
    ax.set_title("Top-call composition", fontsize=8.0, fontweight="bold", pad=2.5)
    clean_axis(ax, grid_y=False)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, -0.35),
        ncol=3,
        fontsize=5.5,
        handlelength=1.2,
        columnspacing=0.7,
    )


def draw_panel_e(ax, heat: pd.DataFrame) -> None:
    add_panel_label(ax, "e", x=-0.07, y=1.03)
    mat = heat.pivot(index="method_label", columns="range_class", values="aupr").reindex(METHOD_ORDER)[RANGE_ORDER]
    cmap = LinearSegmentedColormap.from_list("soft_mechanism", ["#F7F4EC", "#E6C06A", "#2F78A8"])
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0.0, vmax=0.9)
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.1)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([])
    ax.set_title("Range-class AUPR", fontsize=8.0, fontweight="bold", pad=2.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Very thin separators keep cells readable without the heavy white grid look.
    ax.set_xticks(np.arange(-0.5, len(RANGE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if np.isfinite(val) and (i == 0 or val > 0.75):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.2, color="white" if val > 0.62 else COL["ink"])
    # Keep this compact; exact values are in the source table.


def draw_panel_f(ax, decoy: pd.DataFrame) -> None:
    add_panel_label(ax, "f", x=-0.08, y=1.03)
    dat = decoy.copy()
    dat["method_label"] = dat["method_label"].astype(str)
    dat = dat[dat["method_label"].isin(METHOD_ORDER)].copy()
    dat = dat.sort_values("wrong_range_rate_top5pct", ascending=True)
    y = np.arange(len(dat))
    colors = [COL["ours"] if m == "SpatialSelf" else COL["baseline_light"] for m in dat["method_label"]]
    edge = [COL["ours_dark"] if m == "SpatialSelf" else COL["baseline"] for m in dat["method_label"]]
    ax.scatter(dat["wrong_range_rate_top5pct"], y, s=34, c=colors, edgecolors=edge, linewidths=0.7, zorder=3)
    ax.scatter(
        dat["too_close_long_range_rate_top5pct"],
        y + 0.18,
        s=20,
        marker="D",
        c=[COL["blue"] if m == "SpatialSelf" else "#D9CABB" for m in dat["method_label"]],
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(dat["method_label"], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(-0.015, max(0.75, float(dat["too_close_long_range_rate_top5pct"].max()) + 0.05))
    ax.set_xlabel("decoy rate in top 5%", fontsize=6.8)
    ax.set_title("Wrong-range suppression", fontsize=8.0, fontweight="bold", pad=2.5)
    clean_axis(ax, grid_y=True)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.legend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["baseline_light"], markeredgecolor=COL["baseline"], markersize=4.5),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#D9CABB", markeredgecolor="white", markersize=3.8),
        ],
        ["wrong range", "too-close long"],
        loc="lower right",
        fontsize=5.7,
        handletextpad=0.3,
        borderpad=0.1,
    )


def draw_panel_g(ax, align: pd.DataFrame) -> None:
    add_panel_label(ax, "g", x=-0.08, y=1.03)
    dat = align[align["method_label"].isin(["SpatialSelf", "COMMOT", "SpaOTsc"])].copy()
    for method, marker, alpha, z in [("COMMOT", "o", 0.35, 2), ("SpaOTsc", "s", 0.32, 1), ("SpatialSelf", "o", 0.98, 5)]:
        sub = dat[dat["method_label"].eq(method)]
        for rc in RANGE_ORDER:
            ss = sub[sub["range_class"].eq(rc)]
            if ss.empty:
                continue
            ax.scatter(
                ss["true_positive_median_distance"],
                ss["predicted_top_median_distance"],
                s=42 if method == "SpatialSelf" else 25,
                marker=marker,
                color=COL[rc] if method == "SpatialSelf" else "#C7C0B6",
                edgecolor="white",
                linewidth=0.55,
                alpha=alpha,
                zorder=z,
            )
    lim = [0.0, 0.45]
    ax.plot(lim, lim, color=COL["muted"], lw=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true median distance", fontsize=6.8)
    ax.set_ylabel("predicted top median", fontsize=6.8)
    ax.set_title("Effective range alignment", fontsize=8.0, fontweight="bold", pad=2.5)
    clean_axis(ax, grid_y=True)
    ax.text(0.015, 0.43, "colors as c", fontsize=5.5, color=COL["muted"], ha="left", va="top")


def draw_panel_h(ax, abl: pd.DataFrame) -> None:
    add_panel_label(ax, "h", x=-0.08, y=1.03)
    dat = abl.sort_values("selection_score", ascending=True).copy()
    y = np.arange(len(dat))
    vals = dat["selection_score"].to_numpy()
    colors = [COL["ours"] if r == "range_context_guard" else "#D4C8BA" for r in dat["run_id"]]
    ax.barh(y, vals, height=0.55, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(dat["variant_label"], fontsize=5.9)
    ax.set_xlim(0, max(vals) * 1.08)
    ax.set_xlabel("balanced mechanism score", fontsize=6.8)
    ax.set_title("Scale/context ablation", fontsize=8.0, fontweight="bold", pad=2.5)
    clean_axis(ax, grid_y=False)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    best = dat.iloc[-1]
    ax.text(best["selection_score"] + 0.012, len(dat) - 1, "final", fontsize=5.8, va="center", color=COL["ours_dark"], fontweight="bold")


def draw_figure(tables: dict[str, pd.DataFrame]) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig = plt.figure(figsize=(7.35, 8.2), constrained_layout=False)
    outer = gridspec.GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.35, 1.18, 1.55, 1.45],
        hspace=0.62,
        wspace=0.80,
        left=0.065,
        right=0.985,
        top=0.975,
        bottom=0.065,
    )
    ax_a = fig.add_subplot(outer[0, :3])
    ax_b = fig.add_subplot(outer[0, 3:])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b, tables["panel_b_truth_design"])
    draw_panel_c(fig, outer[1, :], tables["panel_c_score_distance_profiles"])
    ax_d = fig.add_subplot(outer[2, :2])
    ax_e = fig.add_subplot(outer[2, 2:4])
    ax_f = fig.add_subplot(outer[2, 4:])
    draw_panel_d(ax_d, tables["panel_d_top_call_composition"])
    draw_panel_e(ax_e, tables["panel_e_mechanism_aupr"])
    draw_panel_f(ax_f, tables["panel_f_wrong_range_suppression"])
    ax_g = fig.add_subplot(outer[3, :3])
    ax_h = fig.add_subplot(outer[3, 3:])
    draw_panel_g(ax_g, tables["panel_g_effective_range_alignment"])
    draw_panel_h(ax_h, tables["panel_h_scale_context_ablation"])

    stem = FIGDIR / "fig5_lr_scale_mechanism"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = write_source_data()
    stem = draw_figure(tables)
    print(f"Wrote {stem}.svg/.pdf/.png/.tiff")
    print(f"Source data: {SOURCE}")


if __name__ == "__main__":
    main()
