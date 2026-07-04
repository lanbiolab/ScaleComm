#!/usr/bin/env python
"""Plot dedicated SPP1-CD44 negative-control evidence panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"
RESULT_DIR = PHASE11_DIR / "results"
FIG_DIR = PHASE11_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

TEXT = "#202020"
GRID = "#D8D8D8"
OBS = "#2F6B4F"
CONTROL = "#B8B8B8"
WARN = "#C85A5A"
BLUE = "#456C9B"

ORDER = [
    "observed",
    "shuffled_receiver",
    "shuffled_lr_identity",
    "wrong_niche_masked",
    "wrong_receiver_masked",
    "wrong_type_masked",
]
LABELS = {
    "observed": "Observed",
    "shuffled_receiver": "Shuffled\nreceiver",
    "shuffled_lr_identity": "Shuffled\nLR identity",
    "wrong_niche_masked": "Wrong\nniche",
    "wrong_receiver_masked": "Wrong\nreceiver",
    "wrong_type_masked": "Wrong\ntype",
}


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    empirical_path = RESULT_DIR / "spp1_cd44_negative_control_empirical_fdr_summary.csv"
    controls = pd.read_csv(empirical_path if empirical_path.exists() else RESULT_DIR / "spp1_cd44_case_control_summary.csv")
    panel = build_panel_table(controls)
    panel.to_csv(RESULT_DIR / "spp1_cd44_negative_control_panel_summary.csv", index=False)
    panel.to_csv(SOURCE_DIR / "spp1_cd44_negative_control_panel_summary.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.08,
        right=0.985,
        top=0.84,
        bottom=0.18,
        wspace=0.42,
        hspace=0.58,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    plot_overlap(axes[0], panel)
    plot_rank(axes[1], panel)
    plot_delta_r2(axes[2], panel)
    plot_fdr(axes[3], panel)
    plot_observed_minus_control(axes[4], panel)
    plot_status_matrix(axes[5], panel)

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], axes):
        add_panel_label(ax, label)

    fig.suptitle(
        "SPP1-CD44 negative controls distinguish true context from high-expression artifacts",
        x=0.08,
        y=0.965,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.08,
        0.905,
        "Shuffled receiver, wrong-niche and wrong-receiver controls reduce target overlap, rank and impact; wrong-type remains competitive and is flagged.",
        fontsize=6.3,
        color="#555555",
    )
    fig.text(
        0.08,
        0.055,
        "Empirical FDR is calibrated with receiver-shuffled SPP1-CD44 incoming scores when available. Delta R2 is receiver target-gene impact from the Phase 10 control analysis.",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, FIG_DIR / "phase11_spp1_cd44_negative_control_panel")
    plt.close(fig)
    print(panel.to_string(index=False))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def build_panel_table(controls: pd.DataFrame) -> pd.DataFrame:
    panel = controls.set_index("control_type").reindex(ORDER).reset_index()
    panel["control_label"] = panel["control_type"].map(LABELS)
    if "empirical_fdr_combined" not in panel.columns:
        panel["target_enrichment_bh_fdr"] = bh_adjust(panel["target_enrichment_p"].to_numpy(dtype=float))
        panel["empirical_fdr_combined"] = np.nan
        panel["empirical_fdr_delta_r2"] = np.nan
        panel["empirical_fdr_overlap"] = np.nan
        panel["empirical_fdr_mean_rank"] = np.nan
    elif "target_enrichment_bh_fdr" not in panel.columns:
        panel["target_enrichment_bh_fdr"] = bh_adjust(panel["target_enrichment_p"].to_numpy(dtype=float))
    observed = panel[panel["control_type"].eq("observed")].iloc[0]
    panel["observed_minus_target_overlap"] = observed["target_overlap_top_n"] - panel["target_overlap_top_n"]
    panel["observed_minus_mean_target_rank"] = panel["mean_target_rank"] - observed["mean_target_rank"]
    panel["observed_minus_delta_r2"] = observed["delta_cv_r2"] - panel["delta_cv_r2"]
    panel["control_group"] = np.where(panel["control_type"].eq("observed"), "observed", "control")
    panel.loc[panel["control_status"].eq("control_remains_competitive"), "control_group"] = "competitive_control"
    return panel


def colors(panel: pd.DataFrame) -> list[str]:
    return [OBS if x == "observed" else WARN if x == "competitive_control" else CONTROL for x in panel["control_group"]]


def plot_overlap(ax: plt.Axes, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    ax.bar(x, panel["target_overlap_top_n"], color=colors(panel), edgecolor="none")
    ax.set_title("Target overlap", loc="left", pad=4)
    ax.set_ylabel("top-target overlap")
    set_xlabels(ax, panel)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_rank(ax: plt.Axes, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    ax.bar(x, panel["mean_target_rank"], color=colors(panel), edgecolor="none")
    ax.set_title("Mean target rank", loc="left", pad=4)
    ax.set_ylabel("lower is better")
    set_xlabels(ax, panel)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_delta_r2(ax: plt.Axes, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    ax.bar(x, panel["delta_cv_r2"], color=colors(panel), edgecolor="none")
    ax.set_title("Target-gene impact", loc="left", pad=4)
    ax.set_ylabel("delta R2")
    set_xlabels(ax, panel)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_fdr(ax: plt.Axes, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    fdr_column = "empirical_fdr_combined" if panel["empirical_fdr_combined"].notna().any() else "target_enrichment_bh_fdr"
    vals = -np.log10(np.clip(panel[fdr_column].fillna(1), 1e-300, 1))
    ax.bar(x, vals, color=colors(panel), edgecolor="none")
    ax.set_title("Empirical FDR", loc="left", pad=4)
    ax.set_ylabel("-log10 FDR")
    set_xlabels(ax, panel)
    ax.grid(axis="y", color=GRID, linewidth=0.45)


def plot_observed_minus_control(ax: plt.Axes, panel: pd.DataFrame) -> None:
    plot = panel[~panel["control_type"].eq("observed")].copy()
    x = np.arange(len(plot))
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.bar(x - 0.18, plot["observed_minus_target_overlap"], width=0.34, color=OBS, edgecolor="none", label="overlap gain")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, plot["observed_minus_delta_r2"], width=0.34, color=BLUE, edgecolor="none", label="delta R2 gain")
    ax.set_title("Observed minus control", loc="left", pad=4)
    ax.set_ylabel("target overlap gain")
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", labelsize=5.3, pad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["control_label"], fontsize=5.3)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax2.spines["top"].set_visible(False)
    ax.text(0.98, 0.94, "blue: delta R2 gain", transform=ax.transAxes, ha="right", va="top", fontsize=5.1, color=BLUE)


def plot_status_matrix(ax: plt.Axes, panel: pd.DataFrame) -> None:
    fdr_column = "empirical_fdr_combined" if panel["empirical_fdr_combined"].notna().any() else "target_enrichment_bh_fdr"
    metrics = ["target_overlap_top_n", "mean_target_rank", "delta_cv_r2", fdr_column]
    labels = ["overlap", "rank", "impact", "FDR"]
    values = panel[metrics].copy()
    values["mean_target_rank"] = -values["mean_target_rank"]
    values[fdr_column] = -np.log10(np.clip(panel[fdr_column].fillna(1), 1e-300, 1))
    scaled = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0)).replace(0, np.nan)
    arr = scaled.fillna(0).to_numpy(dtype=float).T
    im = ax.imshow(arr, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Specificity profile", loc="left", pad=4)
    ax.set_xticks(np.arange(len(panel)))
    ax.set_xticklabels(panel["control_label"], fontsize=5.2)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=5.1, color="#202020")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("row-scaled support", fontsize=5.2)
    cb.ax.tick_params(labelsize=5, length=2)


def set_xlabels(ax: plt.Axes, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    ax.set_xticks(x)
    ax.set_xticklabels(panel["control_label"], rotation=25, ha="right", fontsize=5.5)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=TEXT)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    finite_p = pvals[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out[idx[order]] = adjusted
    return out


if __name__ == "__main__":
    main()
