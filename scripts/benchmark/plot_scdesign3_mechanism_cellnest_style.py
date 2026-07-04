#!/usr/bin/env python
"""CellNEST Fig. 3f-j inspired mechanism benchmark figure for SpatialSelf."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs" / "reference_based_benchmark" / "full_scdesign3_strongest"
DETAIL = BENCH / "final_tables" / "scdesign3_strongest_comparison_detail.csv"
OUT = BENCH / "figures"
SOURCE = BENCH / "source_data"

METHOD_ORDER = ["SpatialSelf", "COMMOT", "CellChat", "DeepTalk", "SpaTalk", "CellNEST"]
DATASET_ORDER = ["lymph_scdesign3", "melanoma_scdesign3"]
DATASET_LABEL = {"lymph_scdesign3": "Lymph node", "melanoma_scdesign3": "Melanoma"}
METHOD_LABEL_TO_DIR = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "CellChat": "cellchat",
    "DeepTalk": "deeptalk",
    "SpaTalk": "spatalk",
    "CellNEST": "cellnest",
}
DATASET_DIR = {"lymph_scdesign3": "lymph", "melanoma_scdesign3": "melanoma"}

MECHANISMS = [
    ("juxtacrine_contact", "Contact"),
    ("short_range_paracrine", "Short"),
    ("medium_range_paracrine", "Medium"),
    ("long_range_chemokine_niche", "Chemokine\nniche"),
]

PALETTE = {
    "ink": "#20252B",
    "muted": "#6E7781",
    "line": "#C9D0D8",
    "grid": "#E8EBEF",
    "blue": "#2E74B5",
    "light_blue": "#DDEAF6",
    "sender": "#2E74B5",
    "receiver": "#58A6D6",
    "negative": "#B95E5A",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7.4,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(DETAIL)
    records: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_dir = DATASET_DIR[dataset]
        for method_label in METHOD_ORDER:
            method_dir = METHOD_LABEL_TO_DIR[method_label]
            metric_files = sorted((BENCH / dataset_dir / method_dir).glob("*_metrics.json"))
            row = detail[(detail["dataset"].eq(dataset)) & (detail["method_label"].eq(method_label))]
            status = str(row.iloc[0]["status"]) if len(row) else "missing"
            if not metric_files:
                for mech_key, mech_label in MECHANISMS:
                    records.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASET_LABEL[dataset],
                            "method_label": method_label,
                            "mechanism": mech_key,
                            "mechanism_label": mech_label,
                            "aupr": np.nan,
                            "positive_count": np.nan,
                            "status": status,
                        }
                    )
                continue
            with metric_files[0].open() as fh:
                metrics = json.load(fh)
            for mech_key, mech_label in MECHANISMS:
                records.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABEL[dataset],
                        "method_label": method_label,
                        "mechanism": mech_key,
                        "mechanism_label": mech_label,
                        "aupr": metrics.get(f"aupr__{mech_key}", np.nan),
                        "positive_count": metrics.get(f"positive_count__{mech_key}", np.nan),
                        "status": status,
                    }
                )
    mech = pd.DataFrame(records)
    return detail, mech


def save_outputs(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def generate_pattern(pattern: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_bg = 210
    bg = rng.uniform(0.05, 0.95, size=(n_bg, 2))
    bg_type = np.repeat("background", n_bg)

    if pattern == "contact":
        sender = rng.normal([0.45, 0.52], [0.075, 0.075], size=(36, 2))
        receiver = sender + rng.normal([0.035, 0.0], [0.018, 0.018], size=(36, 2))
        radius = 0.08
        caption = "direct adjacency"
    elif pattern == "paracrine":
        theta = rng.uniform(0, 2 * np.pi, size=46)
        sender = rng.normal([0.36, 0.55], [0.055, 0.055], size=(34, 2))
        receiver = np.column_stack([0.57 + 0.13 * np.cos(theta), 0.55 + 0.13 * np.sin(theta)])
        receiver += rng.normal(0, 0.025, size=receiver.shape)
        radius = 0.22
        caption = "local diffusion"
    elif pattern == "chemokine":
        sender = rng.normal([0.28, 0.35], [0.045, 0.045], size=(30, 2))
        receiver = rng.normal([0.74, 0.72], [0.06, 0.06], size=(44, 2))
        niche = rng.normal([0.72, 0.70], [0.11, 0.09], size=(80, 2))
        bg = np.vstack([bg, niche])
        bg_type = np.concatenate([bg_type, np.repeat("niche", len(niche))])
        radius = 0.30
        caption = "range plus niche"
    else:
        raise ValueError(pattern)

    df = pd.DataFrame(bg, columns=["x", "y"])
    df["kind"] = bg_type
    df = pd.concat(
        [
            df,
            pd.DataFrame({"x": sender[:, 0], "y": sender[:, 1], "kind": "sender"}),
            pd.DataFrame({"x": receiver[:, 0], "y": receiver[:, 1], "kind": "receiver"}),
        ],
        ignore_index=True,
    )
    df[["x", "y"]] = df[["x", "y"]].clip(0.02, 0.98)
    df["radius"] = radius
    df["caption"] = caption
    df["pattern"] = pattern
    return df


def draw_pattern(ax: plt.Axes, df: pd.DataFrame, title: str, subtitle: str, label: str) -> None:
    panel_label(ax, label, x=-0.095, y=1.085)
    ax.set_title(title, loc="center", pad=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D7DDE3")
        spine.set_linewidth(0.6)

    if "niche" in set(df["kind"]):
        niche = df[df["kind"].eq("niche")]
        ax.scatter(niche["x"], niche["y"], s=9, color=PALETTE["light_blue"], edgecolors="none", alpha=0.55, zorder=1)

    bg = df[df["kind"].eq("background")]
    sender = df[df["kind"].eq("sender")]
    receiver = df[df["kind"].eq("receiver")]
    ax.scatter(bg["x"], bg["y"], s=7, color="#D8DEE5", edgecolors="none", alpha=0.55, zorder=1)
    ax.scatter(receiver["x"], receiver["y"], s=14, color=PALETTE["receiver"], edgecolors="white", linewidths=0.25, zorder=3)
    ax.scatter(sender["x"], sender["y"], s=17, color=PALETTE["sender"], edgecolors="white", linewidths=0.25, zorder=4)

    center = {
        "contact": (0.47, 0.52),
        "paracrine": (0.47, 0.55),
        "chemokine": (0.58, 0.54),
    }[str(df["pattern"].iloc[0])]
    ax.add_patch(Circle(center, float(df["radius"].iloc[0]), fill=False, edgecolor=PALETTE["blue"], linewidth=0.9, alpha=0.65))
    ax.text(0.02, -0.09, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=PALETTE["muted"])


def draw_heatmap(ax: plt.Axes, mech: pd.DataFrame, dataset: str, label: str, *, show_ylabels: bool = True) -> mpl.image.AxesImage:
    panel_label(ax, label, x=-0.075, y=1.055)
    sub = mech[mech["dataset"].eq(dataset)].copy()
    matrix = (
        sub.pivot(index="method_label", columns="mechanism_label", values="aupr")
        .reindex(index=METHOD_ORDER, columns=[x[1] for x in MECHANISMS])
    )
    arr = matrix.to_numpy(float)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(color="#EFEFF1")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
    ax.set_title(f"{DATASET_LABEL[dataset]}: mechanism-specific AUPR", loc="left", pad=7)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index if show_ylabels else [])
    ax.tick_params(length=0)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                ax.text(j, i, "X", ha="center", va="center", fontsize=8.5, color=PALETTE["negative"], fontweight="bold")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.8, color="white" if val > 0.42 else PALETTE["ink"])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D7DDE3")
        spine.set_linewidth(0.6)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def export_source_data(detail: pd.DataFrame, mech: pd.DataFrame) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    mech.to_csv(SOURCE / "scdesign3_mechanism_cellnest_style_aupr.csv", index=False)
    detail.to_csv(SOURCE / "scdesign3_mechanism_cellnest_style_status.csv", index=False)
    patterns = pd.concat(
        [
            generate_pattern("contact", seed=11),
            generate_pattern("paracrine", seed=12),
            generate_pattern("chemokine", seed=13),
        ],
        ignore_index=True,
    )
    patterns.to_csv(SOURCE / "scdesign3_mechanism_cellnest_style_schematic_points.csv", index=False)
    pd.DataFrame(
        [
            ("f", "contact-dependent LR benchmark pattern", "scdesign3_mechanism_cellnest_style_schematic_points.csv"),
            ("g", "short/medium paracrine benchmark pattern", "scdesign3_mechanism_cellnest_style_schematic_points.csv"),
            ("h", "chemokine-like long-range niche benchmark pattern", "scdesign3_mechanism_cellnest_style_schematic_points.csv"),
            ("i", "lymph node mechanism-specific AUPR", "scdesign3_mechanism_cellnest_style_aupr.csv"),
            ("j", "melanoma mechanism-specific AUPR", "scdesign3_mechanism_cellnest_style_aupr.csv"),
        ],
        columns=["panel", "content", "source_data"],
    ).to_csv(SOURCE / "scdesign3_mechanism_cellnest_style_manifest.csv", index=False)


def main() -> None:
    apply_style()
    detail, mech = load_metrics()
    export_source_data(detail, mech)

    fig = plt.figure(figsize=(7.35, 4.65), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 1.22],
        left=0.065,
        right=0.965,
        bottom=0.12,
        top=0.94,
        wspace=0.72,
        hspace=0.62,
    )
    ax_f = fig.add_subplot(gs[0, 0:2])
    ax_g = fig.add_subplot(gs[0, 2:4])
    ax_h = fig.add_subplot(gs[0, 4:6])
    ax_i = fig.add_subplot(gs[1, 0:3])
    ax_j = fig.add_subplot(gs[1, 3:6])

    draw_pattern(ax_f, generate_pattern("contact", seed=11), "Contact LR", "adjacent sender-receiver edges", "f")
    draw_pattern(ax_g, generate_pattern("paracrine", seed=12), "Paracrine LR", "finite-radius diffusion with nearby decoys", "g")
    draw_pattern(ax_h, generate_pattern("chemokine", seed=13), "Niche-restricted LR", "distant signaling only in the correct niche", "h")
    im = draw_heatmap(ax_i, mech, "lymph_scdesign3", "i")
    draw_heatmap(ax_j, mech, "melanoma_scdesign3", "j", show_ylabels=False)

    cbar = fig.colorbar(im, ax=[ax_i, ax_j], fraction=0.025, pad=0.025)
    cbar.set_label("AUPR", rotation=270, labelpad=10)
    cbar.outline.set_visible(False)

    legend_ax = fig.add_axes([0.065, 0.025, 0.45, 0.055])
    legend_ax.set_axis_off()
    handles = [
        legend_ax.scatter([], [], s=22, color=PALETTE["sender"], edgecolors="white", linewidths=0.3, label="sender"),
        legend_ax.scatter([], [], s=20, color=PALETTE["receiver"], edgecolors="white", linewidths=0.3, label="receiver"),
        legend_ax.scatter([], [], s=16, color="#D8DEE5", edgecolors="none", label="background"),
    ]
    legend_ax.legend(handles=handles, loc="center left", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.1)

    stem = OUT / "scdesign3_mechanism_benchmark_cellnest_style"
    save_outputs(fig, stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
