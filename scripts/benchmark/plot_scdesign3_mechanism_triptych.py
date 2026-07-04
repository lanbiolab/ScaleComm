#!/usr/bin/env python
"""Triptych mechanism benchmark figure: mechanism panels above matched results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs" / "reference_based_benchmark" / "full_scdesign3_strongest"
OUT = BENCH / "figures"
SOURCE = BENCH / "source_data"

METHODS = ["SpatialSelf", "COMMOT", "CellChat", "DeepTalk", "SpaTalk", "CellNEST"]
METHOD_DIRS = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "CellChat": "cellchat",
    "DeepTalk": "deeptalk",
    "SpaTalk": "spatalk",
    "CellNEST": "cellnest",
}
DATASET_DIRS = {"lymph_scdesign3": "lymph", "melanoma_scdesign3": "melanoma"}
MECH = {
    "Contact": "juxtacrine_contact",
    "Short": "short_range_paracrine",
    "Medium": "medium_range_paracrine",
    "Niche": "long_range_chemokine_niche",
}

COL = {
    "ink": "#22272E",
    "muted": "#69717A",
    "axis": "#CBD3DB",
    "grid": "#EEF1F4",
    "bg": "#DDE3EA",
    "bg_soft": "#EEF2F5",
    "sender": "#0F7F7D",
    "receiver": "#4EA3D8",
    "decoy": "#C69A48",
    "niche": "#E0F0F1",
    "spatialself": "#0F7F7D",
    "commot": "#4F5964",
    "cellchat": "#AEB8C2",
    "deeptalk": "#C8D0D8",
    "spatalk": "#C3A256",
    "cellnest": "#B5A5C9",
}

METHOD_COLORS = {
    "SpatialSelf": COL["spatialself"],
    "COMMOT": COL["commot"],
    "CellChat": COL["cellchat"],
    "DeepTalk": COL["deeptalk"],
    "SpaTalk": COL["spatalk"],
    "CellNEST": COL["cellnest"],
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 6.7,
            "axes.titlesize": 6.9,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.9,
        fontweight="bold",
        color=COL["ink"],
    )


def prep_schematic_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#DEE4EA")
        spine.set_linewidth(0.45)


def draw_cell(ax: plt.Axes, xy: tuple[float, float], color: str, *, s: float = 28, z: int = 3, alpha: float = 1.0) -> None:
    ax.scatter([xy[0]], [xy[1]], s=s, color=color, edgecolors="white", linewidths=0.35, zorder=z, alpha=alpha)


def draw_background(ax: plt.Axes, seed: int, *, n: int = 170) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.04, 0.96, size=(n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=COL["bg"], edgecolors="none", alpha=0.46, zorder=1)


def draw_group(ax: plt.Axes, center: tuple[float, float], n: int, spread: tuple[float, float], color: str, seed: int, *, size: int = 23, z: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(center, spread, size=(n, 2))
    pts = np.clip(pts, 0.04, 0.96)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, color=color, edgecolors="white", linewidths=0.28, zorder=z)
    return pts


def draw_contact(ax: plt.Axes) -> None:
    panel_label(ax, "a", x=-0.08, y=1.03)
    ax.set_title("Contact LR", pad=5)
    prep_schematic_axis(ax)
    draw_background(ax, 31, n=145)
    rng = np.random.default_rng(32)
    senders = rng.normal([0.46, 0.52], [0.070, 0.065], size=(34, 2))
    receivers = senders + rng.normal([0.038, 0.002], [0.016, 0.015], size=(34, 2))
    senders = np.clip(senders, 0.08, 0.92)
    receivers = np.clip(receivers, 0.08, 0.92)
    for spt, rpt in zip(senders[::2], receivers[::2], strict=False):
        ax.plot([spt[0], rpt[0]], [spt[1], rpt[1]], color="#9FB4C7", linewidth=0.52, alpha=0.72, zorder=2)
    ax.scatter(receivers[:, 0], receivers[:, 1], s=21, color=COL["receiver"], edgecolors="white", linewidths=0.28, zorder=4)
    ax.scatter(senders[:, 0], senders[:, 1], s=23, color=COL["sender"], edgecolors="white", linewidths=0.28, zorder=5)


def draw_paracrine(ax: plt.Axes) -> None:
    panel_label(ax, "b", x=-0.08, y=1.03)
    ax.set_title("Paracrine LR", pad=5)
    prep_schematic_axis(ax)
    draw_background(ax, 41, n=160)
    ax.add_patch(Circle((0.44, 0.54), 0.245, fill=False, edgecolor=COL["sender"], linewidth=0.95, alpha=0.72, zorder=2))
    draw_group(ax, (0.35, 0.53), 28, (0.045, 0.052), COL["sender"], 42, size=23, z=5)
    theta = np.linspace(-0.95, 1.10, 32)
    rec = np.column_stack([0.55 + 0.12 * np.cos(theta), 0.54 + 0.17 * np.sin(theta)])
    rec += np.random.default_rng(43).normal(0, 0.018, size=rec.shape)
    ax.scatter(rec[:, 0], rec[:, 1], s=21, color=COL["receiver"], edgecolors="white", linewidths=0.28, zorder=4)
    for xy in [(0.78, 0.32), (0.80, 0.76), (0.73, 0.52)]:
        draw_cell(ax, xy, COL["decoy"], s=18, z=3, alpha=0.88)


def draw_niche(ax: plt.Axes) -> None:
    panel_label(ax, "c", x=-0.08, y=1.03)
    ax.set_title("Niche-restricted LR", pad=5)
    prep_schematic_axis(ax)
    draw_background(ax, 51, n=160)
    ax.add_patch(Ellipse((0.72, 0.70), 0.38, 0.28, angle=12, facecolor=COL["niche"], edgecolor="none", alpha=0.95, zorder=1))
    ax.add_patch(Circle((0.58, 0.55), 0.325, fill=False, edgecolor=COL["sender"], linewidth=0.95, alpha=0.70, zorder=2))
    draw_group(ax, (0.28, 0.34), 26, (0.043, 0.045), COL["sender"], 52, size=23, z=5)
    draw_group(ax, (0.72, 0.70), 36, (0.057, 0.057), COL["receiver"], 53, size=21, z=4)
    for xy in [(0.80, 0.24), (0.26, 0.74)]:
        draw_cell(ax, xy, COL["decoy"], s=18, z=3, alpha=0.86)
    ax.add_patch(
        FancyArrowPatch(
            (0.34, 0.40),
            (0.65, 0.64),
            connectionstyle="arc3,rad=-0.16",
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.70,
            color=COL["sender"],
            alpha=0.72,
            zorder=3,
        )
    )


def load_mechanism_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, dataset_dir in DATASET_DIRS.items():
        for method, method_dir in METHOD_DIRS.items():
            files = sorted((BENCH / dataset_dir / method_dir).glob("*_metrics.json"))
            if not files:
                continue
            with files[0].open() as fh:
                metrics = json.load(fh)
            for label, key in MECH.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "mechanism": label,
                        "aupr": float(metrics.get(f"aupr__{key}", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def mean_metric(metrics: pd.DataFrame, mechanisms: list[str]) -> pd.DataFrame:
    sub = metrics[metrics["mechanism"].isin(mechanisms)].copy()
    return sub.groupby(["method"], as_index=False)["aupr"].mean()


def clean_result_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 0.90)
    ax.set_xlabel("AUPR")
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COL["axis"])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=COL["muted"], length=2.2, width=0.5)


def draw_lollipop(ax: plt.Axes, metrics: pd.DataFrame, mechanisms: list[str], title: str, label: str) -> None:
    panel_label(ax, label, x=-0.08, y=1.04)
    ax.set_title(title, loc="left", pad=6)
    vals = mean_metric(metrics, mechanisms).set_index("method")["aupr"].reindex(METHODS)
    y = np.arange(len(METHODS))[::-1]
    for yi, method in zip(y, METHODS, strict=True):
        val = float(vals.loc[method])
        if np.isnan(val):
            ax.text(0.03, yi, "X", ha="left", va="center", fontsize=7.0, color="#B85C5A", fontweight="bold")
            continue
        ax.plot([0, val], [yi, yi], color=METHOD_COLORS[method], linewidth=1.05 if method == "SpatialSelf" else 0.70, alpha=0.94)
        ax.scatter([val], [yi], s=34 if method == "SpatialSelf" else 25, color=METHOD_COLORS[method], edgecolors="white", linewidths=0.45, zorder=3)
        ax.text(val + 0.018, yi, f"{val:.2f}", ha="left", va="center", fontsize=5.0, color=COL["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels(METHODS)
    ax.set_ylim(-0.65, len(METHODS) - 0.35)
    clean_result_axis(ax)


def export_source(metrics: pd.DataFrame) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SOURCE / "scdesign3_mechanism_triptych_metrics.csv", index=False)
    (
        metrics.groupby(["method", "mechanism"], as_index=False)["aupr"]
        .mean()
        .to_csv(SOURCE / "scdesign3_mechanism_triptych_mean.csv", index=False)
    )
    panel_rows: list[dict[str, object]] = []
    for mechanism_group, mechanisms in {
        "Contact": ["Contact"],
        "Paracrine": ["Short", "Medium"],
        "Niche": ["Niche"],
    }.items():
        sub = mean_metric(metrics, mechanisms)
        sub["mechanism_group"] = mechanism_group
        panel_rows.append(sub)
    pd.concat(panel_rows, ignore_index=True)[["method", "mechanism_group", "aupr"]].to_csv(
        SOURCE / "scdesign3_mechanism_triptych_panel_aupr.csv",
        index=False,
    )
    pd.DataFrame(
        [
            ("a", "contact-dependent communication schematic", "schematic"),
            ("b", "paracrine communication schematic", "schematic"),
            ("c", "niche-restricted communication schematic", "schematic"),
            ("d", "contact AUPR by method", "scdesign3_mechanism_triptych_panel_aupr.csv"),
            ("e", "short/medium paracrine mean AUPR by method", "scdesign3_mechanism_triptych_panel_aupr.csv"),
            ("f", "niche AUPR by method", "scdesign3_mechanism_triptych_panel_aupr.csv"),
        ],
        columns=["panel", "content", "source_data"],
    ).to_csv(SOURCE / "scdesign3_mechanism_triptych_manifest.csv", index=False)


def save_outputs(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def main() -> None:
    apply_style()
    metrics = load_mechanism_metrics()
    export_source(metrics)

    fig = plt.figure(figsize=(7.25, 4.12), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.98, 1.08],
        left=0.075,
        right=0.985,
        bottom=0.112,
        top=0.932,
        wspace=0.45,
        hspace=0.31,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_contact(ax_a)
    draw_paracrine(ax_b)
    draw_niche(ax_c)
    draw_lollipop(ax_d, metrics, ["Contact"], "Contact recovery", "d")
    draw_lollipop(ax_e, metrics, ["Short", "Medium"], "Paracrine recovery", "e")
    draw_lollipop(ax_f, metrics, ["Niche"], "Niche recovery", "f")

    stem = OUT / "scdesign3_mechanism_triptych"
    save_outputs(fig, stem)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
