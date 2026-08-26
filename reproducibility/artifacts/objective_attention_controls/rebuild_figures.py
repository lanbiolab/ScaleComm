#!/usr/bin/env python
"""Render Response Figs. R1-R4 from the Comment 1 controlled result set.

Figure contract
---------------
Core conclusion: the controlled audits distinguish direct
weak-supervision effects, the scale-aware objective, and matched attention
controls without conflating clean and truth-blinded analyses.
Figure archetype: four independent quantitative response figures.
Target: Nature Communications response letter, vector-first.
Evidence: R1 clean matched controls; R2 scale-target KL audit; R3 truth-blinded
overall/mode/error comparison; R4 inspectable scale profiles.
Review risk: the register contains summary values only, so no error bars or
replicate points are added. Captions identify these as mean results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PANDAS_USE_BOTTLENECK"] = "0"
os.environ["PANDAS_USE_NUMEXPR"] = "0"
sys.modules["bottleneck"] = None
sys.modules["numexpr"] = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
SOURCE = OUT / "source_data/wp22_registered_comparison_results.csv"

BLUE = "#2A6795"
BLUE_LIGHT = "#A7BFCE"
GAT = "#C7CDD0"
INK = "#263844"
GRID = "#D8E1E6"
RED = "#BA514B"


def style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK, width=0.7, length=3)


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 700},
        ".tiff": {"dpi": 700},
    }.items():
        fig.savefig(FIG / f"{stem}{suffix}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


def grouped(ax: plt.Axes, data: pd.DataFrame, groups: list[str], methods: list[str], *, title: str, ylim: tuple[float, float]) -> None:
    x = np.arange(len(groups))
    width = 0.31 if len(methods) == 2 else 0.25
    offsets = np.linspace(-(len(methods) - 1) * width / 2, (len(methods) - 1) * width / 2, len(methods))
    color_map = {
        "ScaleComm diffusion encoder": BLUE,
        "ScaleComm diffusion encoder + auxiliary": BLUE_LIGHT,
        "ScaleComm full": BLUE,
        "Full ScaleComm": BLUE,
        "BCE-only": BLUE_LIGHT,
        "BCE + scale-target KL": BLUE,
        "Full minus scale-target KL": BLUE_LIGHT,
        "Matched GAT": GAT,
        "Matched GAT + auxiliary": GAT,
    }
    for method, offset in zip(methods, offsets):
        vals = [
            float(data.loc[(data.analysis == group) & (data.method == method), "value"].iloc[0])
            for group in groups
        ]
        bars = ax.bar(x + offset, vals, width=width, color=color_map.get(method, BLUE_LIGHT),
                      edgecolor=INK, linewidth=0.6, zorder=2, label=method)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim[1] * 0.025, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Event-level AUPR", fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=7)
    style(ax)


def main() -> None:
    data = pd.read_csv(SOURCE)
    mpl.rcParams.update({"axes.linewidth": 0.8, "font.size": 8})

    # R1: clean objective and matched GAT controls.
    r1 = data.loc[data.figure == "R1"].copy()
    fig, ax = plt.subplots(figsize=(6.15, 3.55))
    groups = [
        ("BCE only", ["ScaleComm diffusion encoder", "Matched GAT"]),
        ("BCE + non-scale\nauxiliary encoder audit", ["ScaleComm diffusion encoder", "Matched GAT"]),
        ("System-control\naudit", ["ScaleComm full", "ScaleComm diffusion encoder control", "Matched GAT control"]),
    ]
    analysis_names = ["Clean BCE", "BCE + non-scale auxiliary encoder audit", "System-control audit"]
    x = np.arange(len(groups))
    for i, (label, methods) in enumerate(groups):
        offsets = np.linspace(-0.16, 0.16, len(methods)) if len(methods) == 2 else np.array([-0.22, 0.0, 0.22])
        for method, offset in zip(methods, offsets):
            value = float(r1.loc[(r1.analysis == analysis_names[i]) & (r1.method == method), "value"].iloc[0])
            color = BLUE if method == "ScaleComm full" else (BLUE_LIGHT if "diffusion encoder" in method else GAT)
            width = 0.25 if len(methods) == 2 else 0.17
            ax.bar(i + offset, value, width=width, color=color, edgecolor=INK,
                   linewidth=0.6, zorder=2)
            ax.text(i + offset, value + 0.017, f"{value:.3f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x, [label for label, _ in groups])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Event-level AUPR", fontsize=8.5, color=INK)
    ax.set_title("Objective and matched-input encoder controls", loc="left", fontsize=10, fontweight="bold", color=INK, pad=7)
    style(ax)
    save(fig, "response_fig_r1_objective_attention_controls")

    # R2: paired and full-objective scale-target KL evidence.
    r2 = data.loc[data.figure == "R2"].copy()
    fig, ax = plt.subplots(figsize=(5.65, 3.5))
    pairs = [
        ("BCE only", "BCE-only", BLUE_LIGHT),
        ("BCE +\nscale-target KL", "BCE + scale-target KL", BLUE),
        ("Full minus\nscale-target\nKL", "Full minus scale-target KL", BLUE_LIGHT),
        ("Full\nScaleComm", "Full ScaleComm", BLUE),
    ]
    x = np.array([0.0, 0.85, 2.65, 3.62])
    for xpos, (label, method, color) in zip(x, pairs):
        value = float(r2.loc[r2.method == method, "value"].iloc[0])
        ax.bar(xpos, value, width=0.52, color=color, edgecolor=INK, linewidth=0.6, zorder=2)
        ax.text(xpos, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.axvline(1.72, color=GRID, linewidth=1.0)
    ax.text(0.425, 0.705, "Paired scale-target KL audit", ha="center", va="top", fontsize=7.2, color=INK)
    ax.text(3.135, 0.705, "Full-objective ablation", ha="center", va="top", fontsize=7.2, color=INK)
    ax.set_xlim(-0.48, 4.10)
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Event-level AUPR", fontsize=8.5, color=INK)
    ax.set_title("Scale-target KL ablation", loc="left", fontsize=10, fontweight="bold", color=INK, pad=7)
    style(ax)
    save(fig, "response_fig_r2_specific_scale_loss_ablation")

    # R3: blinded overall and mode-specific GAT comparison.
    r3 = data.loc[(data.figure == "R3") & (data.metric == "Event-level AUPR")].copy()
    groups = ["Truth-blinded overall", "Truth-blinded contact", "Truth-blinded short-range", "Truth-blinded medium-range", "Truth-blinded long-niche"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    grouped(ax, r3, groups, ["ScaleComm full", "Matched GAT + auxiliary"],
            title="Simulator-truth-blinded ScaleComm and GAT comparison", ylim=(0, 0.38))
    ax.set_xticklabels(["Overall", "Contact", "Short-range", "Medium-range", "Long-niche"])
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Full ScaleComm", "Matched GAT + non-scale aux."],
              frameon=False, fontsize=7, loc="upper right")
    save(fig, "response_fig_r3_candidate_blinded_control")

    # R4: retain full-model scale profiles.
    profiles = pd.read_csv(OUT / "source_data/wp22_full_scale_profiles_by_mode.csv")
    scale_cols = [f"scale_{i}" for i in range(6)]
    order = ["juxtacrine_contact", "short_range_paracrine", "medium_range_paracrine", "long_range_chemokine_niche"]
    labels = ["Contact", "Short-range", "Medium-range", "Long-niche"]
    summary = profiles.groupby("mechanism", as_index=False)[scale_cols].mean()
    summary["order"] = summary.mechanism.map({name: i for i, name in enumerate(order)})
    heat = summary.loc[summary["order"].notna()].sort_values("order")[scale_cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.9, 3.8))
    im = ax.imshow(heat, cmap="Blues", vmin=0, vmax=max(0.55, float(heat.max())), aspect="auto")
    ax.set_title("Full-model scale profiles", loc="left", fontsize=10, fontweight="bold", color=INK, pad=7)
    ax.set_xticks(np.arange(6), ["s0", "s1", "s2", "s3", "s4", "s5"])
    ax.set_yticks(np.arange(4), labels)
    ax.tick_params(length=0, labelsize=8)
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            ax.text(col, row, f"{heat[row, col]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if heat[row, col] > heat.max() * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.06)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("Mean scale weight", fontsize=8)
    save(fig, "response_fig_r4_full_model_scale_profiles")


if __name__ == "__main__":
    main()
