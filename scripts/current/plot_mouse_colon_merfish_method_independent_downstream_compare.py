#!/usr/bin/env python
"""Method-independent downstream response comparison for DSS9 Cxcl10-Cxcr3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.current.plot_mouse_colon_merfish_method_downstream_v3 import (
    BASE,
    GRID,
    LR_CASE as DEFAULT_LR_CASE,
    METHOD_COLORS,
    METHODS,
    MUTED,
    _receiver_scores_for_method,
    load_expression_slice,
    set_spatial_limits,
    setup_style,
)
from scripts.current.run_mouse_colon_merfish_downstream_signature_validation import CURATED_SIGNATURES, LR_SIGNATURE_MAP
from src.spitialself.fdr import bh_adjust


OUT = BASE / "figures/merfish_dss_method_independent_downstream_compare"
LR_PAIR = DEFAULT_LR_CASE
SIGNATURE = LR_SIGNATURE_MAP[LR_PAIR]
STAGE = "DSS9"
REPLICATES = [1, 2]
TOP_FRACTION = 0.20
N_PERMUTATIONS = 2000
SEED = 20260621

INK = "#232323"
BACKGROUND = "#DCD7CE"
MODULE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "curated_response_module",
    ["#FBF5EA", "#EAC77D", "#C86E4E", "#652B2B"],
)

SIGNATURE_LABELS = {
    "Cxcr3_ifng_response": "curated IFN/CXCR3\nresponse module",
    "Cxcr4_repair_remodeling": "curated repair/remodeling\nresponse module",
    "Ccr2_monocyte_inflammation": "curated monocyte inflammation\nresponse module",
    "Ccr5_tcell_myeloid_inflammation": "curated T cell/myeloid\nresponse module",
    "Tnf_nfkb_response": "curated TNF/NF-kB\nresponse module",
}


def safe_lr_name(lr_pair: str) -> str:
    return lr_pair.replace("-", "_").replace("/", "_")


def curated_response_genes(lr_pair: str | None = None) -> list[str]:
    lr_pair = lr_pair or LR_PAIR
    ligand, receptor = lr_pair.split("-")
    blocked = {ligand, receptor}
    return [gene for gene in CURATED_SIGNATURES[SIGNATURE] if gene not in blocked]


def response_module_cells(replicate: int) -> pd.DataFrame:
    expr = load_expression_slice(STAGE, replicate).copy()
    genes = [gene for gene in curated_response_genes() if gene in expr.columns]
    if len(genes) < 3:
        raise ValueError(f"Too few response genes present: {genes}")
    scores = np.log1p(expr[genes].to_numpy(float)).mean(axis=1)
    out = expr[["cell_id", "x", "y"]].copy()
    out["response_score"] = scores
    out["n_response_genes"] = len(genes)
    out["response_genes"] = ";".join(genes)
    return out


def method_receiver_table(method: str, replicate: int) -> pd.DataFrame:
    scores, _ = _receiver_scores_for_method(STAGE, LR_PAIR, method, replicate)
    return pd.DataFrame({"cell_id": scores.index, "receiver_score": scores.values})


def evaluate_method(method: str, replicate: int, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    cells = response_module_cells(replicate)
    receiver = method_receiver_table(method, replicate)
    data = cells.merge(receiver, on="cell_id", how="inner")
    data = data.sort_values("receiver_score", ascending=False).reset_index(drop=True)
    k = max(5, int(np.floor(TOP_FRACTION * len(data))))
    data["is_top_hotspot"] = False
    data.loc[: k - 1, "is_top_hotspot"] = True
    top = data.loc[data["is_top_hotspot"], "response_score"].to_numpy(float)
    rest = data.loc[~data["is_top_hotspot"], "response_score"].to_numpy(float)
    observed = float(top.mean() - rest.mean())

    null = np.empty(N_PERMUTATIONS, dtype=float)
    values = data["response_score"].to_numpy(float)
    n = len(values)
    all_idx = np.arange(n)
    for i in range(N_PERMUTATIONS):
        top_idx = rng.choice(all_idx, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True
        null[i] = float(values[mask].mean() - values[~mask].mean())
    p_one_sided = float((np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1))
    try:
        mw_p = float(mannwhitneyu(top, rest, alternative="greater").pvalue)
    except ValueError:
        mw_p = np.nan
    row = {
        "stage": STAGE,
        "replicate": replicate,
        "method": method,
        "lr_pair": LR_PAIR,
        "signature": SIGNATURE,
        "n_receiver_cells": int(len(data)),
        "n_top_hotspot": int(k),
        "n_response_genes": int(data["n_response_genes"].iloc[0]),
        "response_genes": str(data["response_genes"].iloc[0]),
        "top_response_mean": float(top.mean()),
        "rest_response_mean": float(rest.mean()),
        "observed_effect": observed,
        "null_mean": float(null.mean()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "empirical_p_value": p_one_sided,
        "mannwhitney_p": mw_p,
    }
    data["method"] = method
    data["replicate"] = replicate
    return row, data


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    cell_rows = []
    for replicate in REPLICATES:
        for method in METHODS:
            row, data = evaluate_method(method, replicate, rng)
            rows.append(row)
            cell_rows.append(data)
    result = pd.DataFrame(rows)
    result["bh_fdr"] = bh_adjust(result["empirical_p_value"].to_numpy(float))
    prefix = safe_lr_name(LR_PAIR)
    result.to_csv(OUT / f"{prefix}_method_independent_downstream_enrichment.csv", index=False)
    cells = pd.concat(cell_rows, ignore_index=True)
    cells.to_csv(OUT / f"{prefix}_method_independent_downstream_cells.csv", index=False)
    summary = (
        result.groupby("method", as_index=False)
        .agg(
            mean_effect=("observed_effect", "mean"),
            min_effect=("observed_effect", "min"),
            n_positive=("observed_effect", lambda x: int((x > 0).sum())),
            min_empirical_p=("empirical_p_value", "min"),
            max_empirical_p=("empirical_p_value", "max"),
            min_bh_fdr=("bh_fdr", "min"),
        )
        .sort_values("mean_effect", ascending=False)
    )
    summary.to_csv(OUT / f"{prefix}_method_independent_downstream_summary.csv", index=False)
    return result, cells


def draw_module_background(ax: plt.Axes, cells: pd.DataFrame, *, low_alpha: float = 0.25) -> None:
    values = cells["response_score"].to_numpy(float)
    positive = values > 0
    vmax = float(np.quantile(values[positive], 0.985)) if positive.any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = np.clip(values / vmax, 0, 1)
    visible = norm > 0.02
    ax.scatter(cells["x"], cells["y"], s=1.10, color=BACKGROUND, alpha=low_alpha, linewidths=0, zorder=1)
    ax.scatter(
        cells.loc[visible, "x"],
        cells.loc[visible, "y"],
        c=norm[visible],
        s=2.8 + 13.5 * np.power(norm[visible], 0.78),
        cmap=MODULE_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.90,
        edgecolors="none",
        zorder=2,
    )
    set_spatial_limits(ax, cells)


def draw_colorbar(ax: plt.Axes) -> None:
    cax = ax.inset_axes([0.53, -0.115, 0.34, 0.045])
    for i in range(96):
        cax.add_patch(Rectangle((i / 96, 0), 1 / 96, 1, facecolor=MODULE_CMAP(i / 95), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["low", "high"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=0.7, labelsize=5.3)
    cax.set_xlabel("curated response", fontsize=5.4, color=MUTED, labelpad=0.6)
    for spine in cax.spines.values():
        spine.set_visible(False)


def plot_figure(result: pd.DataFrame, cell_rows: pd.DataFrame) -> None:
    setup_style()
    mpl.rcParams.update(
        {
            "font.size": 6.6,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    )
    rep1 = response_module_cells(1)
    fig = plt.figure(figsize=(7.20, 4.15))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.03, 0.92],
        left=0.055,
        right=0.975,
        bottom=0.145,
        top=0.890,
        hspace=0.420,
    )
    top = outer[0, 0].subgridspec(1, 4, width_ratios=[1.05, 1, 1, 1], wspace=0.060)
    axes = [fig.add_subplot(top[0, i]) for i in range(4)]

    draw_module_background(axes[0], rep1)
    axes[0].set_title(SIGNATURE_LABELS.get(SIGNATURE, "curated response\nmodule"), fontsize=7.0, pad=1.8)
    draw_colorbar(axes[0])
    for i, method in enumerate(METHODS, start=1):
        data = cell_rows.loc[(cell_rows["replicate"] == 1) & (cell_rows["method"] == method)].copy()
        draw_module_background(axes[i], rep1, low_alpha=0.18)
        top_cells = data.loc[data["is_top_hotspot"]]
        axes[i].scatter(
            top_cells["x"],
            top_cells["y"],
            s=34,
            facecolors="none",
            edgecolors=METHOD_COLORS[method],
            linewidths=0.86,
            alpha=0.96,
            zorder=5,
        )
        axes[i].set_title(method, color=METHOD_COLORS[method], fontsize=7.1, pad=1.8)

    fig.text(axes[0].get_position().x0 - 0.026, axes[0].get_position().y1 + 0.012, "a", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(axes[1].get_position().x0 - 0.026, axes[1].get_position().y1 + 0.012, "b", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(
        axes[1].get_position().x0,
        axes[1].get_position().y1 + 0.040,
        f"DSS9 {LR_PAIR}: top receiver hotspots over method-independent response",
        fontsize=7.6,
        ha="left",
        va="bottom",
        color=INK,
    )

    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.310)
    ax_eff = fig.add_subplot(bottom[0, 0])
    ax_score = fig.add_subplot(bottom[0, 1])

    method_x = {method: i for i, method in enumerate(METHODS)}
    for method in METHODS:
        sub = result.loc[result["method"] == method].sort_values("replicate")
        x = method_x[method]
        for row in sub.itertuples(index=False):
            ax_eff.plot([x - 0.18, x + 0.18], [row.null_q025, row.null_q025], color="#CFC8BE", lw=0.55, zorder=1)
            ax_eff.plot([x - 0.18, x + 0.18], [row.null_q975, row.null_q975], color="#CFC8BE", lw=0.55, zorder=1)
            ax_eff.vlines(x, row.null_q025, row.null_q975, color="#DED8CF", lw=3.2, alpha=0.72, zorder=1)
        jitter = np.array([-0.055, 0.055])[: len(sub)]
        ax_eff.scatter(
            np.full(len(sub), x) + jitter,
            sub["observed_effect"],
            s=26,
            color=METHOD_COLORS[method],
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )
        if len(sub):
            ax_eff.plot([x - 0.16, x + 0.16], [sub["observed_effect"].mean()] * 2, color=METHOD_COLORS[method], lw=1.45, zorder=5)
    ax_eff.axhline(0, color="#8D877F", lw=0.65, linestyle=(0, (2.2, 2.2)))
    ax_eff.set_xticks(range(len(METHODS)))
    ax_eff.set_xticklabels(METHODS)
    ax_eff.set_ylabel("top hotspot - rest\nresponse score")
    ax_eff.grid(axis="y", color=GRID, lw=0.45)
    ax_eff.tick_params(axis="x", length=0)
    ax_eff.spines["top"].set_visible(False)
    ax_eff.spines["right"].set_visible(False)
    ax_eff.set_title("hotspot enrichment versus random receiver sets", loc="left", pad=2.0)
    fig.text(ax_eff.get_position().x0 - 0.026, ax_eff.get_position().y1 + 0.012, "c", fontsize=9.0, fontweight="bold", color=INK)

    score_rows = []
    for method in METHODS:
        for replicate in REPLICATES:
            data = cell_rows.loc[(cell_rows["method"] == method) & (cell_rows["replicate"] == replicate)]
            score_rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "Top hotspot": float(data.loc[data["is_top_hotspot"], "response_score"].mean()),
                    "Other receivers": float(data.loc[~data["is_top_hotspot"], "response_score"].mean()),
                }
            )
    score = pd.DataFrame(score_rows)
    categories = ["Top hotspot", "Other receivers"]
    offsets = [-0.13, 0.13]
    fill = {"Top hotspot": "#9E332E", "Other receivers": "#BEB7AC"}
    for i, method in enumerate(METHODS):
        sub = score.loc[score["method"] == method]
        for cat, offset in zip(categories, offsets):
            vals = sub[cat].to_numpy(float)
            ax_score.scatter(
                np.full(len(vals), i + offset) + np.linspace(-0.020, 0.020, len(vals)),
                vals,
                s=22,
                color=fill[cat] if cat == "Top hotspot" else "#AFA89F",
                edgecolors="white",
                linewidths=0.30,
                alpha=0.96,
                zorder=3,
            )
            ax_score.plot([i + offset - 0.07, i + offset + 0.07], [vals.mean(), vals.mean()], color=fill[cat], lw=1.25)
    ax_score.set_xticks(range(len(METHODS)))
    ax_score.set_xticklabels(METHODS)
    ax_score.set_ylabel("response score")
    ax_score.grid(axis="y", color=GRID, lw=0.45)
    ax_score.tick_params(axis="x", length=0)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    ax_score.set_title("receiver response levels", loc="left", pad=2.0)
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=fill["Top hotspot"], markersize=4.1, label="top hotspot"),
        Line2D([0], [0], marker="o", lw=0, color="#AFA89F", markersize=4.1, label="other receivers"),
    ]
    ax_score.legend(handles=handles, loc="upper right", frameon=False, fontsize=5.7, handletextpad=0.25, borderaxespad=0.2)
    fig.text(ax_score.get_position().x0 - 0.026, ax_score.get_position().y1 + 0.012, "d", fontsize=9.0, fontweight="bold", color=INK)

    fig.text(
        0.055,
        0.035,
        "Curated response genes exclude the tested ligand/receptor; dots show DSS9 biological replicates; grey bars show 95% receiver-permutation null.",
        fontsize=5.8,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    export(fig, OUT / f"dss9_{safe_lr_name(LR_PAIR).lower()}_method_independent_downstream_compare")


def export(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global LR_PAIR, SIGNATURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-pair", default=DEFAULT_LR_CASE, choices=sorted(LR_SIGNATURE_MAP))
    args = parser.parse_args()
    LR_PAIR = args.lr_pair
    SIGNATURE = LR_SIGNATURE_MAP[LR_PAIR]
    OUT.mkdir(parents=True, exist_ok=True)
    result, cells = run_analysis()
    plot_figure(result, cells)
    svg = OUT / f"dss9_{safe_lr_name(LR_PAIR).lower()}_method_independent_downstream_compare.svg"
    print(result.sort_values(["method", "replicate"]).to_string(index=False))
    print(f"{OUT} image_tags={svg.read_text(errors='ignore').count('<image')}")


if __name__ == "__main__":
    main()
