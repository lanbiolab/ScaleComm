#!/usr/bin/env python
"""Plot Phase 6 SpatialSelf-specific LR discovery figure and source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_LR_ID = "CD48_CD244_Tcell_phase6_contact_weak"


METHOD_COLORS = {
    "SpatialSelf": "#3A6EA5",
    "COMMOT": "#7D8790",
    "CellChat": "#A9A9A9",
    "SpaTalk": "#8C7AA9",
    "CellNEST": "#B7A99A",
}

REGION_COLORS = {
    "T cell zone": "#3A6EA5",
    "B follicle": "#D6A33D",
    "Germinal center": "#7AA974",
    "Mixed": "#8E8E8E",
    "Tumor boundary": "#B05A5A",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase6_spatialself_specific_discovery_expanded.yaml",
    )
    parser.add_argument("--target-lr-id", default=TARGET_LR_ID)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output_dir"]
    figure_dir = output_dir / "figures"
    source_dir = output_dir / "figure_source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    data = _load_inputs(config, output_dir)
    target = _select_target(data["candidates"], args.target_lr_id)
    source_tables = _build_source_tables(config, data, target, source_dir)
    figure_path_base = figure_dir / "fig_phase6_spatialself_specific_discovery"
    _draw_figure(source_tables, target, figure_path_base)
    index_path = _write_source_index(source_tables, source_dir)
    _write_manifest(config_path, figure_path_base, source_tables, index_path, target, figure_dir)

    print(f"Wrote Phase 6 figure: {figure_path_base}.pdf/.svg/.tiff")
    print(f"Wrote source data: {source_dir}")


def _load_inputs(config: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    dataset_dir = PROJECT_ROOT / config["primary_dataset_dir"]
    spatialself_prediction_path = PROJECT_ROOT / config["methods"]["spatialself"]["prediction_path"]
    return {
        "rank": pd.read_csv(output_dir / "method_lr_rank_table.csv"),
        "candidates": pd.read_csv(output_dir / "spatialself_only_top10_candidates.csv"),
        "annotation": pd.read_csv(output_dir / "weak_or_context_specific_annotation.csv"),
        "validation": pd.read_csv(output_dir / "spatialself_only_candidate_validation.csv"),
        "fdr": pd.read_csv(output_dir / "spatialself_only_candidate_fdr.csv"),
        "cells": pd.read_csv(dataset_dir / "cells.csv"),
        "expression": pd.read_csv(dataset_dir / "expression.csv"),
        "candidate_edges": pd.read_csv(dataset_dir / "candidate_interactions.csv"),
        "spatialself_predictions": pd.read_csv(spatialself_prediction_path),
    }


def _select_target(candidates: pd.DataFrame, target_lr_id: str) -> dict:
    target_rows = candidates.loc[candidates["lr_id"] == target_lr_id]
    if target_rows.empty:
        strict = candidates.loc[candidates["candidate_status"] == "candidate"].copy()
        strict = strict.loc[strict["true_lr"].astype(str).str.lower().eq("true")]
        if strict.empty:
            raise ValueError("No true strict SpatialSelf-only candidate is available for plotting.")
        strict = strict.sort_values("spatialself_rank", ascending=True)
        return strict.iloc[0].to_dict()
    target = target_rows.iloc[0].to_dict()
    if target.get("candidate_status") != "candidate":
        raise ValueError(f"Target LR is not a strict SpatialSelf-only candidate: {target_lr_id}")
    return target


def _build_source_tables(config: dict, data: dict[str, pd.DataFrame], target: dict, source_dir: Path) -> dict[str, Path]:
    lr_id = target["lr_id"]
    source_tables: dict[str, Path] = {}

    rank_table = (
        data["rank"]
        .loc[data["rank"]["lr_id"] == lr_id, ["method_label", "method", "lr_rank", "lr_score", "absent"]]
        .copy()
        .sort_values("lr_rank")
    )
    rank_table["rank_class"] = np.where(rank_table["lr_rank"] <= 10, "top10", "rank_gt10")
    source_tables["A"] = _write_panel(source_dir, "panel_a_method_ranks.csv", rank_table)

    scored_edges = (
        data["candidate_edges"].loc[data["candidate_edges"]["lr_id"] == lr_id].merge(
            data["spatialself_predictions"][["candidate_id", "score"]],
            on="candidate_id",
            how="left",
        )
    )
    cells = data["cells"].copy()
    incoming = (
        scored_edges.groupby("receiver_id", as_index=False)
        .agg(
            incoming_spatialself_score=("score", "sum"),
            incoming_edge_count=("candidate_id", "count"),
            max_edge_score=("score", "max"),
        )
        .rename(columns={"receiver_id": "cell_id"})
    )
    incoming = cells.merge(incoming, on="cell_id", how="left")
    for col in ["incoming_spatialself_score", "incoming_edge_count", "max_edge_score"]:
        incoming[col] = incoming[col].fillna(0.0)
    incoming["incoming_score_percentile"] = incoming["incoming_spatialself_score"].rank(pct=True)
    incoming["top10pct_receiver"] = incoming["incoming_score_percentile"] >= 0.9
    incoming["candidate_lr_id"] = lr_id
    source_tables["B"] = _write_panel(
        source_dir,
        "panel_b_receiver_incoming_spatialself_scores.csv",
        incoming[
            [
                "cell_id",
                "x",
                "y",
                "region",
                "cell_type",
                "dominant_cell_type",
                "incoming_spatialself_score",
                "incoming_edge_count",
                "max_edge_score",
                "incoming_score_percentile",
                "top10pct_receiver",
                "candidate_lr_id",
            ]
        ],
    )

    expression = data["expression"]
    ligand = str(target["ligand"])
    receptor = str(target["receptor"])
    expression_map = cells[["cell_id", "x", "y", "region", "cell_type", "dominant_cell_type"]].merge(
        expression[["cell_id", ligand, receptor]],
        on="cell_id",
        how="left",
    )
    expression_map = expression_map.rename(columns={ligand: "ligand_expression", receptor: "receptor_expression"})
    expression_map["ligand"] = ligand
    expression_map["receptor"] = receptor
    source_tables["C"] = _write_panel(source_dir, "panel_c_ligand_receptor_expression_maps.csv", expression_map)

    proxy = incoming[["cell_id", "x", "y", "region", "cell_type", "incoming_spatialself_score"]].merge(
        expression[["cell_id", receptor]],
        on="cell_id",
        how="left",
    )
    proxy = proxy.rename(columns={receptor: "receiver_receptor_expression"})
    rho, p_value = stats.spearmanr(proxy["incoming_spatialself_score"], proxy["receiver_receptor_expression"])
    proxy["spearman_rho"] = float(rho)
    proxy["spearman_p_value"] = float(p_value)
    proxy["receiver_response_proxy"] = f"incoming_spatialself_score_vs_receiver_{receptor}_expression"
    source_tables["D"] = _write_panel(source_dir, "panel_d_receiver_response_proxy.csv", proxy)

    fdr = data["fdr"].loc[data["fdr"]["lr_id"] == lr_id].iloc[0].to_dict()
    fdr_rows = []
    for level, p_col, fdr_col, label in [
        ("LR-pair", "spatial_permutation_p", "spatial_permutation_fdr", "Spatial permutation"),
        ("LR-pair", "receiver_permutation_p", "receiver_permutation_fdr", "Receiver permutation"),
        ("LR-pair", "lr_identity_permutation_p", "lr_identity_permutation_fdr", "LR identity permutation"),
        ("Cell-type pair", "spatial_cell_type_pair_p", "spatial_cell_type_pair_fdr", "Spatial permutation"),
        ("Cell-type pair", "receiver_cell_type_pair_p", "receiver_cell_type_pair_fdr", "Receiver permutation"),
        ("Cell-type pair", "lr_identity_cell_type_pair_p", "lr_identity_cell_type_pair_fdr", "LR identity permutation"),
    ]:
        fdr_value = fdr.get(fdr_col, np.nan)
        fdr_rows.append(
            {
                "candidate_lr_id": lr_id,
                "test_level": level,
                "null_model": label,
                "empirical_p_value": fdr.get(p_col, np.nan),
                "bh_fdr": fdr_value,
                "significant_at_0_05": bool(pd.notna(fdr_value) and fdr_value <= 0.05),
            }
        )
    source_tables["E"] = _write_panel(source_dir, "panel_e_permutation_fdr_summary.csv", pd.DataFrame(fdr_rows))

    controls = data["candidates"].copy()
    controls["control_class"] = np.select(
        [
            controls["lr_id"].eq(lr_id),
            controls["mechanism"].astype(str).str.contains("false_positive", regex=False),
            controls["candidate_status"].eq("candidate") & ~controls["true_lr"].astype(str).str.lower().eq("true"),
            controls["true_lr"].astype(str).str.lower().eq("true"),
        ],
        ["SpatialSelf-specific true candidate", "High-expression false positive", "Background candidate control", "Other true LR"],
        default="Screened/background LR",
    )
    controls["top10_by_spatialself"] = controls["spatialself_rank"] <= 10
    controls = controls[
        [
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
            "true_lr",
            "candidate_status",
            "control_class",
            "spatialself_rank",
            "spatialself_score",
            "commot_rank",
            "cellchat_rank",
            "spatalk_rank",
            "cellnest_rank",
            "top10_by_spatialself",
        ]
    ].sort_values(["control_class", "spatialself_rank"])
    source_tables["F"] = _write_panel(source_dir, "panel_f_control_rank_context.csv", controls)

    summary = pd.DataFrame(
        [
            {
                "candidate_lr_id": lr_id,
                "ligand": target["ligand"],
                "receptor": target["receptor"],
                "mechanism": target["mechanism"],
                "spatialself_rank": target["spatialself_rank"],
                "commot_rank": target["commot_rank"],
                "cellchat_rank": target["cellchat_rank"],
                "spatalk_rank": target["spatalk_rank"],
                "cellnest_rank": target["cellnest_rank"],
                "rank_claim": "SpatialSelf top 10 and all retained baselines rank > 10",
                "interpretation_limit": "Benchmark candidate validation; not a native-tissue novel mechanism claim.",
            }
        ]
    )
    source_tables["summary"] = _write_panel(source_dir, "figure_candidate_summary.csv", summary)
    return source_tables


def _draw_figure(source_tables: dict[str, Path], target: dict, figure_path_base: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
        }
    )

    rank_table = pd.read_csv(source_tables["A"])
    incoming = pd.read_csv(source_tables["B"])
    expression = pd.read_csv(source_tables["C"])
    proxy = pd.read_csv(source_tables["D"])
    fdr = pd.read_csv(source_tables["E"])
    controls = pd.read_csv(source_tables["F"])

    fig = plt.figure(figsize=(7.4, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.065,
        wspace=0.48,
        hspace=0.62,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[2, 0:2])
    ax_f = fig.add_subplot(gs[2, 2:4])

    _panel_label(ax_a, "a")
    _plot_rank_panel(ax_a, rank_table)
    _panel_label(ax_b, "b")
    _plot_spatial_score_panel(ax_b, incoming)
    _panel_label(ax_c1, "c")
    _plot_expression_panel(ax_c1, expression, "ligand_expression", str(target["ligand"]))
    _plot_expression_panel(ax_c2, expression, "receptor_expression", str(target["receptor"]))
    _panel_label(ax_d, "d")
    _plot_receiver_proxy_panel(ax_d, proxy)
    _panel_label(ax_e, "e")
    _plot_fdr_panel(ax_e, fdr)
    _panel_label(ax_f, "f")
    _plot_control_panel(ax_f, controls, str(target["lr_id"]))

    fig.suptitle(
        "SpatialSelf prioritizes weak/context-specific CD48-CD244 communication",
        x=0.055,
        ha="left",
        y=0.992,
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.964,
        "Expanded lymph-node real-background benchmark; retained baselines rank the candidate outside the top 10.",
        ha="left",
        va="top",
        fontsize=7,
        color="#4A4A4A",
    )

    fig.savefig(f"{figure_path_base}.svg", bbox_inches="tight")
    fig.savefig(f"{figure_path_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{figure_path_base}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_panel(ax: plt.Axes, rank_table: pd.DataFrame) -> None:
    rank_table = rank_table.sort_values("lr_rank", ascending=False)
    colors = [METHOD_COLORS.get(label, "#8A8A8A") for label in rank_table["method_label"]]
    ax.barh(rank_table["method_label"], rank_table["lr_rank"], color=colors, height=0.62)
    ax.axvline(10.5, color="#B64D4D", lw=0.9, ls="--")
    ax.text(10.65, -0.53, "top 10 cutoff", color="#B64D4D", fontsize=6, va="center")
    for _, row in rank_table.iterrows():
        ax.text(float(row["lr_rank"]) + 0.35, row["method_label"], f"rank {int(row['lr_rank'])}", va="center", fontsize=6)
    ax.set_xlabel("LR-pair rank")
    ax.set_ylabel("")
    ax.set_xlim(0, max(27, rank_table["lr_rank"].max() + 3))
    ax.set_title("CD48-CD244 rank across methods", loc="left", fontsize=8, pad=5)
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)


def _plot_spatial_score_panel(ax: plt.Axes, incoming: pd.DataFrame) -> None:
    base_colors = incoming["region"].map(REGION_COLORS).fillna("#BDBDBD")
    ax.scatter(incoming["x"], incoming["y"], s=10, c=base_colors, alpha=0.28, linewidths=0)
    score = incoming["incoming_spatialself_score"].to_numpy()
    nonzero = np.maximum(score, 0)
    size = 10 + 38 * _scale01(nonzero)
    sc = ax.scatter(
        incoming["x"],
        incoming["y"],
        s=size,
        c=nonzero,
        cmap="viridis",
        alpha=0.92,
        linewidths=0.1,
        edgecolors="#FFFFFF",
    )
    ax.set_title("Receiver incoming SpatialSelf score", loc="left", fontsize=8, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.034, pad=0.015)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("incoming score", fontsize=6)


def _plot_expression_panel(ax: plt.Axes, expression: pd.DataFrame, column: str, title: str) -> None:
    values = expression[column].to_numpy()
    sc = ax.scatter(
        expression["x"],
        expression["y"],
        s=13,
        c=values,
        cmap="magma",
        alpha=0.9,
        linewidths=0,
    )
    ax.set_title(f"{title} expression", loc="left", fontsize=8, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.015)
    cbar.ax.tick_params(labelsize=6, length=2)


def _plot_receiver_proxy_panel(ax: plt.Axes, proxy: pd.DataFrame) -> None:
    x = proxy["receiver_receptor_expression"].to_numpy()
    y = proxy["incoming_spatialself_score"].to_numpy()
    region_colors = proxy["region"].map(REGION_COLORS).fillna("#A6A6A6")
    ax.scatter(x, y, s=15, c=region_colors, alpha=0.72, linewidths=0)
    if len(np.unique(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#222222", lw=1)
    rho = proxy["spearman_rho"].iloc[0]
    p_value = proxy["spearman_p_value"].iloc[0]
    ax.text(
        0.03,
        0.95,
        f"Spearman rho = {rho:.2f}\np = {p_value:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel("Receiver CD244 expression")
    ax.set_ylabel("Incoming SpatialSelf score")
    ax.set_title("Receiver response proxy", loc="left", fontsize=8, pad=5)
    ax.grid(color="#E8E8E8", lw=0.5)


def _plot_fdr_panel(ax: plt.Axes, fdr: pd.DataFrame) -> None:
    plot_df = fdr.copy()
    plot_df["label"] = np.where(plot_df["test_level"].eq("LR-pair"), "LR: ", "CT: ") + plot_df["null_model"].str.replace(" permutation", "", regex=False)
    plot_df = plot_df.iloc[::-1]
    colors = np.where(plot_df["bh_fdr"] <= 0.05, "#3A6EA5", "#C7C7C7")
    ax.barh(plot_df["label"], plot_df["bh_fdr"], color=colors, height=0.62)
    ax.axvline(0.05, color="#B64D4D", lw=0.9, ls="--")
    ax.text(0.052, -0.55, "FDR 0.05", color="#B64D4D", fontsize=6, va="center")
    for _, row in plot_df.iterrows():
        ax.text(min(float(row["bh_fdr"]) + 0.018, 0.94), row["label"], f"{row['bh_fdr']:.3g}", va="center", fontsize=6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("BH-adjusted FDR")
    ax.set_title("Permutation support", loc="left", fontsize=8, pad=5)
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)


def _plot_control_panel(ax: plt.Axes, controls: pd.DataFrame, target_lr_id: str) -> None:
    class_order = [
        "SpatialSelf-specific true candidate",
        "High-expression false positive",
        "Background candidate control",
        "Other true LR",
        "Screened/background LR",
    ]
    controls = controls.copy()
    controls["control_class"] = pd.Categorical(controls["control_class"], categories=class_order, ordered=True)
    key_rows = controls.loc[
        controls["control_class"].isin(
            [
                "SpatialSelf-specific true candidate",
                "High-expression false positive",
                "Background candidate control",
            ]
        )
    ].copy()
    key_rows = key_rows.sort_values(["control_class", "spatialself_rank"])
    y = np.arange(len(key_rows))
    palette = {
        "SpatialSelf-specific true candidate": "#3A6EA5",
        "High-expression false positive": "#B64D4D",
        "Background candidate control": "#A6A6A6",
    }
    colors = [palette[str(c)] for c in key_rows["control_class"]]
    ax.scatter(key_rows["spatialself_rank"], y, s=24, c=colors, zorder=3)
    for yi, (_, row) in zip(y, key_rows.iterrows()):
        if row["lr_id"] == target_lr_id or row["spatialself_rank"] <= 10:
            label = f"{row['ligand']}-{row['receptor']}"
            ax.text(float(row["spatialself_rank"]) + 0.35, yi, label, va="center", fontsize=5.8)
    ax.axvline(10.5, color="#B64D4D", lw=0.9, ls="--")
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("SpatialSelf rank")
    ax.set_title("Control rank context", loc="left", fontsize=8, pad=5)
    ax.set_xlim(0, 27)
    ax.grid(axis="x", color="#E8E8E8", lw=0.5)
    high_expr = controls.loc[controls["control_class"].eq("High-expression false positive")]
    n_high_top10 = int((high_expr["spatialself_rank"] <= 10).sum())
    ax.text(
        0.98,
        0.05,
        f"High-expression controls in top 10: {n_high_top10}/{len(high_expr)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def _scale01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _write_panel(source_dir: Path, filename: str, table: pd.DataFrame) -> Path:
    path = source_dir / filename
    table.to_csv(path, index=False)
    return path


def _write_source_index(source_tables: dict[str, Path], source_dir: Path) -> Path:
    descriptions = {
        "A": "Method-level LR-pair ranks for the CD48-CD244 strict candidate.",
        "B": "Receiver-cell incoming SpatialSelf scores used for the spatial score map.",
        "C": "Cell-level CD48 ligand and CD244 receptor expression used for expression maps.",
        "D": "Receiver response proxy: incoming SpatialSelf score versus receiver CD244 expression.",
        "E": "Permutation empirical p-values and BH-FDR values for LR-pair and cell-type-pair tests.",
        "F": "SpatialSelf rank context for true candidate, high-expression false positives, and controls.",
        "summary": "Candidate-level figure summary and interpretation boundary.",
    }
    rows = []
    for panel, path in source_tables.items():
        rows.append(
            {
                "panel": panel,
                "source_data_file": str(path.relative_to(PROJECT_ROOT)),
                "description": descriptions.get(panel, ""),
                "n_rows": len(pd.read_csv(path)),
            }
        )
    index = pd.DataFrame(rows)
    index_path = source_dir / "phase6_figure_source_data_index.csv"
    index.to_csv(index_path, index=False)
    return index_path


def _write_manifest(
    config_path: Path,
    figure_path_base: Path,
    source_tables: dict[str, Path],
    index_path: Path,
    target: dict,
    figure_dir: Path,
) -> None:
    manifest = {
        "figure": str(figure_path_base.relative_to(PROJECT_ROOT)),
        "exports": [
            str(Path(f"{figure_path_base}.svg").relative_to(PROJECT_ROOT)),
            str(Path(f"{figure_path_base}.pdf").relative_to(PROJECT_ROOT)),
            str(Path(f"{figure_path_base}.tiff").relative_to(PROJECT_ROOT)),
        ],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "target_lr_id": target["lr_id"],
        "source_data_index": str(index_path.relative_to(PROJECT_ROOT)),
        "source_tables": {panel: str(path.relative_to(PROJECT_ROOT)) for panel, path in source_tables.items()},
        "claim": "SpatialSelf prioritizes weak/context-specific CD48-CD244 in the expanded lymph-node benchmark while retained baselines rank it outside the top 10.",
        "limitation": "Receiver response is a receptor-expression proxy, not curated downstream target validation.",
    }
    (figure_dir / "fig_phase6_spatialself_specific_discovery_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
