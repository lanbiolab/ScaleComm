#!/usr/bin/env python
"""Train a lightweight SpatialSelf range-context diagnostic and plot learned gates.

The existing benchmark runs store predictions but not model weights. This script
re-trains the final range-context configuration on scDesign3 200-LR references,
then exports internal range/context gate diagnostics before the model leaves
memory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import run_spatialself as rs
from src.spitialself.evaluation import evaluate_predictions_auto
from src.spitialself.model_factory import build_model


BENCH_200 = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = PROJECT_ROOT / "outputs/current_fig5_lr_scale_mechanism/internal_gate_diagnostic"
SOURCE = OUT / "source_data"
FIGDIR = OUT / "figures"

RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABEL = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}
DATASETS = {
    "lymph_scdesign3_200lr": BENCH_200 / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00",
    "melanoma_scdesign3_200lr": BENCH_200 / "datasets/reference_melanoma_scdesign3_200lr_replicate_00",
}
COL = {
    "ink": "#2B2826",
    "muted": "#6F6860",
    "grid": "#E5E0D7",
    "ours": "#237A78",
    "ours_dark": "#155C5B",
    "contact": "#8A8177",
    "short": "#D9992B",
    "medium": "#C94F48",
    "long_niche": "#237A78",
    "decoy": "#B85450",
    "sand": "#E6C06A",
    "blue": "#2F78A8",
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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
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
    ax.tick_params(labelsize=6.2, length=2, width=0.45, pad=1.5)
    if grid_y:
        ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)
    else:
        ax.grid(False)


def train_and_export(dataset_id: str, replicate_dir: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_design = pd.read_csv(replicate_dir / "lr_panel_design.csv")

    data, tensors = rs._build_training_data(
        cells,
        expression,
        lr_pairs,
        candidates,
        k_nn=12,
        positive_quantile=0.9,
        negative_quantile=0.2,
        pseudo_label_strategy="mechanism_niche_expression_distance",
        soft_labels=False,
        weak_positive_loss_weight=1.0,
        weak_positive_expression_quantile=0.7,
        train_sampling_strategy="balanced_per_lr",
        lr_cap_positive=50,
        lr_cap_negative=200,
        use_spatial_graph=True,
        use_expression_features=True,
        use_cell_type_features=True,
        use_niche_features=True,
        use_coordinate_features=True,
        use_edge_prior_features=True,
        max_train_samples=args.max_train_samples,
        context_contrastive_negative_mode="random_receiver",
        spatial_null_negative_mode="random_receiver",
        background_guard_targets=None,
        seed=args.seed,
    )

    model = build_model(
        "spatialself_range_context",
        in_dim=data.x.shape[1],
        h_dim=64,
        lr_dim=24,
        n_lr=len(lr_pairs),
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        use_lr_embedding=True,
        lr_encoding_mode="learned",
        use_distance=True,
        edge_feature_dim=int(tensors["edge_features"].shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_idx = tensors["train_idx"]
    labels = tensors["pseudo_label"][train_idx]
    sample_weight = tensors["sample_weight"][train_idx]

    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad()
        h = model.encode_nodes(data)
        logits = model.score_edges(
            h,
            tensors["src_idx"][train_idx],
            tensors["dst_idx"][train_idx],
            tensors["lr_idx"][train_idx],
            tensors["distance"][train_idx].unsqueeze(-1),
            tensors["edge_features"][train_idx],
        )
        loss = rs._weighted_bce_or_focal_loss(
            logits,
            labels,
            weight=sample_weight,
            focal_gamma=0.0,
            focal_alpha=0.5,
        )
        hard_negative_loss_weight = 0.05
        hard_neg_idx = tensors["hard_negative_train_idx"]
        if hard_negative_loss_weight > 0 and hard_neg_idx.numel() > 0:
            hard_neg_logits = model.score_edges(
                h,
                tensors["src_idx"][hard_neg_idx],
                tensors["dst_idx"][hard_neg_idx],
                tensors["lr_idx"][hard_neg_idx],
                tensors["distance"][hard_neg_idx].unsqueeze(-1),
                tensors["edge_features"][hard_neg_idx],
            )
            loss = loss + hard_negative_loss_weight * torch.nn.functional.softplus(hard_neg_logits).mean()

        mechanism_guard_loss_weight = 0.03
        mechanism_guard_idx = tensors["mechanism_guard_train_idx"]
        if mechanism_guard_loss_weight > 0 and mechanism_guard_idx.numel() > 0:
            mechanism_guard_logits = model.score_edges(
                h,
                tensors["src_idx"][mechanism_guard_idx],
                tensors["dst_idx"][mechanism_guard_idx],
                tensors["lr_idx"][mechanism_guard_idx],
                tensors["distance"][mechanism_guard_idx].unsqueeze(-1),
                tensors["edge_features"][mechanism_guard_idx],
            )
            mechanism_guard_weight = tensors["mechanism_guard_weight"][mechanism_guard_idx]
            loss = loss + mechanism_guard_loss_weight * (
                torch.nn.functional.softplus(mechanism_guard_logits) * mechanism_guard_weight
            ).mean()
        loss.backward()
        optimizer.step()
        if args.verbose and (epoch + 1) % 100 == 0:
            print(f"{dataset_id} epoch {epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")

    model.eval()
    batch_size = 65536
    scores = []
    range_rows = []
    candidate_gate_rows = []
    with torch.no_grad():
        h = model.encode_nodes(data)
        centers = model.edge_scorer.range_basis_centers.detach().cpu().numpy()
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            src = tensors["src_idx"][start:end]
            dst = tensors["dst_idx"][start:end]
            lr = tensors["lr_idx"][start:end]
            dist = tensors["distance"][start:end].unsqueeze(-1)
            edge_features = tensors["edge_features"][start:end]
            logits = model.score_edges(h, src, dst, lr, dist, edge_features)
            score = torch.sigmoid(logits)
            weights = model.range_distribution(h, src, dst, lr, dists=dist, edge_features=edge_features)
            ctx = _context_gate_value(model, h, src, dst, lr, dist, edge_features)
            compat = _range_compatibility(model, weights, dist)
            scores.append(score.detach().cpu().numpy())
            local = candidates.iloc[start:end].copy()
            for basis_i in range(weights.shape[1]):
                local[f"basis_{basis_i}"] = weights[:, basis_i].detach().cpu().numpy()
            local["score"] = score.detach().cpu().numpy()
            local["context_gate"] = ctx.detach().cpu().numpy()
            local["range_compatibility"] = compat.detach().cpu().numpy()
            range_rows.append(local)
            candidate_gate_rows.append(
                local[
                    [
                        "candidate_id",
                        "lr_id",
                        "mechanism",
                        "distance",
                        "ground_truth",
                        "wrong_range_false_positive",
                        "niche_rule_satisfied",
                        "false_positive_high_expression",
                        "score",
                        "context_gate",
                        "range_compatibility",
                    ]
                ]
            )

    scores_np = np.concatenate(scores)
    prediction = pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "SpatialSelf diagnostic", "score": scores_np})
    metrics = evaluate_predictions_auto(candidates, prediction)
    merged = pd.concat(range_rows, ignore_index=True)
    candidate_gate = pd.concat(candidate_gate_rows, ignore_index=True)

    lr_to_class = lr_design.set_index("lr_id")["range_class"].to_dict()
    merged["range_class"] = merged["lr_id"].map(lr_to_class).fillna("unknown")
    merged = merged[merged["range_class"].isin(RANGE_ORDER)].copy()
    merged["rank_pct_within_lr"] = merged.groupby("lr_id")["score"].rank(pct=True)
    active = merged[merged["rank_pct_within_lr"] >= args.top_lr_quantile].copy()

    basis_cols = [f"basis_{i}" for i in range(len(centers))]
    lr_gate = (
        active.groupby(["lr_id", "range_class"], as_index=False)
        .agg(
            n_active_edges=("candidate_id", "size"),
            score_mean=("score", "mean"),
            context_gate_mean=("context_gate", "mean"),
            range_compatibility_mean=("range_compatibility", "mean"),
            **{col: (col, "mean") for col in basis_cols},
        )
    )
    for i, center in enumerate(centers):
        lr_gate[f"basis_center_{i}"] = float(center)
    weight_mat = lr_gate[basis_cols].to_numpy(dtype=float)
    lr_gate["expected_basis"] = (weight_mat * np.arange(len(basis_cols))[None, :]).sum(axis=1)
    lr_gate["expected_normalized_distance"] = (weight_mat * centers[None, :]).sum(axis=1)
    lr_gate["dataset_id"] = dataset_id

    mechanism_gate = (
        lr_gate.groupby("range_class", as_index=False)
        .agg(
            n_lr=("lr_id", "nunique"),
            expected_basis_mean=("expected_basis", "mean"),
            expected_basis_sem=("expected_basis", _sem),
            expected_normalized_distance_mean=("expected_normalized_distance", "mean"),
            expected_normalized_distance_sem=("expected_normalized_distance", _sem),
            **{col: (col, "mean") for col in basis_cols},
        )
    )
    mechanism_gate["dataset_id"] = dataset_id

    candidate_gate["range_class"] = candidate_gate["lr_id"].map(lr_to_class).fillna("unknown")
    candidate_gate["candidate_group"] = _candidate_group(candidate_gate)
    group_gate = (
        candidate_gate.groupby("candidate_group", as_index=False)
        .agg(
            n_candidates=("candidate_id", "size"),
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            context_gate_mean=("context_gate", "mean"),
            context_gate_median=("context_gate", "median"),
            range_compatibility_mean=("range_compatibility", "mean"),
            range_compatibility_median=("range_compatibility", "median"),
        )
        .sort_values("candidate_group")
    )
    group_gate["dataset_id"] = dataset_id
    candidate_gate["dataset_id"] = dataset_id

    metrics_row = pd.DataFrame([{**metrics, "dataset_id": dataset_id, "epochs": args.epochs, "top_lr_quantile": args.top_lr_quantile}])
    return {
        "lr_gate": lr_gate,
        "mechanism_gate": mechanism_gate,
        "candidate_gate": candidate_gate,
        "group_gate": group_gate,
        "metrics": metrics_row,
    }


def _context_gate_value(model, h, src_idx, dst_idx, lr_ids, dist, edge_features):
    h_src = h[src_idx]
    h_dst = h[dst_idx]
    e_lr = model._encode_lr(lr_ids, h.device)
    parts = [h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, e_lr]
    if model.edge_scorer.use_distance:
        parts.append(dist / model.edge_scorer.distance_scale)
    if model.edge_scorer.edge_feature_dim:
        parts.append(edge_features)
    context_input = torch.cat(parts, dim=-1)
    return model.edge_scorer.context_gate(context_input).squeeze(-1)


def _range_compatibility(model, weights, dist):
    dist_norm = dist.squeeze(-1) / model.edge_scorer.distance_scale
    centers = model.edge_scorer.range_basis_centers
    widths = model.edge_scorer.range_basis_widths
    basis = torch.exp(-0.5 * torch.square((dist_norm.unsqueeze(-1) - centers.unsqueeze(0)) / widths.unsqueeze(0)))
    return torch.sum(weights * basis, dim=-1).clamp_min(1e-6)


def _candidate_group(df: pd.DataFrame) -> pd.Series:
    truth = df["ground_truth"].astype(bool)
    wrong_range = df["wrong_range_false_positive"].astype(bool)
    wrong_niche = (~df["niche_rule_satisfied"].astype(bool)) & (~truth)
    high_expr = df["false_positive_high_expression"].astype(bool) & (~truth)
    group = pd.Series("Other background", index=df.index, dtype="object")
    group[truth] = "True"
    group[wrong_range & (~truth)] = "Wrong range"
    group[wrong_niche] = "Wrong niche"
    group[high_expr] = "High-expression decoy"
    return group


def _sem(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def write_outputs(parts: list[dict[str, pd.DataFrame]], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    combined: dict[str, pd.DataFrame] = {}
    for key in ["lr_gate", "mechanism_gate", "candidate_gate", "group_gate", "metrics"]:
        combined[key] = pd.concat([p[key] for p in parts], ignore_index=True)
        combined[key].to_csv(SOURCE / f"{key}.csv", index=False)

    basis_cols = [c for c in combined["lr_gate"].columns if re.fullmatch(r"basis_\d+", c)]
    class_summary = (
        combined["lr_gate"].groupby("range_class", as_index=False)
        .agg(
            n_lr=("lr_id", "nunique"),
            expected_basis_mean=("expected_basis", "mean"),
            expected_basis_sem=("expected_basis", _sem),
            expected_normalized_distance_mean=("expected_normalized_distance", "mean"),
            expected_normalized_distance_sem=("expected_normalized_distance", _sem),
            **{col: (col, "mean") for col in basis_cols},
        )
    )
    class_summary["range_class"] = pd.Categorical(class_summary["range_class"], RANGE_ORDER, ordered=True)
    class_summary = class_summary.sort_values("range_class")
    class_summary.to_csv(SOURCE / "class_gate_summary.csv", index=False)
    combined["class_gate_summary"] = class_summary

    manifest = {
        "purpose": "Internal range/context gate diagnostic for SpatialSelf.",
        "model": "spatialself_range_context",
        "epochs": args.epochs,
        "top_lr_quantile": args.top_lr_quantile,
        "datasets": list(args.datasets),
        "claim_boundary": "Gate weights are context-dependent model diagnostics, not direct physical signaling-length measurements.",
        "source_data": {k: f"source_data/{k}.csv" for k in combined},
    }
    with open(OUT / "internal_gate_diagnostic_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return combined


def plot_figure(data: dict[str, pd.DataFrame]) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig = plt.figure(figsize=(7.35, 3.15), constrained_layout=False)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 0.95, 1.0], wspace=0.58, left=0.075, right=0.985, top=0.86, bottom=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_heatmap(ax_a, data["class_gate_summary"])
    draw_expected_basis(ax_b, data["lr_gate"])
    draw_gate_components(ax_c, data["group_gate"])
    stem = FIGDIR / "fig5_internal_range_context_gate"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return stem


def draw_heatmap(ax, class_summary: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.16, y=1.03)
    basis_cols = [c for c in class_summary.columns if re.fullmatch(r"basis_\d+", c)]
    mat = class_summary.set_index("range_class").reindex(RANGE_ORDER)[basis_cols].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("range_gate", ["#F7F4EC", "#E6C06A", "#2F78A8"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(0.45, float(np.nanmax(mat))))
    ax.set_xticks(np.arange(len(basis_cols)))
    ax.set_xticklabels([str(i) for i in range(len(basis_cols))], fontsize=6.2)
    ax.set_yticks(np.arange(len(RANGE_ORDER)))
    ax.set_yticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.5)
    ax.set_xlabel("learned range basis", fontsize=6.8, labelpad=2)
    ax.set_title("Internal range-gate weights", fontsize=8.0, fontweight="bold", pad=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(basis_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RANGE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)


def draw_expected_basis(ax, lr_gate: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.18, y=1.03)
    rng = np.random.default_rng(1)
    for i, rc in enumerate(RANGE_ORDER):
        sub = lr_gate[lr_gate["range_class"].eq(rc)]
        x = np.full(len(sub), i, dtype=float) + rng.normal(0, 0.055, len(sub))
        ax.scatter(
            x,
            sub["expected_basis"],
            s=13,
            color=COL[rc],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.25,
            zorder=2,
        )
        med = float(sub["expected_basis"].median())
        ax.plot([i - 0.22, i + 0.22], [med, med], color=COL["ink"], lw=1.2, zorder=3)
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], rotation=25, ha="right", fontsize=6.2)
    ax.set_ylabel("expected range basis", fontsize=6.8)
    ax.set_title("LR-level scale preference", fontsize=8.0, fontweight="bold", pad=3)
    ax.set_xlim(-0.5, len(RANGE_ORDER) - 0.5)
    ax.set_ylim(-0.05, max(2.2, float(lr_gate["expected_basis"].quantile(0.98)) + 0.3))
    clean_axis(ax, grid_y=True)


def draw_gate_components(ax, group_gate: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.16, y=1.03)
    order = ["True", "Wrong range", "Wrong niche", "High-expression decoy", "Other background"]
    metric_label = [
        ("score_mean", "score", COL["ours"]),
        ("range_compatibility_mean", "range compat.", COL["blue"]),
        ("context_gate_mean", "context gate", COL["sand"]),
    ]
    summary = group_gate.groupby("candidate_group", as_index=False).agg(
        score_mean=("score_mean", "mean"),
        range_compatibility_mean=("range_compatibility_mean", "mean"),
        context_gate_mean=("context_gate_mean", "mean"),
    )
    summary = summary.set_index("candidate_group").reindex(order)
    y = np.arange(len(order))
    offsets = [-0.18, 0.0, 0.18]
    for (metric, label, color), off in zip(metric_label, offsets, strict=False):
        ax.scatter(summary[metric], y + off, s=28, color=color, edgecolors="white", linewidths=0.45, label=label, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(["True", "Wrong\nrange", "Wrong\nniche", "High-expr.\ndecoy", "Other"], fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlabel("mean internal value", fontsize=6.8)
    ax.set_title("Gate values by candidate type", fontsize=8.0, fontweight="bold", pad=3)
    ax.set_xlim(0, 1.02)
    clean_axis(ax, grid_y=True)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.3, borderpad=0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lymph_scdesign3_200lr", "melanoma_scdesign3_200lr"], choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--top-lr-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    parts = []
    for dataset_id in args.datasets:
        print(f"Training internal gate diagnostic on {dataset_id} ({args.epochs} epochs)")
        parts.append(train_and_export(dataset_id, DATASETS[dataset_id], args))
    combined = write_outputs(parts, args)
    stem = plot_figure(combined)
    print(f"Wrote {stem}.svg/.pdf/.png/.tiff")
    print(f"Source data: {SOURCE}")


if __name__ == "__main__":
    main()
