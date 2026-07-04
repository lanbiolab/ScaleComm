#!/usr/bin/env python
"""Train SpatialSelf wavelet-adaptive diagnostic and plot LR-specific scale gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import run_spatialself as rs
from src.spitialself.evaluation import evaluate_predictions_auto
from src.spitialself.model_factory import build_model


BENCH_200 = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = PROJECT_ROOT / "outputs/current_fig5_lr_scale_mechanism/wavelet_scale_gate_diagnostic"
SOURCE = OUT / "source_data"
FIGDIR = OUT / "figures"

DATASETS = {
    "lymph_scdesign3_200lr": BENCH_200 / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00",
    "melanoma_scdesign3_200lr": BENCH_200 / "datasets/reference_melanoma_scdesign3_200lr_replicate_00",
}
RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABEL = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}
SCALE_LABEL = ["raw", "detail 1", "detail 2", "detail 4", "detail 8", "smooth"]
COL = {
    "ink": "#2B2826",
    "muted": "#6F6860",
    "grid": "#E5E0D7",
    "contact": "#8A8177",
    "short": "#D9992B",
    "medium": "#C94F48",
    "long_niche": "#237A78",
    "ours": "#237A78",
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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="bottom")


def clean_axis(ax, grid_y: bool = True) -> None:
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(labelsize=6.2, length=2, width=0.45, pad=1.5)
    if grid_y:
        ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)


def train_dataset(dataset_id: str, replicate_dir: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
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
        "spatialself_wavelet_adaptive",
        in_dim=data.x.shape[1],
        h_dim=64,
        lr_dim=24,
        n_lr=len(lr_pairs),
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
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
        loss = rs._weighted_bce_or_focal_loss(logits, labels, weight=sample_weight, focal_gamma=0.0, focal_alpha=0.5)

        hard_neg_idx = tensors["hard_negative_train_idx"]
        if hard_neg_idx.numel() > 0:
            hard_neg_logits = model.score_edges(
                h,
                tensors["src_idx"][hard_neg_idx],
                tensors["dst_idx"][hard_neg_idx],
                tensors["lr_idx"][hard_neg_idx],
                tensors["distance"][hard_neg_idx].unsqueeze(-1),
                tensors["edge_features"][hard_neg_idx],
            )
            loss = loss + 0.05 * torch.nn.functional.softplus(hard_neg_logits).mean()

        mechanism_guard_idx = tensors["mechanism_guard_train_idx"]
        if mechanism_guard_idx.numel() > 0:
            guard_logits = model.score_edges(
                h,
                tensors["src_idx"][mechanism_guard_idx],
                tensors["dst_idx"][mechanism_guard_idx],
                tensors["lr_idx"][mechanism_guard_idx],
                tensors["distance"][mechanism_guard_idx].unsqueeze(-1),
                tensors["edge_features"][mechanism_guard_idx],
            )
            guard_weight = tensors["mechanism_guard_weight"][mechanism_guard_idx]
            loss = loss + 0.03 * (torch.nn.functional.softplus(guard_logits) * guard_weight).mean()

        target_scale = rs._wavelet_response_scale_targets(
            h,
            tensors["src_idx"],
            tensors["dst_idx"],
            tensors["lr_idx"],
            tensors["scale_alignment_weight"],
            n_lr=len(lr_pairs),
        )
        if target_scale is not None:
            lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
            pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
            loss = loss + args.scale_alignment_weight * torch.nn.functional.kl_div(
                torch.log(pred_scale.clamp_min(1e-8)),
                target_scale,
                reduction="batchmean",
            )
        loss.backward()
        optimizer.step()
        if args.verbose and (epoch + 1) % 100 == 0:
            print(f"{dataset_id} epoch {epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")

    model.eval()
    scores = []
    with torch.no_grad():
        h = model.encode_nodes(data)
        for start in range(0, len(candidates), 65536):
            end = min(start + 65536, len(candidates))
            logits = model.score_edges(
                h,
                tensors["src_idx"][start:end],
                tensors["dst_idx"][start:end],
                tensors["lr_idx"][start:end],
                tensors["distance"][start:end].unsqueeze(-1),
                tensors["edge_features"][start:end],
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
        lr_ids = torch.arange(len(lr_pairs), dtype=torch.long)
        scale = model.lr_scale_distribution(lr_ids).detach().cpu().numpy()

    prediction = pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "SpatialSelf wavelet diagnostic", "score": np.concatenate(scores)})
    metrics = evaluate_predictions_auto(candidates, prediction)
    lr_to_class = lr_design.set_index("lr_id")["range_class"].to_dict()
    rows = []
    for idx, lr_id in enumerate(lr_pairs["lr_id"].astype(str)):
        row = {
            "dataset_id": dataset_id,
            "lr_id": lr_id,
            "range_class": lr_to_class.get(lr_id, "unknown"),
        }
        for scale_i in range(scale.shape[1]):
            row[f"scale_{scale_i}"] = float(scale[idx, scale_i])
        row["expected_scale"] = float((scale[idx] * np.arange(scale.shape[1])).sum())
        row["local_weight"] = float(scale[idx, :2].sum())
        row["broad_weight"] = float(scale[idx, -2:].sum())
        rows.append(row)
    lr_scale = pd.DataFrame(rows)
    lr_scale = lr_scale[lr_scale["range_class"].isin(RANGE_ORDER)].copy()
    metrics_row = pd.DataFrame([{**metrics, "dataset_id": dataset_id, "epochs": args.epochs, "scale_alignment_weight": args.scale_alignment_weight}])
    return {"lr_scale": lr_scale, "metrics": metrics_row}


def sem(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def write_outputs(parts: list[dict[str, pd.DataFrame]], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    lr_scale = pd.concat([p["lr_scale"] for p in parts], ignore_index=True)
    metrics = pd.concat([p["metrics"] for p in parts], ignore_index=True)
    scale_cols = [c for c in lr_scale.columns if c.startswith("scale_") and c.split("_")[-1].isdigit()]
    class_summary = (
        lr_scale.groupby("range_class", as_index=False)
        .agg(
            n_lr=("lr_id", "nunique"),
            expected_scale_mean=("expected_scale", "mean"),
            expected_scale_sem=("expected_scale", sem),
            local_weight_mean=("local_weight", "mean"),
            broad_weight_mean=("broad_weight", "mean"),
            **{c: (c, "mean") for c in scale_cols},
        )
    )
    class_summary["range_class"] = pd.Categorical(class_summary["range_class"], RANGE_ORDER, ordered=True)
    class_summary = class_summary.sort_values("range_class")
    lr_scale.to_csv(SOURCE / "lr_wavelet_scale_gate.csv", index=False)
    class_summary.to_csv(SOURCE / "class_wavelet_scale_gate.csv", index=False)
    metrics.to_csv(SOURCE / "metrics.csv", index=False)
    with open(OUT / "wavelet_scale_gate_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "purpose": "LR-specific wavelet scale gate diagnostic.",
                "model": "spatialself_wavelet_adaptive",
                "epochs": args.epochs,
                "scale_alignment_weight": args.scale_alignment_weight,
                "claim_boundary": "This diagnostic tests whether the LR identity gate separates spatial scales in the adaptive wavelet model.",
                "source_data": {
                    "lr_wavelet_scale_gate": "source_data/lr_wavelet_scale_gate.csv",
                    "class_wavelet_scale_gate": "source_data/class_wavelet_scale_gate.csv",
                    "metrics": "source_data/metrics.csv",
                },
            },
            fh,
            indent=2,
        )
    return {"lr_scale": lr_scale, "class_summary": class_summary, "metrics": metrics}


def plot(data: dict[str, pd.DataFrame]) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.15), gridspec_kw={"width_ratios": [1.2, 1.0, 0.95]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22, wspace=0.55)
    draw_heatmap(axes[0], data["class_summary"])
    draw_expected(axes[1], data["lr_scale"])
    draw_local_broad(axes[2], data["class_summary"])
    stem = FIGDIR / "fig5_wavelet_scale_gate"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return stem


def draw_heatmap(ax, class_summary: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.15, y=1.04)
    scale_cols = [c for c in class_summary.columns if c.startswith("scale_") and c.split("_")[-1].isdigit()]
    mat = class_summary.set_index("range_class").reindex(RANGE_ORDER)[scale_cols].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("scale_gate", ["#F7F4EC", "#E6C06A", "#2F78A8"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(0.25, float(np.nanmax(mat))))
    ax.set_yticks(np.arange(len(RANGE_ORDER)))
    ax.set_yticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.5)
    ax.set_xticks(np.arange(len(scale_cols)))
    ax.set_xticklabels(SCALE_LABEL[: len(scale_cols)], rotation=35, ha="right", fontsize=6.0)
    ax.set_title("LR-specific wavelet scale gate", fontsize=8.0, fontweight="bold", pad=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(scale_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RANGE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)


def draw_expected(ax, lr_scale: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.18, y=1.04)
    rng = np.random.default_rng(4)
    for i, rc in enumerate(RANGE_ORDER):
        sub = lr_scale[lr_scale["range_class"].eq(rc)]
        x = np.full(len(sub), i, dtype=float) + rng.normal(0, 0.055, len(sub))
        ax.scatter(x, sub["expected_scale"], s=13, color=COL[rc], alpha=0.55, edgecolors="white", linewidths=0.25)
        med = float(sub["expected_scale"].median())
        ax.plot([i - 0.22, i + 0.22], [med, med], color=COL["ink"], lw=1.2)
    ax.set_xticks(np.arange(len(RANGE_ORDER)))
    ax.set_xticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], rotation=25, ha="right", fontsize=6.2)
    ax.set_ylabel("expected wavelet scale", fontsize=6.8)
    ax.set_title("LR-level scale preference", fontsize=8.0, fontweight="bold", pad=3)
    ax.set_xlim(-0.5, len(RANGE_ORDER) - 0.5)
    ax.set_ylim(-0.05, 5.05)
    clean_axis(ax, grid_y=True)


def draw_local_broad(ax, class_summary: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.17, y=1.04)
    sub = class_summary.set_index("range_class").reindex(RANGE_ORDER)
    y = np.arange(len(RANGE_ORDER))
    ax.barh(y, sub["local_weight_mean"], height=0.52, color="#C8C1B8", edgecolor="white", linewidth=0.4, label="local")
    ax.barh(
        y,
        sub["broad_weight_mean"],
        left=sub["local_weight_mean"],
        height=0.52,
        color=COL["ours"],
        edgecolor="white",
        linewidth=0.4,
        label="broad",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mean gate weight", fontsize=6.8)
    ax.set_title("Local vs broad scales", fontsize=8.0, fontweight="bold", pad=3)
    clean_axis(ax, grid_y=False)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.7, handlelength=1.1, borderpad=0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lymph_scdesign3_200lr", "melanoma_scdesign3_200lr"], choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-alignment-weight", type=float, default=0.4)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for dataset_id in args.datasets:
        print(f"Training wavelet scale gate diagnostic on {dataset_id} ({args.epochs} epochs)")
        parts.append(train_dataset(dataset_id, DATASETS[dataset_id], args))
    data = write_outputs(parts, args)
    stem = plot(data)
    print(f"Wrote {stem}.svg/.pdf/.png/.tiff")
    print(f"Source data: {SOURCE}")


if __name__ == "__main__":
    main()
