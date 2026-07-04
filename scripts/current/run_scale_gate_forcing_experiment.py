#!/usr/bin/env python
"""Diagnostic experiment: force SpatialSelf wavelet scale gate to carry LR scale.

This is intentionally separate from the frozen baseline. It asks which training
signals make the LR-specific scale gate non-collapsed and mechanism-aligned.
"""

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
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import run_spatialself as rs
from src.spitialself.evaluation import evaluate_predictions_auto
from src.spitialself.model_factory import build_model
from src.spitialself.models import GraphWaveletPyramidEncoder, LREncoder


BENCH_200 = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = PROJECT_ROOT / "outputs/current_fig5_lr_scale_mechanism/scale_gate_forcing_experiment"
SOURCE = OUT / "source_data"
FIGDIR = OUT / "figures"

DATASETS = {
    "lymph_scdesign3_200lr": BENCH_200 / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00",
    "melanoma_scdesign3_200lr": BENCH_200 / "datasets/reference_melanoma_scdesign3_200lr_replicate_00",
}
VARIANTS = [
    "current_response_align",
    "oracle_class_target",
    "pseudo_distance_target",
    "pseudo_distance_diverse",
    "pseudo_profile_target",
    "pseudo_profile_diverse",
    "pseudo_profile_responsibility",
    "pseudo_hybrid_diverse",
    "pseudo_rankdist_diverse",
    "pseudo_ranklocal_diverse",
    "mixture_distance_responsibility",
    "mixture_hybrid_responsibility",
    "mixture_rankdist_responsibility",
    "mixture_ranklocal_responsibility",
    "mixture_profile_diverse",
    "mixture_profile_responsibility",
]
VARIANT_LABEL = {
    "current_response_align": "current",
    "oracle_class_target": "oracle class",
    "pseudo_distance_target": "pseudo distance",
    "pseudo_distance_diverse": "pseudo + diverse",
    "pseudo_profile_target": "profile",
    "pseudo_profile_diverse": "profile + diverse",
    "pseudo_profile_responsibility": "profile + resp.",
    "pseudo_hybrid_diverse": "hybrid + diverse",
    "pseudo_rankdist_diverse": "rankdist + diverse",
    "pseudo_ranklocal_diverse": "ranklocal + diverse",
    "mixture_distance_responsibility": "mixture distance",
    "mixture_hybrid_responsibility": "mixture hybrid",
    "mixture_rankdist_responsibility": "mixture rankdist",
    "mixture_ranklocal_responsibility": "mixture ranklocal",
    "mixture_profile_diverse": "mixture + diverse",
    "mixture_profile_responsibility": "mixture + resp.",
}
RANGE_ORDER = ["contact", "short", "medium", "long_niche"]
RANGE_LABEL = {
    "contact": "Contact",
    "short": "Short",
    "medium": "Medium",
    "long_niche": "Long niche",
}
SCALE_LABEL = ["raw", "d1", "d2", "d4", "d8", "smooth"]
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.55,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
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


class ScaleMixtureWaveletScorer(nn.Module):
    """Diagnostic scorer where LR scale gate directly mixes per-scale scores."""

    def __init__(
        self,
        h_dim: int,
        lr_dim: int,
        n_scales: int,
        distance_scale: float,
        edge_feature_dim: int,
        hidden: int = 96,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.edge_feature_dim = int(edge_feature_dim)
        self.n_scales = int(n_scales)
        self.scale_gate = nn.Sequential(
            nn.Linear(lr_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_scales),
        )
        in_dim = h_dim * 4 + lr_dim + 1 + self.edge_feature_dim
        self.scale_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(h_dim * 2 + lr_dim + 1 + self.edge_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, scale_bank, src_idx, dst_idx, dist, e_lr, edge_features=None):
        scale_weights = torch.softmax(self.scale_gate(e_lr), dim=-1)
        return self.forward_with_scale_weights(scale_bank, src_idx, dst_idx, dist, e_lr, scale_weights, edge_features)

    def forward_with_scale_weights(self, scale_bank, src_idx, dst_idx, dist, e_lr, scale_weights, edge_features=None):
        src_scales = scale_bank[src_idx]
        dst_scales = scale_bank[dst_idx]
        if edge_features is None:
            edge_features = scale_bank.new_zeros((src_idx.numel(), self.edge_feature_dim))
        dist_scaled = dist / self.distance_scale

        per_scale_logits = []
        for scale_i in range(self.n_scales):
            h_src = src_scales[:, scale_i, :]
            h_dst = dst_scales[:, scale_i, :]
            feat = torch.cat(
                [
                    h_src,
                    h_dst,
                    torch.abs(h_src - h_dst),
                    h_src * h_dst,
                    e_lr,
                    dist_scaled,
                    edge_features,
                ],
                dim=-1,
            )
            per_scale_logits.append(self.scale_mlp(feat).squeeze(-1))
        per_scale = torch.stack(per_scale_logits, dim=1)

        raw_src = src_scales[:, 0, :]
        raw_dst = dst_scales[:, 0, :]
        bias_feat = torch.cat([raw_src, raw_dst, e_lr, dist_scaled, edge_features], dim=-1)
        bias = 0.15 * self.bias_mlp(bias_feat).squeeze(-1)
        return torch.sum(per_scale * scale_weights, dim=1) + bias

    def scale_distribution(self, e_lr):
        return torch.softmax(self.scale_gate(e_lr), dim=-1)


class ScaleMixtureWaveletModel(nn.Module):
    """Temporary diagnostic model; not part of the frozen SpatialSelf baseline."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        lr_dim: int,
        n_lr: int,
        diffusion_steps=(1, 2, 4, 8),
        dropout: float = 0.2,
        distance_scale: float = 250.0,
        edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.node_encoder = GraphWaveletPyramidEncoder(
            in_dim=in_dim,
            hidden_dim=h_dim,
            out_dim=h_dim,
            diffusion_steps=diffusion_steps,
            dropout=dropout,
            undirected=True,
        )
        self.lr_encoder = LREncoder(n_lr, lr_dim, dropout=0.1, mode="learned")
        self.edge_scorer = ScaleMixtureWaveletScorer(
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_scales=self.node_encoder.n_scales,
            distance_scale=distance_scale,
            edge_feature_dim=edge_feature_dim,
            dropout=dropout,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h, src_idx, dst_idx, dists, e_lr, edge_features)

    def lr_scale_distribution(self, lr_ids, device=None):
        if device is None:
            device = self.lr_encoder.emb.weight.device
        e_lr = self._encode_lr(lr_ids, device)
        return self.edge_scorer.scale_distribution(e_lr)

    def _encode_lr(self, lr_ids, device):
        return self.lr_encoder(lr_ids.to(device))


def _build_diagnostic_model(
    variant: str,
    *,
    in_dim: int,
    h_dim: int,
    lr_dim: int,
    n_lr: int,
    distance_scale: float,
    edge_feature_dim: int,
):
    if variant.startswith("mixture_"):
        return ScaleMixtureWaveletModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            diffusion_steps=(1, 2, 4, 8),
            dropout=0.2,
            distance_scale=distance_scale,
            edge_feature_dim=edge_feature_dim,
        )
    return build_model(
        "spatialself_wavelet_adaptive",
        in_dim=in_dim,
        h_dim=h_dim,
        lr_dim=lr_dim,
        n_lr=n_lr,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        distance_scale=distance_scale,
        use_lr_embedding=True,
        lr_encoding_mode="learned",
        use_distance=True,
        edge_feature_dim=edge_feature_dim,
    )


def train_one(dataset_id: str, replicate_dir: Path, variant: str, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
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
    model = _build_diagnostic_model(
        variant,
        in_dim=data.x.shape[1],
        h_dim=64,
        lr_dim=24,
        n_lr=len(lr_pairs),
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        edge_feature_dim=int(tensors["edge_features"].shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_idx = tensors["train_idx"]
    labels = tensors["pseudo_label"][train_idx]
    sample_weight = tensors["sample_weight"][train_idx]
    aux_target = _scale_target(variant, lr_pairs, lr_design, candidates, n_scales=model.node_encoder.n_scales)

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
        loss = loss + _guard_losses(model, h, tensors)

        lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
        pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
        if variant == "current_response_align":
            target_scale = rs._wavelet_response_scale_targets(
                h,
                tensors["src_idx"],
                tensors["dst_idx"],
                tensors["lr_idx"],
                tensors["scale_alignment_weight"],
                n_lr=len(lr_pairs),
            )
            if target_scale is not None:
                loss = loss + args.response_alignment_weight * _kl(pred_scale, target_scale)
        else:
            target = aux_target.to(device=h.device, dtype=pred_scale.dtype)
            loss = loss + _scale_target_weight(variant, args) * _kl(pred_scale, target)
            if variant in {
                "pseudo_distance_diverse",
                "pseudo_profile_diverse",
                "pseudo_profile_responsibility",
                "pseudo_hybrid_diverse",
                "pseudo_rankdist_diverse",
                "pseudo_ranklocal_diverse",
                "mixture_distance_responsibility",
                "mixture_hybrid_responsibility",
                "mixture_rankdist_responsibility",
                "mixture_ranklocal_responsibility",
                "mixture_profile_diverse",
                "mixture_profile_responsibility",
            }:
                loss = loss + _diversity_weight(variant, args) * _diversity_loss(pred_scale)
            if variant in {
                "pseudo_profile_responsibility",
                "mixture_distance_responsibility",
                "mixture_hybrid_responsibility",
                "mixture_rankdist_responsibility",
                "mixture_ranklocal_responsibility",
                "mixture_profile_responsibility",
            }:
                loss = loss + args.responsibility_weight * _scale_responsibility_loss(
                    model,
                    h,
                    tensors,
                    train_idx,
                    labels,
                    target,
                    margin=args.responsibility_margin,
                    max_edges=args.responsibility_edges,
                )
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch + 1) % 100 == 0:
            print(f"{dataset_id} {variant} epoch {epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")

    model.eval()
    with torch.no_grad():
        h = model.encode_nodes(data)
        scores = []
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

    prediction = pd.DataFrame(
        {"candidate_id": candidates["candidate_id"], "method": f"SpatialSelf {variant}", "score": np.concatenate(scores)}
    )
    metrics = evaluate_predictions_auto(candidates, prediction)
    lr_to_class = lr_design.set_index("lr_id")["range_class"].to_dict()
    rows = []
    for i, lr_id in enumerate(lr_pairs["lr_id"].astype(str)):
        row = {"dataset_id": dataset_id, "variant": variant, "lr_id": lr_id, "range_class": lr_to_class.get(lr_id, "unknown")}
        for scale_i in range(scale.shape[1]):
            row[f"scale_{scale_i}"] = float(scale[i, scale_i])
            row[f"target_scale_{scale_i}"] = float(aux_target[i, scale_i])
        row["expected_scale"] = float((scale[i] * np.arange(scale.shape[1])).sum())
        row["target_expected_scale"] = float((aux_target[i].numpy() * np.arange(scale.shape[1])).sum())
        row["local_weight"] = float(scale[i, :2].sum())
        row["broad_weight"] = float(scale[i, -2:].sum())
        rows.append(row)
    lr_scale = pd.DataFrame(rows)
    lr_scale = lr_scale[lr_scale["range_class"].isin(RANGE_ORDER)].copy()
    sep = _separation_metrics(lr_scale)
    target_sep = _separation_metrics(
        lr_scale.rename(columns={"expected_scale": "model_expected_scale", "target_expected_scale": "expected_scale"})
    )
    metrics_row = pd.DataFrame(
        [
            {
                **metrics,
                **sep,
                **{f"target_{k}": v for k, v in target_sep.items()},
                "dataset_id": dataset_id,
                "variant": variant,
                "epochs": int(args.epochs),
                "scale_target_weight": float(args.scale_target_weight),
                "profile_scale_target_weight": float(args.profile_scale_target_weight),
                "response_alignment_weight": float(args.response_alignment_weight),
                "diversity_weight": float(args.diversity_weight),
                "profile_diversity_weight": float(args.profile_diversity_weight),
                "responsibility_weight": float(args.responsibility_weight),
            }
        ]
    )
    return {"lr_scale": lr_scale, "metrics": metrics_row}


def _guard_losses(model, h, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = h.new_tensor(0.0)
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
    guard_idx = tensors["mechanism_guard_train_idx"]
    if guard_idx.numel() > 0:
        guard_logits = model.score_edges(
            h,
            tensors["src_idx"][guard_idx],
            tensors["dst_idx"][guard_idx],
            tensors["lr_idx"][guard_idx],
            tensors["distance"][guard_idx].unsqueeze(-1),
            tensors["edge_features"][guard_idx],
        )
        loss = loss + 0.03 * (torch.nn.functional.softplus(guard_logits) * tensors["mechanism_guard_weight"][guard_idx]).mean()
    return loss


def _kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.kl_div(torch.log(pred.clamp_min(1e-8)), target, reduction="batchmean")


def _diversity_loss(pred: torch.Tensor) -> torch.Tensor:
    entropy_each = -(pred * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
    mean_dist = pred.mean(dim=0)
    entropy_mean = -(mean_dist * torch.log(mean_dist.clamp_min(1e-8))).sum()
    return entropy_each - entropy_mean


def _scale_target_weight(variant: str, args: argparse.Namespace) -> float:
    if variant in {
        "pseudo_profile_target",
        "pseudo_profile_diverse",
        "pseudo_profile_responsibility",
        "pseudo_hybrid_diverse",
        "pseudo_rankdist_diverse",
        "pseudo_ranklocal_diverse",
        "mixture_hybrid_responsibility",
        "mixture_rankdist_responsibility",
        "mixture_ranklocal_responsibility",
        "mixture_profile_diverse",
        "mixture_profile_responsibility",
    }:
        return float(args.profile_scale_target_weight)
    return float(args.scale_target_weight)


def _diversity_weight(variant: str, args: argparse.Namespace) -> float:
    if variant in {
        "pseudo_profile_diverse",
        "pseudo_profile_responsibility",
        "pseudo_hybrid_diverse",
        "pseudo_rankdist_diverse",
        "pseudo_ranklocal_diverse",
        "mixture_distance_responsibility",
        "mixture_hybrid_responsibility",
        "mixture_rankdist_responsibility",
        "mixture_ranklocal_responsibility",
        "mixture_profile_diverse",
        "mixture_profile_responsibility",
    }:
        return float(args.profile_diversity_weight)
    return float(args.diversity_weight)


def _scale_responsibility_loss(
    model,
    h: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    target_scale: torch.Tensor,
    *,
    margin: float,
    max_edges: int,
) -> torch.Tensor:
    """Force matched inferred scales to score above mismatched scales."""

    pos = train_idx[labels > 0.5]
    if pos.numel() == 0:
        return h.new_tensor(0.0)
    if pos.numel() > int(max_edges):
        keep = torch.randperm(pos.numel(), device=pos.device)[: int(max_edges)]
        pos = pos[keep]

    lr_idx = tensors["lr_idx"][pos]
    wrong_scale = torch.flip(target_scale, dims=[1])
    good_logits = _score_edges_with_scale_weights(model, h, tensors, pos, lr_idx, target_scale)
    bad_logits = _score_edges_with_scale_weights(model, h, tensors, pos, lr_idx, wrong_scale)
    return torch.nn.functional.softplus(float(margin) - (good_logits - bad_logits)).mean()


def _score_edges_with_scale_weights(
    model,
    scale_bank: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    idx: torch.Tensor,
    lr_idx: torch.Tensor,
    scale_weights_by_lr: torch.Tensor,
) -> torch.Tensor:
    scorer = model.edge_scorer
    src_idx = tensors["src_idx"][idx]
    dst_idx = tensors["dst_idx"][idx]
    dist = tensors["distance"][idx].unsqueeze(-1)
    edge_features = tensors["edge_features"][idx]
    e_lr = model._encode_lr(lr_idx, scale_bank.device)
    scale_weights = scale_weights_by_lr[lr_idx].to(device=scale_bank.device, dtype=scale_bank.dtype)
    if hasattr(scorer, "forward_with_scale_weights"):
        return scorer.forward_with_scale_weights(scale_bank, src_idx, dst_idx, dist, e_lr, scale_weights, edge_features)

    src_scales = scale_bank[src_idx]
    dst_scales = scale_bank[dst_idx]
    h_src = torch.sum(src_scales * scale_weights.unsqueeze(-1), dim=1)
    h_dst = torch.sum(dst_scales * scale_weights.unsqueeze(-1), dim=1)

    parts = [h_src, h_dst, e_lr]
    if scorer.use_distance:
        parts.append(dist / scorer.distance_scale)
    if scorer.edge_feature_dim:
        parts.append(edge_features)
    base_input = torch.cat(parts, dim=-1)

    context_parts = [h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, e_lr]
    if scorer.use_distance:
        context_parts.append(dist / scorer.distance_scale)
    if scorer.edge_feature_dim:
        context_parts.append(edge_features)
    context_input = torch.cat(context_parts, dim=-1)

    hidden = scorer.base_mlp(base_input)
    gate = scorer.context_gate(context_input)
    return scorer.out(hidden * gate).squeeze(-1)


def _scale_target(
    variant: str,
    lr_pairs: pd.DataFrame,
    lr_design: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    n_scales: int,
) -> torch.Tensor:
    if variant == "oracle_class_target":
        class_map = lr_design.set_index("lr_id")["range_class"].to_dict()
        target = []
        for lr_id in lr_pairs["lr_id"].astype(str):
            rc = class_map.get(lr_id, "medium")
            target.append(_class_scale_distribution(rc, n_scales))
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    if variant in {"pseudo_distance_target", "pseudo_distance_diverse", "mixture_distance_responsibility"}:
        target_by_lr = _pseudo_distance_scale_targets(candidates, n_scales=n_scales)
        target = [target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / n_scales)) for lr_id in lr_pairs["lr_id"].astype(str)]
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    if variant in {
        "pseudo_profile_target",
        "pseudo_profile_diverse",
        "pseudo_profile_responsibility",
        "mixture_profile_diverse",
        "mixture_profile_responsibility",
    }:
        target_by_lr = _pseudo_profile_scale_targets(candidates, n_scales=n_scales)
        target = [target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / n_scales)) for lr_id in lr_pairs["lr_id"].astype(str)]
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    if variant in {"pseudo_hybrid_diverse", "mixture_hybrid_responsibility"}:
        target_by_lr = _pseudo_hybrid_scale_targets(candidates, n_scales=n_scales)
        target = [target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / n_scales)) for lr_id in lr_pairs["lr_id"].astype(str)]
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    if variant in {"pseudo_rankdist_diverse", "mixture_rankdist_responsibility"}:
        target_by_lr = _pseudo_rank_distance_scale_targets(candidates, n_scales=n_scales)
        target = [target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / n_scales)) for lr_id in lr_pairs["lr_id"].astype(str)]
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    if variant in {"pseudo_ranklocal_diverse", "mixture_ranklocal_responsibility"}:
        target_by_lr = _pseudo_rank_local_scale_targets(candidates, n_scales=n_scales)
        target = [target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / n_scales)) for lr_id in lr_pairs["lr_id"].astype(str)]
        return torch.as_tensor(np.vstack(target), dtype=torch.float32)
    return torch.full((len(lr_pairs), n_scales), 1.0 / float(n_scales), dtype=torch.float32)


def _class_scale_distribution(range_class: str, n_scales: int) -> np.ndarray:
    templates = {
        "contact": np.array([0.42, 0.34, 0.14, 0.06, 0.03, 0.01]),
        "short": np.array([0.12, 0.32, 0.28, 0.16, 0.08, 0.04]),
        "medium": np.array([0.05, 0.11, 0.25, 0.30, 0.19, 0.10]),
        "long_niche": np.array([0.02, 0.04, 0.08, 0.17, 0.30, 0.39]),
    }
    v = templates.get(str(range_class), templates["medium"]).astype(float)
    if len(v) != n_scales:
        x_old = np.linspace(0, 1, len(v))
        x_new = np.linspace(0, 1, n_scales)
        v = np.interp(x_new, x_old, v)
    return v / v.sum()


def _pseudo_distance_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = ligand * receptor
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    centers = np.linspace(0.0, 1.0, n_scales)
    width = max(1.0 / max(n_scales - 1, 1), 0.18)
    out = {}
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy()
        e = expr[idx]
        d = dist[idx]
        if len(idx) < 5 or np.nanmax(e) <= 0:
            out[str(lr_id)] = np.full(n_scales, 1.0 / n_scales)
            continue
        keep = e >= np.quantile(e, 0.80)
        if int(keep.sum()) < 4:
            keep = e >= np.quantile(e, 0.60)
        weighted_d = np.average(d[keep] / d95, weights=np.maximum(e[keep], 1e-6))
        target = np.exp(-0.5 * ((centers - weighted_d) / width) ** 2)
        target = target + 0.02
        out[str(lr_id)] = target / target.sum()
    return out


def _pseudo_rank_distance_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    centers = _expression_distance_centers(candidates, quantile=0.60, power=1.0)
    lr_ids = list(centers)
    values = np.array([centers[lr_id] for lr_id in lr_ids], dtype=float)
    order = np.argsort(np.argsort(values))
    rank_centers = order / max(len(order) - 1, 1)
    scale_centers = np.linspace(0.0, 1.0, n_scales)
    out: dict[str, np.ndarray] = {}
    for lr_id, center in zip(lr_ids, rank_centers, strict=False):
        target = np.exp(-0.5 * ((scale_centers - center) / 0.16) ** 2) + 0.02
        out[str(lr_id)] = target / target.sum()
    return out


def _pseudo_rank_local_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    centers = _expression_local_distance_centers(candidates, quantile=0.60, local_fraction=0.35)
    lr_ids = list(centers)
    values = np.array([centers[lr_id] for lr_id in lr_ids], dtype=float)
    order = np.argsort(np.argsort(values))
    rank_centers = order / max(len(order) - 1, 1)
    scale_centers = np.linspace(0.0, 1.0, n_scales)
    out: dict[str, np.ndarray] = {}
    for lr_id, center in zip(lr_ids, rank_centers, strict=False):
        target = np.exp(-0.5 * ((scale_centers - center) / 0.16) ** 2) + 0.02
        out[str(lr_id)] = target / target.sum()
    return out


def _expression_distance_centers(candidates: pd.DataFrame, *, quantile: float, power: float) -> dict[str, float]:
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = np.nan_to_num(ligand * receptor, nan=0.0, posinf=0.0, neginf=0.0)
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    out: dict[str, float] = {}
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy()
        e = expr[idx]
        d = np.clip(dist[idx] / d95, 0.0, 1.0)
        if len(idx) < 5 or np.nanmax(e) <= 0:
            out[str(lr_id)] = 0.5
            continue
        q = np.quantile(e, quantile)
        weight = np.clip(e - q, 0.0, None)
        if weight.sum() <= 1e-8:
            keep = e >= np.quantile(e, max(0.55, quantile - 0.2))
            weight = np.where(keep, np.maximum(e, 1e-6), 0.0)
        weight = np.power(np.maximum(weight, 1e-8), power)
        out[str(lr_id)] = float(np.average(d, weights=weight))
    return out


def _expression_local_distance_centers(candidates: pd.DataFrame, *, quantile: float, local_fraction: float) -> dict[str, float]:
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = np.nan_to_num(ligand * receptor, nan=0.0, posinf=0.0, neginf=0.0)
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    out: dict[str, float] = {}
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy()
        e = expr[idx]
        d = np.clip(dist[idx] / d95, 0.0, 1.0)
        if len(idx) < 5 or np.nanmax(e) <= 0:
            out[str(lr_id)] = 0.5
            continue
        keep = e >= np.quantile(e, quantile)
        if keep.sum() < 4:
            keep = e >= np.quantile(e, max(0.55, quantile - 0.2))
        local_d = d[keep]
        local_w = np.maximum(e[keep], 1e-6)
        cutoff = np.quantile(local_d, local_fraction)
        local_keep = local_d <= cutoff
        if local_keep.sum() == 0:
            out[str(lr_id)] = float(np.average(local_d, weights=local_w))
        else:
            out[str(lr_id)] = float(np.average(local_d[local_keep], weights=local_w[local_keep]))
    return out


def _pseudo_profile_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    """Infer LR scale targets from within-LR expression enrichment by distance.

    This avoids benchmark labels. For each LR, high-expression candidate edges
    are compared against that LR's own distance background; the enriched
    distance bins become the scale target.
    """

    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = np.nan_to_num(ligand * receptor, nan=0.0, posinf=0.0, neginf=0.0)
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    centers = _scale_distance_centers(n_scales)
    width = max(0.11, 0.85 / max(n_scales, 2))
    out: dict[str, np.ndarray] = {}

    for lr_id, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy()
        e = expr[idx]
        d = np.clip(dist[idx] / d95, 0.0, 1.0)
        if len(idx) < 8 or np.nanmax(e) <= 0:
            out[str(lr_id)] = np.full(n_scales, 1.0 / n_scales)
            continue

        q = np.quantile(e, 0.72)
        signal_weight = np.clip(e - q, 0.0, None)
        if signal_weight.sum() <= 1e-8:
            q = np.quantile(e, 0.55)
            signal_weight = np.clip(e - q, 0.0, None)
        signal_weight = np.power(signal_weight, 1.35)

        kernel = np.exp(-0.5 * ((d[:, None] - centers[None, :]) / width) ** 2)
        background = kernel.sum(axis=0) + 1e-6
        signal = (kernel * signal_weight[:, None]).sum(axis=0) + 1e-6
        enrichment = signal / background
        enrichment = enrichment / max(float(np.nanmean(enrichment)), 1e-6)
        logits = np.log(enrichment + 1e-6)
        logits = logits - np.nanmax(logits)
        target = np.exp(logits / 0.62) + 0.025
        out[str(lr_id)] = target / target.sum()
    return out


def _pseudo_hybrid_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    distance_targets = _pseudo_distance_scale_targets(candidates, n_scales=n_scales)
    profile_targets = _pseudo_profile_scale_targets(candidates, n_scales=n_scales)
    out: dict[str, np.ndarray] = {}
    scale_index = np.arange(n_scales, dtype=float)
    for lr_id in sorted(set(distance_targets) | set(profile_targets)):
        d = distance_targets.get(lr_id, np.full(n_scales, 1.0 / n_scales)).astype(float)
        p = profile_targets.get(lr_id, np.full(n_scales, 1.0 / n_scales)).astype(float)
        d = d / d.sum()
        p = p / p.sum()
        d_mean = float((d * scale_index).sum())
        p_mean = float((p * scale_index).sum())
        agreement = float(np.exp(-((d_mean - p_mean) ** 2) / 1.15))
        profile_weight = 0.20 + 0.45 * agreement
        target = (1.0 - profile_weight) * d + profile_weight * p
        target = np.power(np.clip(target, 1e-8, None), 0.85)
        out[str(lr_id)] = target / target.sum()
    return out


def _scale_distance_centers(n_scales: int) -> np.ndarray:
    if n_scales == 6:
        return np.array([0.015, 0.075, 0.18, 0.36, 0.62, 0.88], dtype=float)
    return np.linspace(0.0, 1.0, n_scales)


def _separation_metrics(lr_scale: pd.DataFrame) -> dict[str, float]:
    df = lr_scale.copy()
    df["class_index"] = df["range_class"].map({r: i for i, r in enumerate(RANGE_ORDER)}).astype(float)
    y = df["expected_scale"].to_numpy(float)
    x = df["class_index"].to_numpy(float)
    corr = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(df) > 2 else np.nan
    overall = float(np.nanmean(y))
    ss_total = float(np.nansum((y - overall) ** 2))
    ss_between = 0.0
    for _, group in df.groupby("range_class"):
        yy = group["expected_scale"].to_numpy(float)
        ss_between += len(yy) * (float(np.nanmean(yy)) - overall) ** 2
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    means = df.groupby("range_class")["expected_scale"].mean()
    long_minus_contact = float(means.get("long_niche", np.nan) - means.get("contact", np.nan))
    return {"scale_spearman_range_order": corr, "scale_eta2": eta2, "long_minus_contact_expected_scale": long_minus_contact}


def sem(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def aggregate(parts: list[dict[str, pd.DataFrame]], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    lr_scale = pd.concat([p["lr_scale"] for p in parts], ignore_index=True)
    metrics = pd.concat([p["metrics"] for p in parts], ignore_index=True)
    scale_cols = [c for c in lr_scale.columns if c.startswith("scale_") and c.split("_")[-1].isdigit()]
    class_summary = (
        lr_scale.groupby(["dataset_id", "variant", "range_class"], as_index=False)
        .agg(
            n_lr=("lr_id", "nunique"),
            expected_scale_mean=("expected_scale", "mean"),
            expected_scale_sem=("expected_scale", sem),
            local_weight_mean=("local_weight", "mean"),
            broad_weight_mean=("broad_weight", "mean"),
            **{c: (c, "mean") for c in scale_cols},
        )
    )
    overall_class = (
        class_summary.groupby(["variant", "range_class"], as_index=False)
        .agg(
            expected_scale_mean=("expected_scale_mean", "mean"),
            local_weight_mean=("local_weight_mean", "mean"),
            broad_weight_mean=("broad_weight_mean", "mean"),
            **{c: (c, "mean") for c in scale_cols},
        )
    )
    lr_scale.to_csv(SOURCE / "scale_gate_forcing_lr_scale.csv", index=False)
    class_summary.to_csv(SOURCE / "scale_gate_forcing_class_summary.csv", index=False)
    overall_class.to_csv(SOURCE / "scale_gate_forcing_overall_class_summary.csv", index=False)
    metrics.to_csv(SOURCE / "scale_gate_forcing_metrics.csv", index=False)
    with open(OUT / "scale_gate_forcing_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "purpose": "Temporary diagnostic only; not a frozen baseline.",
                "datasets": args.datasets,
                "variants": args.variants,
                "epochs": args.epochs,
                "source_data": {
                    "lr_scale": "source_data/scale_gate_forcing_lr_scale.csv",
                    "class_summary": "source_data/scale_gate_forcing_class_summary.csv",
                    "overall_class_summary": "source_data/scale_gate_forcing_overall_class_summary.csv",
                    "metrics": "source_data/scale_gate_forcing_metrics.csv",
                },
            },
            fh,
            indent=2,
        )
    return {"lr_scale": lr_scale, "metrics": metrics, "class_summary": class_summary, "overall_class": overall_class}


def plot(data: dict[str, pd.DataFrame]) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.15), gridspec_kw={"width_ratios": [1.2, 0.95, 1.05]})
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.24, wspace=0.55)
    draw_expected_heatmap(axes[0], data["overall_class"])
    draw_separation_tradeoff(axes[1], data["metrics"])
    draw_best_scale_heatmap(axes[2], data["overall_class"], data["metrics"])
    stem = FIGDIR / "scale_gate_forcing_experiment"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=700, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return stem


def draw_expected_heatmap(ax, overall: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.16)
    variants = [v for v in VARIANTS if v in set(overall["variant"].astype(str))]
    mat = overall.pivot(index="range_class", columns="variant", values="expected_scale_mean").reindex(RANGE_ORDER)[variants]
    cmap = LinearSegmentedColormap.from_list("expected_scale", ["#F7F4EC", "#E6C06A", "#2F78A8"])
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=cmap, vmin=0.0, vmax=5.0)
    ax.set_yticks(np.arange(len(RANGE_ORDER)))
    ax.set_yticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.4)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in variants], rotation=35, ha="right", fontsize=6.0)
    ax.set_title("Mean expected scale", fontsize=8.0, fontweight="bold", pad=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RANGE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)


def draw_separation_tradeoff(ax, metrics: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.18)
    df = metrics.groupby("variant", as_index=False).agg(scale_eta2=("scale_eta2", "mean"), aupr=("aupr", "mean"))
    for _, row in df.iterrows():
        color = COL["ours"] if row["variant"] in {"pseudo_distance_diverse", "pseudo_profile_responsibility"} else COL["muted"]
        ax.scatter(row["scale_eta2"], row["aupr"], s=44, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(row["scale_eta2"] + 0.006, row["aupr"], VARIANT_LABEL[row["variant"]], fontsize=5.8, va="center", color=COL["ink"])
    ax.set_xlabel("gate separation (eta-squared)", fontsize=6.8)
    ax.set_ylabel("AUPR", fontsize=6.8)
    ax.set_title("Separation-performance tradeoff", fontsize=8.0, fontweight="bold", pad=3)
    ax.set_xlim(-0.02, max(0.25, float(df["scale_eta2"].max()) + 0.05))
    ymin = max(0.0, float(df["aupr"].min()) - 0.05)
    ymax = min(1.0, float(df["aupr"].max()) + 0.05)
    ax.set_ylim(ymin, ymax)
    clean_axis(ax, grid_y=True)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.45)


def draw_best_scale_heatmap(ax, overall: pd.DataFrame, metrics: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.16)
    score = metrics.groupby("variant", as_index=False).agg(scale_eta2=("scale_eta2", "mean"), aupr=("aupr", "mean"))
    score["combined"] = score["scale_eta2"] + 0.15 * score["aupr"]
    best = str(score.sort_values("combined", ascending=False)["variant"].iloc[0])
    scale_cols = [c for c in overall.columns if c.startswith("scale_") and c.split("_")[-1].isdigit()]
    sub = overall[overall["variant"].eq(best)].copy()
    mat = sub.set_index("range_class").reindex(RANGE_ORDER)[scale_cols].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("scale_gate", ["#F7F4EC", "#E6C06A", "#2F78A8"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(0.35, float(np.nanmax(mat))))
    ax.set_yticks(np.arange(len(RANGE_ORDER)))
    ax.set_yticklabels([RANGE_LABEL[r] for r in RANGE_ORDER], fontsize=6.4)
    ax.set_xticks(np.arange(len(scale_cols)))
    ax.set_xticklabels(SCALE_LABEL[: len(scale_cols)], rotation=35, ha="right", fontsize=6.0)
    ax.set_title(f"Scale-gate weights ({VARIANT_LABEL[best]})", fontsize=8.0, fontweight="bold", pad=3)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lymph_scdesign3_200lr"], choices=sorted(DATASETS))
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--response-alignment-weight", type=float, default=0.4)
    parser.add_argument("--scale-target-weight", type=float, default=0.8)
    parser.add_argument("--profile-scale-target-weight", type=float, default=1.1)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--profile-diversity-weight", type=float, default=0.45)
    parser.add_argument("--responsibility-weight", type=float, default=0.35)
    parser.add_argument("--responsibility-margin", type=float, default=0.35)
    parser.add_argument("--responsibility-edges", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for dataset_id in args.datasets:
        for variant in args.variants:
            print(f"Training scale-gate forcing variant={variant} dataset={dataset_id} epochs={args.epochs}")
            parts.append(train_one(dataset_id, DATASETS[dataset_id], variant, args))
    data = aggregate(parts, args)
    stem = plot(data)
    print(f"Wrote {stem}.svg/.pdf/.png/.tiff")
    print(f"Source data: {SOURCE}")


if __name__ == "__main__":
    main()
