#!/usr/bin/env python
"""Train SpatialSelf variants on benchmark candidates and score all edges."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spitialself.evaluation import evaluate_predictions_auto
from src.spitialself.model_factory import build_model
from src.spitialself.complexes import apply_complex_expression


SCALE_AWARE_BASELINE_NAMES = {"spatialself_wavelet_scale", "wavelet_scale", "spatialself_scale"}


def _read_background_guard_targets(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or str(path) == "":
        return None
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Background guard target panel not found: {target_path}")
    target = pd.read_csv(target_path)
    if "lr_id" not in target and "lr_pair" not in target:
        raise ValueError("Background guard target panel must contain `lr_id` or `lr_pair`.")
    return target


def run_spatialself_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    k_nn: int = 12,
    diffusion_steps: tuple[int, ...] = (1, 2, 4, 8),
    h_dim: int = 64,
    lr_dim: int = 24,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    bce_loss_weight: float = 1.0,
    positive_quantile: float = 0.9,
    negative_quantile: float = 0.2,
    pseudo_label_strategy: str = "mechanism_niche_expression_distance",
    pseudo_label_flip_fraction: float = 0.0,
    soft_labels: bool = False,
    weak_positive_loss_weight: float = 1.0,
    weak_positive_expression_quantile: float = 0.7,
    train_sampling_strategy: str = "balanced_per_lr",
    focal_loss_gamma: float = 0.0,
    focal_loss_alpha: float = 0.5,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 1.0,
    feasibility_loss_weight: float = 0.0,
    hard_negative_loss_weight: float = 0.05,
    context_contrastive_loss_weight: float = 0.0,
    context_contrastive_margin: float = 0.5,
    context_contrastive_negative_mode: str = "random_receiver",
    structural_contrastive_loss_weight: float = 0.0,
    structural_contrastive_margin: float = 0.5,
    lr_identity_contrastive_loss_weight: float = 0.0,
    lr_identity_contrastive_margin: float = 0.5,
    scale_contrastive_loss_weight: float = 0.0,
    scale_contrastive_margin: float = 0.5,
    wavelet_scale_alignment_loss_weight: float = 0.0,
    scale_target_loss_weight: float | None = None,
    scale_diversity_loss_weight: float | None = None,
    scale_responsibility_loss_weight: float | None = None,
    scale_responsibility_margin: float = 0.35,
    scale_responsibility_edges: int = 2048,
    lr_scale_target_strategy: str = "ranklocal",
    control_contrastive_loss_weight: float = 0.0,
    control_contrastive_margin: float = 0.5,
    null_candidate_loss_weight: float = 0.0,
    spatial_null_contrastive_loss_weight: float = 0.0,
    spatial_null_contrastive_margin: float = 0.5,
    spatial_null_negative_mode: str = "random_receiver",
    mechanism_guard_loss_weight: float = 0.03,
    background_guard_loss_weight: float = 0.0,
    background_guard_target_path: str | Path | None = None,
    evaluation_truth_path: str | Path | None = None,
    lr_cap_positive: int = 50,
    lr_cap_negative: int = 200,
    use_lr_embedding: bool = True,
    lr_encoding_mode: str = "learned",
    use_distance: bool = True,
    use_spatial_graph: bool = True,
    use_expression_features: bool = True,
    use_cell_type_features: bool = True,
    use_niche_features: bool = True,
    use_coordinate_features: bool = True,
    use_edge_prior_features: bool = True,
    fixed_uniform_range: bool = False,
    use_context_gate: bool = True,
    n_range_basis: int = 6,
    complex_expression_mode: str = "pairwise",
    complex_min_subunit_expression: float = 0.0,
    max_train_samples: int = 12000,
    seed: int = 7,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cells = pd.read_csv(replicate_dir / "cells.csv")
    expression = pd.read_csv(replicate_dir / "expression.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    if complex_expression_mode == "geometric_mean":
        candidates = apply_complex_expression(
            expression=expression,
            candidates=candidates,
            lr_pairs=lr_pairs,
            min_subunit_expression=float(complex_min_subunit_expression),
        )
        ranges = candidates["lr_id"].map(lr_pairs.set_index("lr_id")["spatial_range"]).to_numpy(dtype=float)
        candidates["coexpression_distance_score"] = (
            np.log1p(candidates["ligand_expr"].to_numpy(dtype=float))
            * np.log1p(candidates["receptor_expr"].to_numpy(dtype=float))
            * np.exp(-candidates["distance"].to_numpy(dtype=float) / np.maximum(ranges, 1e-6))
        )
    elif complex_expression_mode != "pairwise":
        raise ValueError("complex_expression_mode must be 'pairwise' or 'geometric_mean'.")
    background_guard_targets = _read_background_guard_targets(background_guard_target_path)

    data, candidate_tensors = _build_training_data(
        cells,
        expression,
        lr_pairs,
        candidates,
        k_nn=k_nn,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        pseudo_label_strategy=pseudo_label_strategy,
        pseudo_label_flip_fraction=pseudo_label_flip_fraction,
        soft_labels=soft_labels,
        weak_positive_loss_weight=weak_positive_loss_weight,
        weak_positive_expression_quantile=weak_positive_expression_quantile,
        train_sampling_strategy=train_sampling_strategy,
        lr_cap_positive=lr_cap_positive,
        lr_cap_negative=lr_cap_negative,
        use_spatial_graph=use_spatial_graph,
        use_expression_features=use_expression_features,
        use_cell_type_features=use_cell_type_features,
        use_niche_features=use_niche_features,
        use_coordinate_features=use_coordinate_features,
        use_edge_prior_features=use_edge_prior_features,
        max_train_samples=max_train_samples,
        context_contrastive_negative_mode=context_contrastive_negative_mode,
        spatial_null_negative_mode=spatial_null_negative_mode,
        build_spatial_null_contrastive=bool(spatial_null_contrastive_loss_weight > 0),
        background_guard_targets=background_guard_targets,
        seed=seed,
    )
    device_obj = torch.device(device)
    data = data.to(device_obj)
    candidate_tensors = {
        name: value.to(device_obj) if isinstance(value, torch.Tensor) else value
        for name, value in candidate_tensors.items()
    }

    model = build_model(
        model_name,
        in_dim=data.x.shape[1],
        h_dim=h_dim,
        lr_dim=lr_dim,
        n_lr=len(lr_pairs),
        diffusion_steps=diffusion_steps,
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        use_lr_embedding=use_lr_embedding,
        lr_encoding_mode=lr_encoding_mode,
        use_distance=use_distance,
        edge_feature_dim=int(candidate_tensors["edge_features"].shape[1]),
        fixed_uniform_range=fixed_uniform_range,
        use_context_gate=use_context_gate,
        n_range_basis=n_range_basis,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_idx = candidate_tensors["train_idx"]
    labels = candidate_tensors["pseudo_label"][train_idx]
    sample_weight = candidate_tensors["sample_weight"][train_idx]
    ranking_pos_idx = candidate_tensors["ranking_pos_idx"]
    ranking_neg_idx = candidate_tensors["ranking_neg_idx"]
    ranking_loss_weight = float(ranking_loss_weight)
    bce_loss_weight = float(bce_loss_weight)
    ranking_margin = float(ranking_margin)
    feasibility_loss_weight = float(feasibility_loss_weight)
    hard_negative_loss_weight = float(hard_negative_loss_weight)
    context_contrastive_loss_weight = float(context_contrastive_loss_weight)
    context_contrastive_margin = float(context_contrastive_margin)
    structural_contrastive_loss_weight = float(structural_contrastive_loss_weight)
    structural_contrastive_margin = float(structural_contrastive_margin)
    lr_identity_contrastive_loss_weight = float(lr_identity_contrastive_loss_weight)
    lr_identity_contrastive_margin = float(lr_identity_contrastive_margin)
    scale_contrastive_loss_weight = float(scale_contrastive_loss_weight)
    scale_contrastive_margin = float(scale_contrastive_margin)
    wavelet_scale_alignment_loss_weight = float(wavelet_scale_alignment_loss_weight)
    scale_target_loss_weight = _default_scale_weight(model_name, scale_target_loss_weight, 1.1)
    scale_diversity_loss_weight = _default_scale_weight(model_name, scale_diversity_loss_weight, 0.45)
    scale_responsibility_loss_weight = _default_scale_weight(model_name, scale_responsibility_loss_weight, 0.35)
    scale_responsibility_margin = float(scale_responsibility_margin)
    scale_responsibility_edges = int(scale_responsibility_edges)
    control_contrastive_loss_weight = float(control_contrastive_loss_weight)
    control_contrastive_margin = float(control_contrastive_margin)
    null_candidate_loss_weight = float(null_candidate_loss_weight)
    spatial_null_contrastive_loss_weight = float(spatial_null_contrastive_loss_weight)
    spatial_null_contrastive_margin = float(spatial_null_contrastive_margin)
    mechanism_guard_loss_weight = float(mechanism_guard_loss_weight)
    background_guard_loss_weight = float(background_guard_loss_weight)
    scale_target = None
    if (
        hasattr(model, "lr_scale_distribution")
        and (
            scale_target_loss_weight > 0
            or scale_responsibility_loss_weight > 0
        )
    ):
        n_scales = int(getattr(model.node_encoder, "n_scales", 0))
        if n_scales > 0:
            scale_target = _lr_scale_target(
                lr_pairs,
                candidates,
                strategy=str(lr_scale_target_strategy),
                n_scales=n_scales,
            )
    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        h = model.encode_nodes(data)
        logits = model.score_edges(
            h,
            candidate_tensors["src_idx"][train_idx],
            candidate_tensors["dst_idx"][train_idx],
            candidate_tensors["lr_idx"][train_idx],
            candidate_tensors["distance"][train_idx].unsqueeze(-1),
            candidate_tensors["edge_features"][train_idx],
        )
        loss = logits.new_tensor(0.0)
        if bce_loss_weight > 0:
            loss = loss + bce_loss_weight * _weighted_bce_or_focal_loss(
                logits,
                labels,
                weight=sample_weight,
                focal_gamma=focal_loss_gamma,
                focal_alpha=focal_loss_alpha,
            )
        if ranking_loss_weight > 0:
            ranking_logits_pos = model.score_edges(
                h,
                candidate_tensors["src_idx"][ranking_pos_idx],
                candidate_tensors["dst_idx"][ranking_pos_idx],
                candidate_tensors["lr_idx"][ranking_pos_idx],
                candidate_tensors["distance"][ranking_pos_idx].unsqueeze(-1),
                candidate_tensors["edge_features"][ranking_pos_idx],
            )
            ranking_logits_neg = model.score_edges(
                h,
                candidate_tensors["src_idx"][ranking_neg_idx],
                candidate_tensors["dst_idx"][ranking_neg_idx],
                candidate_tensors["lr_idx"][ranking_neg_idx],
                candidate_tensors["distance"][ranking_neg_idx].unsqueeze(-1),
                candidate_tensors["edge_features"][ranking_neg_idx],
            )
            loss = loss + ranking_loss_weight * _indexed_pairwise_ranking_loss(
                ranking_logits_pos,
                ranking_logits_neg,
                margin=ranking_margin,
            )
        if feasibility_loss_weight > 0:
            feasibility_idx = candidate_tensors["feasibility_train_idx"]
            feasibility_logits = model.score_edges(
                h,
                candidate_tensors["src_idx"][feasibility_idx],
                candidate_tensors["dst_idx"][feasibility_idx],
                candidate_tensors["lr_idx"][feasibility_idx],
                candidate_tensors["distance"][feasibility_idx].unsqueeze(-1),
                candidate_tensors["edge_features"][feasibility_idx],
            )
            loss = loss + feasibility_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                feasibility_logits,
                candidate_tensors["feasibility_target"][feasibility_idx],
                weight=candidate_tensors["feasibility_weight"][feasibility_idx],
            )
        if hard_negative_loss_weight > 0:
            hard_neg_idx = candidate_tensors["hard_negative_train_idx"]
            if hard_neg_idx.numel() > 0:
                hard_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][hard_neg_idx],
                    candidate_tensors["dst_idx"][hard_neg_idx],
                    candidate_tensors["lr_idx"][hard_neg_idx],
                    candidate_tensors["distance"][hard_neg_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][hard_neg_idx],
                )
                loss = loss + hard_negative_loss_weight * torch.nn.functional.softplus(hard_neg_logits).mean()
        if context_contrastive_loss_weight > 0:
            context_idx = candidate_tensors["context_contrastive_idx"]
            context_neg_dst_idx = candidate_tensors["context_contrastive_neg_dst_idx"]
            if context_idx.numel() > 0:
                context_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][context_idx],
                    candidate_tensors["dst_idx"][context_idx],
                    candidate_tensors["lr_idx"][context_idx],
                    candidate_tensors["distance"][context_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][context_idx],
                )
                context_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][context_idx],
                    context_neg_dst_idx,
                    candidate_tensors["lr_idx"][context_idx],
                    candidate_tensors["context_contrastive_neg_distance"].unsqueeze(-1),
                    candidate_tensors["context_contrastive_neg_edge_features"],
                )
                loss = loss + context_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    context_pos_logits,
                    context_neg_logits,
                    margin=context_contrastive_margin,
                )
        if structural_contrastive_loss_weight > 0:
            structural_idx = candidate_tensors["structural_contrastive_idx"]
            structural_neg_dst_idx = candidate_tensors["structural_contrastive_neg_dst_idx"]
            if structural_idx.numel() > 0:
                structural_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][structural_idx],
                    candidate_tensors["dst_idx"][structural_idx],
                    candidate_tensors["lr_idx"][structural_idx],
                    candidate_tensors["distance"][structural_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][structural_idx],
                )
                structural_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][structural_idx],
                    structural_neg_dst_idx,
                    candidate_tensors["lr_idx"][structural_idx],
                    candidate_tensors["structural_contrastive_neg_distance"].unsqueeze(-1),
                    candidate_tensors["structural_contrastive_neg_edge_features"],
                )
                loss = loss + structural_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    structural_pos_logits,
                    structural_neg_logits,
                    margin=structural_contrastive_margin,
                )
        if lr_identity_contrastive_loss_weight > 0:
            lr_id_pos_idx = candidate_tensors["lr_identity_pos_idx"]
            lr_id_neg_idx = candidate_tensors["lr_identity_neg_idx"]
            if lr_id_pos_idx.numel() > 0 and lr_id_neg_idx.numel() > 0:
                lr_id_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][lr_id_pos_idx],
                    candidate_tensors["dst_idx"][lr_id_pos_idx],
                    candidate_tensors["lr_idx"][lr_id_pos_idx],
                    candidate_tensors["distance"][lr_id_pos_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][lr_id_pos_idx],
                )
                lr_id_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][lr_id_neg_idx],
                    candidate_tensors["dst_idx"][lr_id_neg_idx],
                    candidate_tensors["lr_idx"][lr_id_neg_idx],
                    candidate_tensors["distance"][lr_id_neg_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][lr_id_neg_idx],
                )
                loss = loss + lr_identity_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    lr_id_pos_logits,
                    lr_id_neg_logits,
                    margin=lr_identity_contrastive_margin,
                )
        if scale_contrastive_loss_weight > 0:
            scale_pos_idx = candidate_tensors["scale_contrastive_pos_idx"]
            scale_neg_idx = candidate_tensors["scale_contrastive_neg_idx"]
            if scale_pos_idx.numel() > 0 and scale_neg_idx.numel() > 0:
                scale_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][scale_pos_idx],
                    candidate_tensors["dst_idx"][scale_pos_idx],
                    candidate_tensors["lr_idx"][scale_pos_idx],
                    candidate_tensors["distance"][scale_pos_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][scale_pos_idx],
                )
                scale_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][scale_neg_idx],
                    candidate_tensors["dst_idx"][scale_neg_idx],
                    candidate_tensors["lr_idx"][scale_neg_idx],
                    candidate_tensors["distance"][scale_neg_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][scale_neg_idx],
                )
                loss = loss + scale_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    scale_pos_logits,
                    scale_neg_logits,
                    margin=scale_contrastive_margin,
                )
        if wavelet_scale_alignment_loss_weight > 0 and hasattr(model, "lr_scale_distribution"):
            target_scale = _wavelet_response_scale_targets(
                h,
                candidate_tensors["src_idx"],
                candidate_tensors["dst_idx"],
                candidate_tensors["lr_idx"],
                candidate_tensors["scale_alignment_weight"],
                n_lr=len(lr_pairs),
            )
            if target_scale is not None:
                lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
                pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
                loss = loss + wavelet_scale_alignment_loss_weight * torch.nn.functional.kl_div(
                    torch.log(pred_scale.clamp_min(1e-8)),
                    target_scale,
                    reduction="batchmean",
                )
        if hasattr(model, "lr_scale_distribution") and (
            scale_target_loss_weight > 0
            or scale_diversity_loss_weight > 0
            or scale_responsibility_loss_weight > 0
        ):
            lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
            pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
            target = None
            if scale_target is not None:
                target = scale_target.to(device=h.device, dtype=pred_scale.dtype)
            if target is not None and scale_target_loss_weight > 0:
                loss = loss + scale_target_loss_weight * _scale_kl(pred_scale, target)
            if scale_diversity_loss_weight > 0:
                loss = loss + scale_diversity_loss_weight * _scale_diversity_loss(pred_scale)
            if target is not None and scale_responsibility_loss_weight > 0:
                loss = loss + scale_responsibility_loss_weight * _scale_responsibility_loss(
                    model,
                    h,
                    candidate_tensors,
                    train_idx,
                    labels,
                    target,
                    margin=scale_responsibility_margin,
                    max_edges=scale_responsibility_edges,
                )
        if control_contrastive_loss_weight > 0:
            control_pos_idx = candidate_tensors["control_contrastive_pos_idx"]
            control_neg_idx = candidate_tensors["control_contrastive_neg_idx"]
            if control_pos_idx.numel() > 0 and control_neg_idx.numel() > 0:
                control_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][control_pos_idx],
                    candidate_tensors["dst_idx"][control_pos_idx],
                    candidate_tensors["lr_idx"][control_pos_idx],
                    candidate_tensors["distance"][control_pos_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][control_pos_idx],
                )
                control_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][control_neg_idx],
                    candidate_tensors["dst_idx"][control_neg_idx],
                    candidate_tensors["lr_idx"][control_neg_idx],
                    candidate_tensors["distance"][control_neg_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][control_neg_idx],
                )
                loss = loss + control_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    control_pos_logits,
                    control_neg_logits,
                    margin=control_contrastive_margin,
                )
        if null_candidate_loss_weight > 0:
            null_idx = candidate_tensors["context_contrastive_idx"]
            null_neg_dst_idx = candidate_tensors["context_contrastive_neg_dst_idx"]
            if null_idx.numel() > 0:
                null_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][null_idx],
                    null_neg_dst_idx,
                    candidate_tensors["lr_idx"][null_idx],
                    candidate_tensors["context_contrastive_neg_distance"].unsqueeze(-1),
                    candidate_tensors["context_contrastive_neg_edge_features"],
                )
                loss = loss + null_candidate_loss_weight * torch.nn.functional.softplus(null_logits).mean()
        if spatial_null_contrastive_loss_weight > 0:
            spatial_null_idx = candidate_tensors["spatial_null_contrastive_idx"]
            if spatial_null_idx.numel() > 0:
                spatial_null_pos_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][spatial_null_idx],
                    candidate_tensors["dst_idx"][spatial_null_idx],
                    candidate_tensors["lr_idx"][spatial_null_idx],
                    candidate_tensors["distance"][spatial_null_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][spatial_null_idx],
                )
                spatial_null_neg_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][spatial_null_idx],
                    candidate_tensors["spatial_null_neg_dst_idx"],
                    candidate_tensors["lr_idx"][spatial_null_idx],
                    candidate_tensors["spatial_null_neg_distance"].unsqueeze(-1),
                    candidate_tensors["spatial_null_neg_edge_features"],
                )
                loss = loss + spatial_null_contrastive_loss_weight * _indexed_pairwise_ranking_loss(
                    spatial_null_pos_logits,
                    spatial_null_neg_logits,
                    margin=spatial_null_contrastive_margin,
                )
        if mechanism_guard_loss_weight > 0:
            mechanism_guard_idx = candidate_tensors["mechanism_guard_train_idx"]
            if mechanism_guard_idx.numel() > 0:
                mechanism_guard_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][mechanism_guard_idx],
                    candidate_tensors["dst_idx"][mechanism_guard_idx],
                    candidate_tensors["lr_idx"][mechanism_guard_idx],
                    candidate_tensors["distance"][mechanism_guard_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][mechanism_guard_idx],
                )
                mechanism_guard_weight = candidate_tensors["mechanism_guard_weight"][mechanism_guard_idx]
                loss = loss + mechanism_guard_loss_weight * (
                    torch.nn.functional.softplus(mechanism_guard_logits) * mechanism_guard_weight
                ).mean()
        if background_guard_loss_weight > 0:
            background_guard_idx = candidate_tensors["background_guard_train_idx"]
            if background_guard_idx.numel() > 0:
                background_guard_logits = model.score_edges(
                    h,
                    candidate_tensors["src_idx"][background_guard_idx],
                    candidate_tensors["dst_idx"][background_guard_idx],
                    candidate_tensors["lr_idx"][background_guard_idx],
                    candidate_tensors["distance"][background_guard_idx].unsqueeze(-1),
                    candidate_tensors["edge_features"][background_guard_idx],
                )
                guard_weight = candidate_tensors["background_guard_weight"][background_guard_idx]
                loss = loss + background_guard_loss_weight * (
                    torch.nn.functional.softplus(background_guard_logits) * guard_weight
                ).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    scores = []
    batch_size = 65536
    with torch.no_grad():
        h = model.encode_nodes(data)
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            logits = model.score_edges(
                h,
                candidate_tensors["src_idx"][start:end],
                candidate_tensors["dst_idx"][start:end],
                candidate_tensors["lr_idx"][start:end],
                candidate_tensors["distance"][start:end].unsqueeze(-1),
                candidate_tensors["edge_features"][start:end],
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    score = np.concatenate(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    method = model_name
    prediction_path = output_dir / f"{replicate_dir.name}__{method}_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__{method}_metrics.json"
    prediction = pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "method": method,
            "score": score,
        }
    )
    prediction.to_csv(prediction_path, index=False)
    scale_distribution_path = ""
    if hasattr(model, "lr_scale_distribution"):
        with torch.no_grad():
            lr_ids = torch.arange(
                len(lr_pairs),
                dtype=torch.long,
                device=next(model.parameters()).device,
            )
            scale_distribution = model.lr_scale_distribution(lr_ids).detach().cpu().numpy()
        scale_distribution_path = output_dir / f"{replicate_dir.name}__{method}_lr_scale_distribution.csv"
        _build_lr_scale_output(
            lr_pairs,
            scale_distribution,
            scale_target.detach().cpu().numpy() if scale_target is not None else None,
            target_strategy=str(lr_scale_target_strategy),
        ).to_csv(scale_distribution_path, index=False)

    evaluation_candidates = candidates
    if evaluation_truth_path:
        truth_path = Path(evaluation_truth_path)
        if not truth_path.exists():
            raise FileNotFoundError(f"Evaluation truth table not found: {truth_path}")
        truth = pd.read_csv(truth_path)
        if "candidate_id" not in truth or "ground_truth" not in truth:
            raise ValueError("Evaluation truth table requires candidate_id and ground_truth columns.")
        evaluation_candidates = candidates[["candidate_id"]].merge(
            truth,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        if evaluation_candidates["ground_truth"].isna().any():
            raise ValueError("Evaluation truth is missing candidate rows.")
    metrics = evaluate_predictions_auto(evaluation_candidates, prediction)
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": method,
            "prediction_path": str(prediction_path),
            "model_name": model_name,
            "k_nn": int(k_nn),
            "diffusion_steps": ";".join(str(step) for step in diffusion_steps),
            "epochs": int(epochs),
            "bce_loss_weight": float(bce_loss_weight),
            "positive_quantile": float(positive_quantile),
            "negative_quantile": float(negative_quantile),
            "pseudo_label_strategy": pseudo_label_strategy,
            "pseudo_label_flip_fraction": float(pseudo_label_flip_fraction),
            "soft_labels": bool(soft_labels),
            "weak_positive_loss_weight": float(weak_positive_loss_weight),
            "weak_positive_expression_quantile": float(weak_positive_expression_quantile),
            "train_sampling_strategy": str(train_sampling_strategy),
            "focal_loss_gamma": float(focal_loss_gamma),
            "focal_loss_alpha": float(focal_loss_alpha),
            "ranking_loss_weight": float(ranking_loss_weight),
            "ranking_margin": float(ranking_margin),
            "feasibility_loss_weight": float(feasibility_loss_weight),
            "hard_negative_loss_weight": float(hard_negative_loss_weight),
            "context_contrastive_loss_weight": float(context_contrastive_loss_weight),
            "context_contrastive_margin": float(context_contrastive_margin),
            "context_contrastive_negative_mode": str(context_contrastive_negative_mode),
            "structural_contrastive_loss_weight": float(structural_contrastive_loss_weight),
            "structural_contrastive_margin": float(structural_contrastive_margin),
            "lr_identity_contrastive_loss_weight": float(lr_identity_contrastive_loss_weight),
            "lr_identity_contrastive_margin": float(lr_identity_contrastive_margin),
            "scale_contrastive_loss_weight": float(scale_contrastive_loss_weight),
            "scale_contrastive_margin": float(scale_contrastive_margin),
            "wavelet_scale_alignment_loss_weight": float(wavelet_scale_alignment_loss_weight),
            "scale_target_loss_weight": float(scale_target_loss_weight),
            "scale_diversity_loss_weight": float(scale_diversity_loss_weight),
            "scale_responsibility_loss_weight": float(scale_responsibility_loss_weight),
            "scale_responsibility_margin": float(scale_responsibility_margin),
            "scale_responsibility_edges": int(scale_responsibility_edges),
            "lr_scale_target_strategy": str(lr_scale_target_strategy),
            "lr_scale_distribution_path": str(scale_distribution_path) if scale_distribution_path else "",
            "control_contrastive_loss_weight": float(control_contrastive_loss_weight),
            "control_contrastive_margin": float(control_contrastive_margin),
            "null_candidate_loss_weight": float(null_candidate_loss_weight),
            "spatial_null_contrastive_loss_weight": float(spatial_null_contrastive_loss_weight),
            "spatial_null_contrastive_margin": float(spatial_null_contrastive_margin),
            "spatial_null_negative_mode": str(spatial_null_negative_mode),
            "mechanism_guard_loss_weight": float(mechanism_guard_loss_weight),
            "background_guard_loss_weight": float(background_guard_loss_weight),
            "background_guard_target_path": str(background_guard_target_path) if background_guard_target_path else "",
            "evaluation_truth_path": str(evaluation_truth_path) if evaluation_truth_path else "",
            "lr_cap_positive": int(lr_cap_positive),
            "lr_cap_negative": int(lr_cap_negative),
            "use_lr_embedding": bool(use_lr_embedding),
            "lr_encoding_mode": str(lr_encoding_mode),
            "use_distance": bool(use_distance),
            "use_spatial_graph": bool(use_spatial_graph),
            "use_expression_features": bool(use_expression_features),
            "use_cell_type_features": bool(use_cell_type_features),
            "use_niche_features": bool(use_niche_features),
            "use_coordinate_features": bool(use_coordinate_features),
            "use_edge_prior_features": bool(use_edge_prior_features),
            "fixed_uniform_range": bool(fixed_uniform_range),
            "use_context_gate": bool(use_context_gate),
            "n_range_basis": int(n_range_basis),
            "complex_expression_mode": str(complex_expression_mode),
            "complex_min_subunit_expression": float(complex_min_subunit_expression),
            "max_train_samples": int(max_train_samples),
            "device": str(device_obj),
            "n_pseudo_positive": int((labels > 0.5).sum().item()),
            "n_pseudo_negative": int((labels == 0).sum().item()),
            "n_pseudo_label_flipped": int(candidate_tensors["n_pseudo_label_flipped"].item()),
            "n_ranking_pairs": int(ranking_pos_idx.numel()),
            "n_feasibility_train": int(candidate_tensors["feasibility_train_idx"].numel()),
            "n_hard_negative_train": int(candidate_tensors["hard_negative_train_idx"].numel()),
            "n_context_contrastive_train": int(candidate_tensors["context_contrastive_idx"].numel()),
            "n_structural_contrastive_train": int(candidate_tensors["structural_contrastive_idx"].numel()),
            "n_lr_identity_contrastive_train": int(candidate_tensors["lr_identity_pos_idx"].numel()),
            "n_scale_contrastive_train": int(candidate_tensors["scale_contrastive_pos_idx"].numel()),
            "n_control_contrastive_train": int(candidate_tensors["control_contrastive_pos_idx"].numel()),
            "n_spatial_null_contrastive_train": int(candidate_tensors["spatial_null_contrastive_idx"].numel()),
            "n_mechanism_guard_train": int(candidate_tensors["mechanism_guard_train_idx"].numel()),
            "n_background_guard_train": int(candidate_tensors["background_guard_train_idx"].numel()),
            "n_background_guard_target_train": int(candidate_tensors["background_guard_target_train_idx"].numel()),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def _build_training_data(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    k_nn: int,
    positive_quantile: float,
    negative_quantile: float,
    pseudo_label_strategy: str,
    pseudo_label_flip_fraction: float,
    soft_labels: bool,
    weak_positive_loss_weight: float,
    weak_positive_expression_quantile: float,
    train_sampling_strategy: str,
    lr_cap_positive: int,
    lr_cap_negative: int,
    use_spatial_graph: bool,
    use_expression_features: bool,
    use_cell_type_features: bool,
    use_niche_features: bool,
    use_coordinate_features: bool,
    use_edge_prior_features: bool,
    max_train_samples: int,
    context_contrastive_negative_mode: str,
    spatial_null_negative_mode: str,
    build_spatial_null_contrastive: bool,
    background_guard_targets: pd.DataFrame | None,
    seed: int,
) -> tuple[Data, dict[str, torch.Tensor]]:
    candidates = _attach_lr_metadata(candidates, lr_pairs)
    expression = expression.set_index("cell_id").loc[cells["cell_id"]]
    feature_blocks = []
    if use_expression_features:
        gene_matrix = np.log1p(expression.to_numpy(dtype=np.float32))
        feature_blocks.append(_zscore(gene_matrix))
    if use_cell_type_features:
        feature_blocks.append(pd.get_dummies(cells["cell_type"], dtype=float).to_numpy(dtype=np.float32))
    if use_niche_features:
        feature_blocks.append(pd.get_dummies(cells["niche"], dtype=float).to_numpy(dtype=np.float32))
    if use_coordinate_features:
        coords = cells[["x", "y"]].to_numpy(dtype=np.float32)
        feature_blocks.append(_zscore(coords))
    if not feature_blocks:
        feature_blocks.append(np.ones((len(cells), 1), dtype=np.float32))
    x = np.concatenate(feature_blocks, axis=1)

    if use_spatial_graph:
        coords_np = cells[["x", "y"]].to_numpy(dtype=float)
        edge_index = _knn_edge_index(coords_np, k_nn=k_nn)
    else:
        coords_np = cells[["x", "y"]].to_numpy(dtype=float)
        edge_index = np.vstack([np.arange(len(cells)), np.arange(len(cells))])
    data = Data(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
    )

    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cells["cell_id"])}
    lr_to_idx = {lr_id: idx for idx, lr_id in enumerate(lr_pairs["lr_id"])}
    src_idx = candidates["sender_id"].map(cell_to_idx).to_numpy(dtype=np.int64)
    dst_idx = candidates["receiver_id"].map(cell_to_idx).to_numpy(dtype=np.int64)
    lr_idx = candidates["lr_id"].map(lr_to_idx).to_numpy(dtype=np.int64)
    if np.any(pd.isna(src_idx)) or np.any(pd.isna(dst_idx)) or np.any(pd.isna(lr_idx)):
        raise ValueError("Candidate table contains unknown cell_id or lr_id values.")
    edge_features = _edge_prior_features(candidates) if use_edge_prior_features else np.zeros((len(candidates), 0), dtype=np.float32)

    pseudo_score = _pseudo_label_score(candidates, strategy=pseudo_label_strategy)
    pos_idx, neg_idx = _pseudo_label_indices(
        candidates,
        pseudo_score,
        strategy=pseudo_label_strategy,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        lr_cap_positive=lr_cap_positive,
        lr_cap_negative=lr_cap_negative,
    )
    train_idx = _balanced_train_indices(
        candidates,
        pos_idx,
        neg_idx,
        strategy=train_sampling_strategy,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    if soft_labels:
        pseudo_label = _minmax(pseudo_score).astype(np.float32)
    else:
        pseudo_label = np.zeros(len(candidates), dtype=np.float32)
        pseudo_label[pos_idx] = 1.0
    n_pseudo_label_flipped = _flip_pseudo_labels(
        pseudo_label,
        train_idx,
        fraction=float(pseudo_label_flip_fraction),
        seed=seed,
    )

    expression_for_hard_neg = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float)) * np.log1p(
        candidates["receptor_expr"].to_numpy(dtype=float)
    )
    sample_weight = _sample_weight(
        candidates,
        pos_idx,
        expression_for_hard_neg,
        weak_positive_loss_weight=float(weak_positive_loss_weight),
        weak_positive_expression_quantile=float(weak_positive_expression_quantile),
    ).astype(np.float32)
    feasibility_target = _feasibility_target(candidates).astype(np.float32)
    feasibility_weight = _feasibility_weight(candidates, feasibility_target).astype(np.float32)
    feasibility_train_idx = _feasibility_train_indices(
        candidates,
        feasibility_target,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    scale_alignment_weight = _scale_alignment_weight(candidates, expression_for_hard_neg).astype(np.float32)
    hard_negative_train_idx = _hard_negative_train_indices(
        candidates,
        expression_for_hard_neg,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    context_contrastive_idx, context_contrastive_neg_dst_idx = _context_contrastive_indices(
        candidates,
        src_idx,
        dst_idx,
        pseudo_score,
        max_train_samples=max_train_samples,
        n_cells=len(cells),
        negative_mode=context_contrastive_negative_mode,
        seed=seed,
    )
    context_contrastive_neg_distance, context_contrastive_neg_edge_features = _context_contrastive_negative_features(
        candidates,
        cells,
        expression,
        context_contrastive_idx,
        src_idx,
        context_contrastive_neg_dst_idx,
        use_edge_prior_features=use_edge_prior_features,
        edge_feature_dim=int(edge_features.shape[1]),
    )
    structural_contrastive_idx, structural_contrastive_neg_dst_idx = _structural_contrastive_indices(
        candidates,
        src_idx,
        dst_idx,
        max_train_samples=max_train_samples,
        n_cells=len(cells),
        seed=seed,
    )
    structural_contrastive_neg_distance, structural_contrastive_neg_edge_features = _context_contrastive_negative_features(
        candidates,
        cells,
        expression,
        structural_contrastive_idx,
        src_idx,
        structural_contrastive_neg_dst_idx,
        use_edge_prior_features=use_edge_prior_features,
        edge_feature_dim=int(edge_features.shape[1]),
    )
    lr_identity_pos_idx, lr_identity_neg_idx = _lr_identity_contrastive_indices(
        candidates,
        pseudo_score,
        max_pairs=max_train_samples // 2,
        seed=seed,
    )
    scale_contrastive_pos_idx, scale_contrastive_neg_idx = _scale_contrastive_indices(
        candidates,
        max_pairs=max_train_samples // 2,
        seed=seed,
    )
    control_contrastive_pos_idx, control_contrastive_neg_idx = _control_contrastive_indices(
        candidates,
        pseudo_score,
        max_pairs=max_train_samples // 2,
        seed=seed,
    )
    spatial_null_contrastive_idx = np.array([], dtype=np.int64)
    spatial_null_neg_dst_idx = np.array([], dtype=np.int64)
    spatial_null_neg_distance = np.array([], dtype=np.float32)
    spatial_null_neg_edge_features = np.zeros((0, int(edge_features.shape[1])), dtype=np.float32)
    if build_spatial_null_contrastive:
        spatial_null_contrastive_idx, spatial_null_neg_dst_idx = _spatial_null_contrastive_indices(
            candidates,
            src_idx,
            dst_idx,
            pseudo_score,
            max_pairs=max_train_samples // 2,
            n_cells=len(cells),
            cells=cells,
            negative_mode=spatial_null_negative_mode,
            seed=seed,
        )
        spatial_null_neg_distance, spatial_null_neg_edge_features = _spatial_null_negative_features(
            candidates,
            cells,
            expression,
            spatial_null_contrastive_idx,
            src_idx,
            spatial_null_neg_dst_idx,
            use_edge_prior_features=use_edge_prior_features,
            edge_feature_dim=int(edge_features.shape[1]),
        )
    mechanism_guard_train_idx, mechanism_guard_weight = _mechanism_guard_train_indices(
        candidates,
        expression_for_hard_neg,
        pseudo_score,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    background_guard_train_idx, background_guard_weight, background_guard_target_train_idx = _background_guard_train_indices(
        candidates,
        expression_for_hard_neg,
        max_train_samples=max_train_samples,
        background_guard_targets=background_guard_targets,
        seed=seed,
    )
    ranking_pos_idx, ranking_neg_idx = _ranking_pair_indices(
        candidates,
        pseudo_score,
        pos_idx,
        neg_idx,
        max_pairs=max_train_samples // 2,
        seed=seed,
    )

    tensors = {
        "src_idx": torch.as_tensor(src_idx, dtype=torch.long),
        "dst_idx": torch.as_tensor(dst_idx, dtype=torch.long),
        "lr_idx": torch.as_tensor(lr_idx, dtype=torch.long),
        "coords": torch.as_tensor(coords_np, dtype=torch.float32),
        "distance": torch.as_tensor(candidates["distance"].to_numpy(dtype=np.float32), dtype=torch.float32),
        "edge_features": torch.as_tensor(edge_features, dtype=torch.float32),
        "pseudo_label": torch.as_tensor(pseudo_label, dtype=torch.float32),
        "n_pseudo_label_flipped": torch.as_tensor(n_pseudo_label_flipped, dtype=torch.long),
        "sample_weight": torch.as_tensor(sample_weight, dtype=torch.float32),
        "scale_alignment_weight": torch.as_tensor(scale_alignment_weight, dtype=torch.float32),
        "feasibility_target": torch.as_tensor(feasibility_target, dtype=torch.float32),
        "feasibility_weight": torch.as_tensor(feasibility_weight, dtype=torch.float32),
        "train_idx": torch.as_tensor(train_idx, dtype=torch.long),
        "feasibility_train_idx": torch.as_tensor(feasibility_train_idx, dtype=torch.long),
        "hard_negative_train_idx": torch.as_tensor(hard_negative_train_idx, dtype=torch.long),
        "context_contrastive_idx": torch.as_tensor(context_contrastive_idx, dtype=torch.long),
        "context_contrastive_neg_dst_idx": torch.as_tensor(context_contrastive_neg_dst_idx, dtype=torch.long),
        "context_contrastive_neg_distance": torch.as_tensor(
            context_contrastive_neg_distance,
            dtype=torch.float32,
        ),
        "context_contrastive_neg_edge_features": torch.as_tensor(
            context_contrastive_neg_edge_features,
            dtype=torch.float32,
        ),
        "structural_contrastive_idx": torch.as_tensor(structural_contrastive_idx, dtype=torch.long),
        "structural_contrastive_neg_dst_idx": torch.as_tensor(
            structural_contrastive_neg_dst_idx,
            dtype=torch.long,
        ),
        "structural_contrastive_neg_distance": torch.as_tensor(
            structural_contrastive_neg_distance,
            dtype=torch.float32,
        ),
        "structural_contrastive_neg_edge_features": torch.as_tensor(
            structural_contrastive_neg_edge_features,
            dtype=torch.float32,
        ),
        "ranking_pos_idx": torch.as_tensor(ranking_pos_idx, dtype=torch.long),
        "ranking_neg_idx": torch.as_tensor(ranking_neg_idx, dtype=torch.long),
        "lr_identity_pos_idx": torch.as_tensor(lr_identity_pos_idx, dtype=torch.long),
        "lr_identity_neg_idx": torch.as_tensor(lr_identity_neg_idx, dtype=torch.long),
        "scale_contrastive_pos_idx": torch.as_tensor(scale_contrastive_pos_idx, dtype=torch.long),
        "scale_contrastive_neg_idx": torch.as_tensor(scale_contrastive_neg_idx, dtype=torch.long),
        "control_contrastive_pos_idx": torch.as_tensor(control_contrastive_pos_idx, dtype=torch.long),
        "control_contrastive_neg_idx": torch.as_tensor(control_contrastive_neg_idx, dtype=torch.long),
        "spatial_null_contrastive_idx": torch.as_tensor(spatial_null_contrastive_idx, dtype=torch.long),
        "spatial_null_neg_dst_idx": torch.as_tensor(spatial_null_neg_dst_idx, dtype=torch.long),
        "spatial_null_neg_distance": torch.as_tensor(spatial_null_neg_distance, dtype=torch.float32),
        "spatial_null_neg_edge_features": torch.as_tensor(spatial_null_neg_edge_features, dtype=torch.float32),
        "mechanism_guard_train_idx": torch.as_tensor(mechanism_guard_train_idx, dtype=torch.long),
        "mechanism_guard_weight": torch.as_tensor(mechanism_guard_weight, dtype=torch.float32),
        "background_guard_train_idx": torch.as_tensor(background_guard_train_idx, dtype=torch.long),
        "background_guard_weight": torch.as_tensor(background_guard_weight, dtype=torch.float32),
        "background_guard_target_train_idx": torch.as_tensor(background_guard_target_train_idx, dtype=torch.long),
    }
    return data, tensors


def _pseudo_label_score(candidates: pd.DataFrame, *, strategy: str) -> np.ndarray:
    base_strategy = _base_pseudo_label_strategy(strategy)
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(dtype=float))
    expression = ligand * receptor
    distance = candidates["distance"].to_numpy(dtype=float)
    bandwidth = max(float(np.quantile(distance, 0.5)), 1e-6)
    distance_score = np.exp(-distance / bandwidth)

    if base_strategy in {"expression_distance", "expression_times_distance", "hard_pseudo_label"}:
        if "coexpression_distance_score" in candidates:
            score = candidates["coexpression_distance_score"].to_numpy(dtype=float)
        else:
            score = expression * distance_score
    elif base_strategy == "expression_only":
        score = expression
    elif base_strategy == "distance_only":
        score = distance_score
    elif base_strategy == "expression_plus_distance":
        score = _minmax(expression) + _minmax(distance_score)
    elif base_strategy == "observed_full":
        score = expression * distance_score * _observed_cell_type_prior(candidates) * _observed_niche_prior(candidates)
    elif base_strategy == "observed_expression_distance_celltype":
        score = expression * distance_score * _observed_cell_type_prior(candidates)
    elif base_strategy == "observed_no_expression":
        score = distance_score * _observed_cell_type_prior(candidates) * _observed_niche_prior(candidates)
    elif base_strategy == "observed_no_distance":
        score = expression * _observed_cell_type_prior(candidates) * _observed_niche_prior(candidates)
    elif base_strategy == "observed_no_niche":
        score = expression * distance_score * _observed_cell_type_prior(candidates)
    elif base_strategy == "observed_no_cell_type":
        score = expression * distance_score * _observed_niche_prior(candidates)
    elif base_strategy == "soft_pseudo_label":
        score = expression * distance_score
    elif base_strategy == "mechanism_expression_distance":
        score = expression * _mechanism_distance_prior(candidates)
    elif base_strategy == "mechanism_niche_distance":
        score = _mechanism_distance_prior(candidates) * _niche_prior(candidates)
    elif base_strategy == "mechanism_niche_expression":
        score = expression * _niche_prior(candidates)
    elif base_strategy == "mechanism_niche_expression_distance":
        score = expression * _mechanism_distance_prior(candidates) * _niche_prior(candidates)
    elif base_strategy == "scale_adaptive_expression":
        score = expression * _lr_scale_preference(candidates, expression)
    elif base_strategy == "scale_adaptive_expression_plus_distance":
        score = expression * (0.65 * _lr_scale_preference(candidates, expression) + 0.35 * distance_score)
    else:
        known = [
            "expression_distance",
            "expression_times_distance",
            "expression_only",
            "distance_only",
            "expression_plus_distance",
            "observed_full",
            "observed_expression_distance_celltype",
            "observed_no_expression",
            "observed_no_distance",
            "observed_no_niche",
            "observed_no_cell_type",
            "soft_pseudo_label",
            "hard_pseudo_label",
            "mechanism_expression_distance",
            "mechanism_niche_distance",
            "mechanism_niche_expression",
            "mechanism_niche_expression_distance",
            "lr_balanced_expression_distance",
            "lr_balanced_expression_only",
            "lr_balanced_expression_plus_distance",
            "lr_balanced_soft_pseudo_label",
            "lr_capped_mechanism_expression_distance",
            "lr_capped_mechanism_niche_expression_distance",
            "lr_prior_lr_capped_mechanism_expression_distance",
            "lr_prior_lr_capped_mechanism_niche_expression_distance",
            "lr_prior_lr_capped_expression_distance",
            "lr_prior_lr_capped_expression_plus_distance",
            "lr_capped_scale_adaptive_expression",
            "lr_balanced_scale_adaptive_expression",
            "scale_adaptive_expression",
            "scale_adaptive_expression_plus_distance",
        ]
        raise ValueError(f"Unknown pseudo_label_strategy: {strategy}. Known: {known}")
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _base_pseudo_label_strategy(strategy: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("lr_prior_", "lr_balanced_", "lr_capped_"):
            if strategy.startswith(prefix):
                strategy = strategy.removeprefix(prefix)
                changed = True
    return strategy


def _pseudo_label_indices(
    candidates: pd.DataFrame,
    pseudo_score: np.ndarray,
    *,
    strategy: str,
    positive_quantile: float,
    negative_quantile: float,
    lr_cap_positive: int = 50,
    lr_cap_negative: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy.startswith("lr_prior_"):
        pos_idx, neg_idx = _pseudo_label_indices(
            candidates,
            pseudo_score,
            strategy=strategy.removeprefix("lr_prior_"),
            positive_quantile=positive_quantile,
            negative_quantile=negative_quantile,
            lr_cap_positive=lr_cap_positive,
            lr_cap_negative=lr_cap_negative,
        )
        prior_negative = _lr_prior_negative_indices(candidates, pseudo_score, negative_quantile)
        if len(prior_negative):
            neg_idx = np.unique(np.concatenate([neg_idx, prior_negative]))
        return pos_idx, neg_idx

    if not (strategy.startswith("lr_balanced_") or strategy.startswith("lr_capped_")):
        high = np.quantile(pseudo_score, positive_quantile)
        low = np.quantile(pseudo_score, negative_quantile)
        return np.flatnonzero(pseudo_score >= high), np.flatnonzero(pseudo_score <= low)

    pos_parts = []
    neg_parts = []
    cap_per_lr = int(lr_cap_positive) if strategy.startswith("lr_capped_") else None
    neg_cap_per_lr = int(lr_cap_negative) if strategy.startswith("lr_capped_") else None
    for _, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        scores = pseudo_score[idx]
        high = np.quantile(scores, positive_quantile)
        low = np.quantile(scores, negative_quantile)
        pos = idx[scores >= high]
        neg = idx[scores <= low]
        if cap_per_lr is not None and cap_per_lr > 0 and len(pos) > cap_per_lr:
            order = np.argsort(-pseudo_score[pos])[:cap_per_lr]
            pos = pos[order]
        if neg_cap_per_lr is not None and neg_cap_per_lr > 0 and len(neg) > neg_cap_per_lr:
            order = np.argsort(pseudo_score[neg])[:neg_cap_per_lr]
            neg = neg[order]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg)
    if not pos_parts or not neg_parts:
        raise ValueError("LR-balanced pseudo-label split produced empty positive or negative set.")
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def _lr_prior_negative_indices(candidates: pd.DataFrame, pseudo_score: np.ndarray, negative_quantile: float) -> np.ndarray:
    if "false_positive_high_expression" in candidates:
        mask = candidates["false_positive_high_expression"].astype(bool).to_numpy()
    elif "true_lr" in candidates:
        mask = ~candidates["true_lr"].astype(bool).to_numpy()
    else:
        return np.array([], dtype=np.int64)
    if not mask.any():
        return np.array([], dtype=np.int64)

    parts = []
    for _, group in candidates.loc[mask].groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        scores = pseudo_score[idx]
        high = np.quantile(scores, max(negative_quantile, 0.8))
        parts.append(idx[scores >= high])
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def _edge_prior_features(candidates: pd.DataFrame) -> np.ndarray:
    expression = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float)) * np.log1p(
        candidates["receptor_expr"].to_numpy(dtype=float)
    )
    expression_rank = _rank01(expression)
    features = [
        _minmax(expression),
        expression_rank,
        _minmax(_mechanism_distance_prior(candidates)),
        _niche_prior(candidates),
    ]
    if "distance" in candidates:
        features.append(_minmax(candidates["distance"].to_numpy(dtype=float)))
    return np.stack(features, axis=1).astype(np.float32)


def _attach_lr_metadata(candidates: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach LR-level priors without overwriting candidate cell annotations."""

    candidates = candidates.copy()
    lr_index = lr_pairs.set_index("lr_id")
    for column in (
        "mechanism",
        "spatial_range",
        "active_niche",
        "control_class",
        "biological_category",
        "selected_reason",
        "source_count",
        "sender_classes",
        "receiver_classes",
    ):
        if column not in candidates.columns and column in lr_index.columns:
            candidates[column] = candidates["lr_id"].map(lr_index[column])
    return candidates


def _observed_niche_prior(candidates: pd.DataFrame) -> np.ndarray:
    """Generic local-context prior based only on observed annotations."""

    required = {"sender_niche", "receiver_niche"}
    if not required.issubset(candidates.columns):
        return np.ones(len(candidates), dtype=float)
    sender = candidates["sender_niche"].astype(str).to_numpy()
    receiver = candidates["receiver_niche"].astype(str).to_numpy()
    return np.where(sender == receiver, 1.0, 0.35).astype(float)


def _parse_annotation_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.replace(";", ",").split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    return {str(item).strip() for item in parsed if str(item).strip()}


def _observed_cell_type_prior(candidates: pd.DataFrame) -> np.ndarray:
    """Use curated sender/receiver annotations without simulator event labels."""

    required = {"sender_cell_type", "receiver_cell_type", "sender_classes", "receiver_classes"}
    if not required.issubset(candidates.columns):
        return np.ones(len(candidates), dtype=float)
    sender = candidates["sender_cell_type"].astype(str).to_numpy()
    receiver = candidates["receiver_cell_type"].astype(str).to_numpy()
    sender_allowed = candidates["sender_classes"].map(_parse_annotation_set).to_numpy()
    receiver_allowed = candidates["receiver_classes"].map(_parse_annotation_set).to_numpy()
    compatible = np.fromiter(
        (
            (not allowed_sender or sender_type in allowed_sender)
            and (not allowed_receiver or receiver_type in allowed_receiver)
            for sender_type, receiver_type, allowed_sender, allowed_receiver in zip(
                sender, receiver, sender_allowed, receiver_allowed
            )
        ),
        dtype=bool,
        count=len(candidates),
    )
    return np.where(compatible, 1.0, 0.20).astype(float)


def _flip_pseudo_labels(
    pseudo_label: np.ndarray,
    train_idx: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> int:
    """Flip a fixed fraction of training pseudo-labels for robustness checks."""

    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("pseudo_label_flip_fraction must be in [0, 1).")
    if fraction == 0.0 or len(train_idx) == 0:
        return 0
    n_flip = int(round(len(train_idx) * fraction))
    if n_flip == 0:
        return 0
    rng = np.random.default_rng(seed + 307)
    selected = rng.choice(np.asarray(train_idx, dtype=np.int64), size=n_flip, replace=False)
    pseudo_label[selected] = 1.0 - pseudo_label[selected]
    return int(n_flip)


def _mechanism_distance_prior(candidates: pd.DataFrame) -> np.ndarray:
    distance = candidates["distance"].to_numpy(dtype=float)
    fallback_bandwidth = max(float(np.quantile(distance, 0.5)), 1e-6)
    if "spatial_range" in candidates:
        spatial_range = candidates["spatial_range"].to_numpy(dtype=float)
        spatial_range = np.nan_to_num(spatial_range, nan=fallback_bandwidth, posinf=fallback_bandwidth)
        spatial_range = np.maximum(spatial_range, 1e-6)
    else:
        spatial_range = np.full(len(candidates), fallback_bandwidth, dtype=float)

    prior = np.exp(-distance / fallback_bandwidth)
    if "mechanism" not in candidates:
        return prior

    mechanism = candidates["mechanism"].astype(str).to_numpy()
    contact = mechanism == "juxtacrine_contact"
    short = mechanism == "short_range_paracrine"
    medium = (mechanism == "short_medium_paracrine") | (mechanism == "medium_range_paracrine")
    long = mechanism == "long_range_chemokine_niche"

    contact_width = np.maximum(0.38 * spatial_range[contact], 1e-6)
    prior[contact] = np.exp(-0.5 * (distance[contact] / contact_width) ** 2)
    prior[contact & (distance > spatial_range)] *= 0.10
    prior[short] = np.exp(-distance[short] / np.maximum(0.55 * spatial_range[short], 1e-6))
    peak_medium = 0.35 * spatial_range[medium]
    width_medium = np.maximum(0.38 * spatial_range[medium], 1e-6)
    prior[medium] = 0.22 + 0.78 * np.exp(
        -((distance[medium] - peak_medium) ** 2) / (2.0 * width_medium**2)
    )

    peak = 0.62 * spatial_range[long]
    width = np.maximum(0.30 * spatial_range[long], 1e-6)
    prior[long] = 0.20 + 0.80 * np.exp(-((distance[long] - peak) ** 2) / (2.0 * width**2))
    prior[long & (distance < 0.35 * spatial_range)] *= 0.20
    return np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)


def _niche_prior(candidates: pd.DataFrame) -> np.ndarray:
    if "niche_rule_satisfied" in candidates:
        matched = candidates["niche_rule_satisfied"].astype(bool).to_numpy()
        return np.where(matched, 1.0, 0.15).astype(float)
    if not {"active_niche", "sender_niche", "receiver_niche"}.issubset(candidates.columns):
        return np.ones(len(candidates), dtype=float)
    active = candidates["active_niche"].astype(str).to_numpy()
    sender = candidates["sender_niche"].astype(str).to_numpy()
    receiver = candidates["receiver_niche"].astype(str).to_numpy()
    unrestricted = (active == "any") | (active == "") | (active == "nan")
    matched = unrestricted | ((sender == active) & (receiver == active))
    return np.where(matched, 1.0, 0.15).astype(float)


def _lr_scale_preference(candidates: pd.DataFrame, expression: np.ndarray | None = None) -> np.ndarray:
    """Estimate LR-specific distance preference from high-expression edge density.

    This is data-driven and intentionally ignores benchmark truth fields such as
    `ground_truth`, `wrong_range_false_positive`, and `range_violation_type`.
    """

    distance = candidates["distance"].to_numpy(dtype=float)
    if expression is None:
        ligand = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float))
        receptor = np.log1p(candidates["receptor_expr"].to_numpy(dtype=float))
        expression = ligand * receptor
    expression = np.asarray(expression, dtype=float)
    preference = np.zeros(len(candidates), dtype=float)
    global_bandwidth = max(float(np.quantile(distance, 0.25)), 1e-6)

    for _, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        d = distance[idx]
        e = expression[idx]
        if len(idx) < 8 or np.nanmax(d) <= np.nanmin(d):
            preference[idx] = 1.0
            continue

        expr_rank = _rank01(e)
        high_mask = expr_rank >= 0.70
        if int(high_mask.sum()) < 4:
            high_mask = expr_rank >= 0.50
        anchors = d[high_mask]
        weights = e[high_mask]
        if len(anchors) == 0:
            preference[idx] = 1.0
            continue

        # The original dense kernel is exact for ordinary data sets but grows as
        # O(n_candidates * n_high_expression) within each LR.  A fixed,
        # quantile-spaced anchor bank preserves the estimated distance-density
        # curve while bounding memory on high-resolution sections.  For the
        # small data sets used in the manuscript no subsampling is applied.
        max_density_anchors = 600
        if len(anchors) > max_density_anchors:
            anchor_order = np.argsort(anchors, kind="mergesort")
            take = np.linspace(0, len(anchor_order) - 1, max_density_anchors).astype(np.int64)
            selected = anchor_order[take]
            anchors = anchors[selected]
            weights = weights[selected]

        bandwidth = max(
            float(np.quantile(d, 0.75) - np.quantile(d, 0.25)) * 0.28,
            global_bandwidth * 0.25,
            1e-6,
        )
        kernel = np.exp(-((d[:, None] - anchors[None, :]) ** 2) / (2.0 * bandwidth**2))
        density = (kernel * (0.10 + _rank01(weights))[None, :]).sum(axis=1)
        background = _distance_background_density(d, bandwidth)
        local_preference = density / np.maximum(background, 1e-6)
        preference[idx] = _minmax(local_preference)

    return np.clip(np.nan_to_num(preference, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _distance_background_density(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if len(distance) == 0:
        return distance
    sample = distance
    if len(sample) > 600:
        order = np.linspace(0, len(sample) - 1, 600).astype(int)
        sample = np.sort(sample)[order]
    kernel = np.exp(-((distance[:, None] - sample[None, :]) ** 2) / (2.0 * max(float(bandwidth), 1e-6) ** 2))
    return kernel.mean(axis=1)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    denom = max(len(values) - 1, 1)
    return ranks / denom


def _balanced_train_indices(
    candidates: pd.DataFrame,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    strategy: str,
    max_train_samples: int,
    seed: int,
) -> np.ndarray:
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Pseudo-label split produced empty positive or negative training set.")
    if strategy == "balanced_per_lr":
        return _balanced_train_indices_per_lr(
            candidates,
            pos_idx,
            neg_idx,
            max_train_samples=max_train_samples,
            seed=seed,
        )
    if strategy != "balanced_global":
        raise ValueError(f"Unknown train_sampling_strategy: {strategy}")
    rng = np.random.default_rng(seed)
    n_each = min(len(pos_idx), len(neg_idx), max_train_samples // 2)
    pos = rng.choice(pos_idx, size=n_each, replace=False)
    neg = rng.choice(neg_idx, size=n_each, replace=False)
    train_idx = np.concatenate([pos, neg])
    rng.shuffle(train_idx)
    return train_idx.astype(np.int64)


def _balanced_train_indices_per_lr(
    candidates: pd.DataFrame,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_set = set(np.asarray(pos_idx, dtype=np.int64).tolist())
    neg_set = set(np.asarray(neg_idx, dtype=np.int64).tolist())
    selected = []
    lr_groups = list(candidates.groupby("lr_id", sort=False))
    if not lr_groups:
        return _balanced_train_indices(
            candidates,
            pos_idx,
            neg_idx,
            strategy="balanced_global",
            max_train_samples=max_train_samples,
            seed=seed,
        )
    per_lr_budget = max(int(max_train_samples) // max(len(lr_groups), 1), 2)
    for _, group in lr_groups:
        idx = group.index.to_numpy(dtype=np.int64)
        pos = np.array([i for i in idx if i in pos_set], dtype=np.int64)
        neg = np.array([i for i in idx if i in neg_set], dtype=np.int64)
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_each = min(len(pos), len(neg), max(per_lr_budget // 2, 1))
        selected.append(rng.choice(pos, size=n_each, replace=False))
        selected.append(rng.choice(neg, size=n_each, replace=False))
    if not selected:
        return _balanced_train_indices(
            candidates,
            pos_idx,
            neg_idx,
            strategy="balanced_global",
            max_train_samples=max_train_samples,
            seed=seed,
        )
    train_idx = np.unique(np.concatenate(selected)).astype(np.int64)
    if len(train_idx) > max_train_samples:
        train_idx = rng.choice(train_idx, size=int(max_train_samples), replace=False)
    rng.shuffle(train_idx)
    return train_idx.astype(np.int64)


def _weighted_bce_or_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: torch.Tensor,
    focal_gamma: float,
    focal_alpha: float,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    if float(focal_gamma) > 0:
        prob = torch.sigmoid(logits)
        pt = prob * labels + (1.0 - prob) * (1.0 - labels)
        alpha = float(focal_alpha) * labels + (1.0 - float(focal_alpha)) * (1.0 - labels)
        bce = alpha * torch.pow((1.0 - pt).clamp_min(1e-6), float(focal_gamma)) * bce
    return (bce * weight).mean()


def _sample_weight(
    candidates: pd.DataFrame,
    pos_idx: np.ndarray,
    expression: np.ndarray,
    *,
    weak_positive_loss_weight: float,
    weak_positive_expression_quantile: float,
) -> np.ndarray:
    weight = np.ones(len(candidates), dtype=float)
    boost = float(weak_positive_loss_weight)
    if boost <= 1.0 or len(pos_idx) == 0:
        return weight

    expression = np.asarray(expression, dtype=float)
    expr_rank = _rank01(expression)
    expression_present = (
        (candidates["ligand_expr"].to_numpy(dtype=float) > 0)
        & (candidates["receptor_expr"].to_numpy(dtype=float) > 0)
    )
    rule_like = _rule_like_mask(candidates)
    low_expr = expr_rank <= float(weak_positive_expression_quantile)
    pos_mask = np.zeros(len(candidates), dtype=bool)
    pos_mask[np.asarray(pos_idx, dtype=np.int64)] = True
    weak_like_positive = pos_mask & expression_present & rule_like & low_expr
    weight[weak_like_positive] *= boost
    return np.clip(weight, 1.0, max(boost, 1.0))


def _ranking_pair_indices(
    candidates: pd.DataFrame,
    pseudo_score: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_set = set(np.asarray(pos_idx, dtype=np.int64).tolist())
    neg_set = set(np.asarray(neg_idx, dtype=np.int64).tolist())
    if not pos_set or not neg_set:
        return np.asarray(pos_idx, dtype=np.int64), np.asarray(neg_idx, dtype=np.int64)

    expression = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float)) * np.log1p(
        candidates["receptor_expr"].to_numpy(dtype=float)
    )
    expression_q70 = float(np.quantile(expression, 0.70))
    weak_like = expression <= expression_q70
    rule_like = _rule_like_mask(candidates)
    hard_negative = _hard_negative_mask(candidates, expression)

    pos_pairs = []
    neg_pairs = []
    for _, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        pos_candidates = np.array([i for i in idx if i in pos_set], dtype=np.int64)
        neg_candidates = np.array([i for i in idx if i in neg_set], dtype=np.int64)
        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue

        preferred_pos = pos_candidates[weak_like[pos_candidates] & rule_like[pos_candidates]]
        if len(preferred_pos) == 0:
            preferred_pos = pos_candidates[rule_like[pos_candidates]]
        if len(preferred_pos) == 0:
            preferred_pos = pos_candidates

        preferred_neg = neg_candidates[hard_negative[neg_candidates]]
        if len(preferred_neg) == 0:
            preferred_neg = neg_candidates[np.argsort(-expression[neg_candidates])[: len(preferred_pos)]]

        n_pairs = min(len(preferred_pos), len(preferred_neg))
        if n_pairs == 0:
            continue
        pos_order = np.argsort(pseudo_score[preferred_pos])[:n_pairs]
        neg_order = np.argsort(-expression[preferred_neg])[:n_pairs]
        pos_pairs.append(preferred_pos[pos_order])
        neg_pairs.append(preferred_neg[neg_order])

    if not pos_pairs or not neg_pairs:
        n_pairs = min(len(pos_idx), len(neg_idx), max(int(max_pairs), 1))
        return np.asarray(pos_idx[:n_pairs], dtype=np.int64), np.asarray(neg_idx[:n_pairs], dtype=np.int64)

    ranking_pos = np.concatenate(pos_pairs).astype(np.int64)
    ranking_neg = np.concatenate(neg_pairs).astype(np.int64)
    if len(ranking_pos) > max_pairs:
        keep = rng.choice(np.arange(len(ranking_pos)), size=int(max_pairs), replace=False)
        ranking_pos = ranking_pos[keep]
        ranking_neg = ranking_neg[keep]
    return ranking_pos, ranking_neg


def _rule_like_mask(candidates: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(candidates), dtype=bool)
    for column in ("type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied"):
        if column in candidates:
            mask &= candidates[column].astype(bool).to_numpy()
    return mask


def _hard_negative_mask(candidates: pd.DataFrame, expression: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(candidates), dtype=bool)
    if "false_positive_high_expression" in candidates:
        mask |= candidates["false_positive_high_expression"].astype(bool).to_numpy()
    if "niche_rule_satisfied" in candidates:
        mask |= ~candidates["niche_rule_satisfied"].astype(bool).to_numpy()
    if "type_rule_satisfied" in candidates:
        mask |= ~candidates["type_rule_satisfied"].astype(bool).to_numpy()
    mask |= expression >= np.quantile(expression, 0.85)
    return mask


def _feasibility_target(candidates: pd.DataFrame) -> np.ndarray:
    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression_present = (ligand > 0) & (receptor > 0)
    target = expression_present.astype(float)
    target *= _rule_like_mask(candidates).astype(float)
    target *= _mechanism_distance_prior(candidates)
    if "false_positive_high_expression" in candidates:
        target *= np.where(candidates["false_positive_high_expression"].astype(bool).to_numpy(), 0.0, 1.0)
    return np.clip(np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _feasibility_weight(candidates: pd.DataFrame, feasibility_target: np.ndarray) -> np.ndarray:
    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression = np.log1p(ligand) * np.log1p(receptor)
    weight = np.ones(len(candidates), dtype=float)
    weight += 2.0 * (feasibility_target <= 0.05)
    if "false_positive_high_expression" in candidates:
        weight += 3.0 * candidates["false_positive_high_expression"].astype(bool).to_numpy()
    if len(expression):
        weight += 1.5 * _rank01(expression)
    return np.clip(weight, 1.0, 8.0)


def _feasibility_train_indices(
    candidates: pd.DataFrame,
    feasibility_target: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    high = np.flatnonzero(feasibility_target >= 0.25)
    low = np.flatnonzero(feasibility_target <= 0.05)
    if len(high) == 0 or len(low) == 0:
        return np.arange(len(candidates), dtype=np.int64)
    n_each = min(len(high), len(low), max(int(max_train_samples) // 2, 1))
    selected = np.concatenate(
        [
            rng.choice(high, size=n_each, replace=False),
            rng.choice(low, size=n_each, replace=False),
        ]
    )
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _hard_negative_train_indices(
    candidates: pd.DataFrame,
    expression: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> np.ndarray:
    hard_negative = _hard_negative_mask(candidates, expression)
    idx = np.flatnonzero(hard_negative)
    if len(idx) == 0:
        return idx.astype(np.int64)
    order = np.argsort(-expression[idx])
    idx = idx[order]
    cap = max(int(max_train_samples) // 2, 1)
    if len(idx) > cap:
        idx = idx[:cap]
    rng = np.random.default_rng(seed + 202)
    rng.shuffle(idx)
    return idx.astype(np.int64)


def _background_guard_train_indices(
    candidates: pd.DataFrame,
    expression: np.ndarray,
    *,
    max_train_samples: int,
    background_guard_targets: pd.DataFrame | None = None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select broad-background and high-expression risk candidates without using truth labels."""

    expression = np.asarray(expression, dtype=float)
    rule_like = _rule_like_mask(candidates)
    expr_rank = _rank01(expression)
    risk_weight = np.zeros(len(candidates), dtype=np.float32)
    target_mask = np.zeros(len(candidates), dtype=bool)

    if "false_positive_high_expression" in candidates:
        false_positive = candidates["false_positive_high_expression"].fillna(False).astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(false_positive, 1.00, 0.0).astype(np.float32))

    if "biological_category" in candidates:
        category = candidates["biological_category"].astype(str)
        high_expr_risk = category.eq("high_expression_negative_risk").to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(high_expr_risk, 0.95, 0.0).astype(np.float32))
    else:
        category = pd.Series([""] * len(candidates), index=candidates.index)

    if "selected_reason" in candidates:
        selected_reason = candidates["selected_reason"].astype(str)
        category_balanced = selected_reason.eq("category_balanced").to_numpy()
        broad_other = category_balanced & category.eq("other").to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(broad_other, 0.85, 0.0).astype(np.float32))
        if "source_count" in candidates:
            source_count = pd.to_numeric(candidates["source_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            low_source_background = category_balanced & (source_count <= 2.0)
            risk_weight = np.maximum(
                risk_weight,
                np.where(low_source_background, 0.70, 0.0).astype(np.float32),
            )

    non_rule_high_expression = (~rule_like) & (expr_rank >= 0.70)
    risk_weight = np.maximum(risk_weight, np.where(non_rule_high_expression, 0.65, 0.0).astype(np.float32))
    if background_guard_targets is not None and not background_guard_targets.empty:
        target_mask, target_weight = _background_guard_target_weights(candidates, background_guard_targets)
        risk_weight = np.maximum(risk_weight, target_weight.astype(np.float32))

    idx = np.flatnonzero(risk_weight > 0)
    if len(idx) == 0:
        return idx.astype(np.int64), np.ones(len(candidates), dtype=np.float32), idx.astype(np.int64)

    risk_score = risk_weight[idx].astype(float) * (0.40 + 0.60 * expr_rank[idx])
    if target_mask.any():
        risk_score = risk_score + np.where(target_mask[idx], 1.0, 0.0)
    order = np.argsort(-risk_score)
    idx = idx[order]
    cap = max(int(max_train_samples) // 2, 1)
    if len(idx) > cap:
        idx = idx[:cap]
    rng = np.random.default_rng(seed + 909)
    rng.shuffle(idx)
    target_idx = idx[target_mask[idx]]
    return (
        idx.astype(np.int64),
        np.clip(risk_weight, 0.05, 1.0).astype(np.float32),
        target_idx.astype(np.int64),
    )


def _mechanism_guard_train_indices(
    candidates: pd.DataFrame,
    expression: np.ndarray,
    pseudo_score: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select mechanism-inconsistent but deceptively plausible negatives.

    This is benchmark-agnostic in the sense that it only uses generic mechanism
    consistency annotations when they exist: rule-satisfied flags, range
    violation tags, niche consistency and optional high-expression negative
    labels. It focuses training on negatives that look plausible by expression
    or pseudo-label score, rather than on easy background negatives.
    """

    expression = np.asarray(expression, dtype=float)
    pseudo_score = np.asarray(pseudo_score, dtype=float)
    expr_rank = _rank01(expression)
    pseudo_rank = _rank01(pseudo_score)
    risk_weight = np.zeros(len(candidates), dtype=np.float32)

    if "wrong_range_false_positive" in candidates:
        wrong_range = candidates["wrong_range_false_positive"].fillna(False).astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(wrong_range, 1.00, 0.0).astype(np.float32))

    if "range_violation_type" in candidates:
        violation = candidates["range_violation_type"].astype(str).to_numpy()
        too_close = violation == "too_close_for_long_range_lr"
        too_far = violation == "too_far_for_lr"
        risk_weight = np.maximum(risk_weight, np.where(too_close, 1.00, 0.0).astype(np.float32))
        risk_weight = np.maximum(risk_weight, np.where(too_far, 0.80, 0.0).astype(np.float32))

    if "niche_rule_satisfied" in candidates:
        niche_bad = ~candidates["niche_rule_satisfied"].astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(niche_bad, 0.85, 0.0).astype(np.float32))

    if "range_rule_satisfied" in candidates:
        range_bad = ~candidates["range_rule_satisfied"].astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(range_bad, 0.75, 0.0).astype(np.float32))

    if "type_rule_satisfied" in candidates:
        type_bad = ~candidates["type_rule_satisfied"].astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(type_bad, 0.60, 0.0).astype(np.float32))

    if "false_positive_high_expression" in candidates:
        high_expr_negative = candidates["false_positive_high_expression"].fillna(False).astype(bool).to_numpy()
        risk_weight = np.maximum(risk_weight, np.where(high_expr_negative, 1.00, 0.0).astype(np.float32))

    if "mechanism" in candidates:
        mechanism = candidates["mechanism"].astype(str).to_numpy()
        long_range = mechanism == "long_range_chemokine_niche"
        risk_weight = np.where(long_range & (risk_weight > 0), risk_weight + 0.10, risk_weight).astype(np.float32)

    idx = np.flatnonzero(risk_weight > 0)
    if len(idx) == 0:
        return idx.astype(np.int64), np.ones(len(candidates), dtype=np.float32)

    plausibility = 0.45 + 0.35 * expr_rank[idx] + 0.20 * pseudo_rank[idx]
    keep_mask = (expr_rank[idx] >= 0.50) | (pseudo_rank[idx] >= 0.75)
    idx = idx[keep_mask]
    if len(idx) == 0:
        return idx.astype(np.int64), np.clip(risk_weight, 0.05, 1.2).astype(np.float32)

    risk_score = risk_weight[idx].astype(float) * plausibility[keep_mask]
    order = np.argsort(-risk_score)
    idx = idx[order]
    cap = max(int(max_train_samples) // 2, 1)
    if len(idx) > cap:
        idx = idx[:cap]
    rng = np.random.default_rng(seed + 919)
    rng.shuffle(idx)
    return idx.astype(np.int64), np.clip(risk_weight, 0.05, 1.2).astype(np.float32)


def _background_guard_target_weights(
    candidates: pd.DataFrame,
    background_guard_targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    target = background_guard_targets.copy()
    if "recommended_loss_weight" not in target:
        target["recommended_loss_weight"] = 1.0
    target["recommended_loss_weight"] = pd.to_numeric(
        target["recommended_loss_weight"],
        errors="coerce",
    ).fillna(1.0)
    weight_by_lr: dict[str, float] = {}
    if "lr_id" in target:
        for lr_id, group in target.groupby("lr_id", dropna=False):
            weight_by_lr[str(lr_id)] = float(group["recommended_loss_weight"].max())
    if "lr_pair" in target:
        for lr_pair, group in target.groupby("lr_pair", dropna=False):
            weight_by_lr[str(lr_pair)] = max(
                float(group["recommended_loss_weight"].max()),
                weight_by_lr.get(str(lr_pair), 0.0),
            )

    weights = np.zeros(len(candidates), dtype=np.float32)
    if "lr_id" in candidates:
        lr_ids = candidates["lr_id"].astype(str).to_numpy()
        for lr_id, value in weight_by_lr.items():
            weights[lr_ids == lr_id] = np.maximum(weights[lr_ids == lr_id], float(value))
    if {"ligand", "receptor"}.issubset(candidates.columns):
        lr_pairs = (candidates["ligand"].astype(str) + "-" + candidates["receptor"].astype(str)).to_numpy()
        for lr_pair, value in weight_by_lr.items():
            weights[lr_pairs == lr_pair] = np.maximum(weights[lr_pairs == lr_pair], float(value))
    mask = weights > 0
    return mask, np.clip(weights, 0.0, 1.0).astype(np.float32)


def _context_contrastive_indices(
    candidates: pd.DataFrame,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    pseudo_score: np.ndarray,
    *,
    max_train_samples: int,
    n_cells: int,
    negative_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression_present = (ligand > 0) & (receptor > 0)
    rule_like = _rule_like_mask(candidates)
    allowed = expression_present & rule_like
    if "false_positive_high_expression" in candidates:
        allowed &= ~candidates["false_positive_high_expression"].astype(bool).to_numpy()
    idx = np.flatnonzero(allowed)
    if len(idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    keep_parts = []
    cap_per_lr = max(int(max_train_samples) // max(candidates["lr_id"].nunique(), 1), 1)
    for _, group in candidates.iloc[idx].groupby("lr_id", sort=False):
        local = group.index.to_numpy(dtype=np.int64)
        order = np.argsort(-pseudo_score[local])[:cap_per_lr]
        keep_parts.append(local[order])
    keep = np.unique(np.concatenate(keep_parts)).astype(np.int64)
    cap = max(int(max_train_samples) // 2, 1)
    rng = np.random.default_rng(seed + 303)
    if len(keep) > cap:
        keep = rng.choice(keep, size=cap, replace=False)

    if negative_mode == "random_receiver":
        neg_dst = rng.integers(0, int(n_cells), size=len(keep), endpoint=False, dtype=np.int64)
    elif negative_mode == "hard_receiver":
        neg_dst = _hard_context_negative_receivers(candidates, keep, dst_idx, receptor, seed=seed)
    else:
        raise ValueError(
            "Unknown context_contrastive_negative_mode: "
            f"{negative_mode}. Expected random_receiver or hard_receiver."
        )
    same = neg_dst == dst_idx[keep]
    if same.any() and n_cells > 1:
        neg_dst[same] = (neg_dst[same] + 1) % int(n_cells)
    same_src = neg_dst == src_idx[keep]
    if same_src.any() and n_cells > 1:
        neg_dst[same_src] = (neg_dst[same_src] + 1) % int(n_cells)
    return keep.astype(np.int64), neg_dst.astype(np.int64)


def _structural_contrastive_indices(
    candidates: pd.DataFrame,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    *,
    max_train_samples: int,
    n_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build label-independent positive/negative pairs from observed expression.

    This diagnostic objective never reads pseudo-scores, rule flags, LR class,
    simulation truth or control labels. It only asks whether an observed
    ligand-positive sender paired with a receptor-positive receiver scores above
    the same sender paired with a randomly selected receiver.
    """

    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression = np.log1p(ligand) * np.log1p(receptor)
    observed = (ligand > 0) & (receptor > 0)
    rng = np.random.default_rng(seed + 707)
    keep_parts: list[np.ndarray] = []
    per_lr_cap = max(int(max_train_samples) // max(candidates["lr_id"].nunique(), 1), 1)
    for _, group in candidates.loc[observed].groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) == 0:
            continue
        order = np.argsort(-expression[idx])[:per_lr_cap]
        keep_parts.append(idx[order])
    if not keep_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    keep = np.unique(np.concatenate(keep_parts)).astype(np.int64)
    cap = max(int(max_train_samples) // 2, 1)
    if len(keep) > cap:
        keep = rng.choice(keep, size=cap, replace=False)
    neg_dst = rng.integers(0, int(n_cells), size=len(keep), endpoint=False, dtype=np.int64)
    same = (neg_dst == dst_idx[keep]) | (neg_dst == src_idx[keep])
    if same.any() and n_cells > 2:
        neg_dst[same] = (neg_dst[same] + 1) % int(n_cells)
        still_same = (neg_dst == dst_idx[keep]) | (neg_dst == src_idx[keep])
        neg_dst[still_same] = (neg_dst[still_same] + 1) % int(n_cells)
    return keep.astype(np.int64), neg_dst.astype(np.int64)


def _hard_context_negative_receivers(
    candidates: pd.DataFrame,
    keep_idx: np.ndarray,
    dst_idx: np.ndarray,
    receptor_expr: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 404)
    hard_mask = np.zeros(len(candidates), dtype=bool)
    if "false_positive_high_expression" in candidates:
        hard_mask |= candidates["false_positive_high_expression"].astype(bool).to_numpy()
    for column in ("type_rule_satisfied", "range_rule_satisfied", "niche_rule_satisfied"):
        if column in candidates:
            hard_mask |= ~candidates[column].astype(bool).to_numpy()
    receptor_rank = _rank01(receptor_expr)
    hard_mask |= receptor_rank >= 0.75

    neg_dst = np.array(dst_idx[keep_idx], copy=True)
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        local_keep = keep_idx[candidates.loc[keep_idx, "lr_id"].astype(str).to_numpy() == str(lr_id)]
        if len(local_keep) == 0:
            continue
        local_candidates = group.index.to_numpy(dtype=np.int64)
        pool = local_candidates[hard_mask[local_candidates]]
        if len(pool) == 0:
            pool = local_candidates
        order = np.argsort(-receptor_expr[pool])
        pool = pool[order[: max(len(local_keep) * 4, len(local_keep), 1)]]
        sampled_rows = rng.choice(pool, size=len(local_keep), replace=len(pool) < len(local_keep))
        neg_dst[np.isin(keep_idx, local_keep)] = dst_idx[sampled_rows]
    return neg_dst.astype(np.int64)


def _context_contrastive_negative_features(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    context_idx: np.ndarray,
    src_idx: np.ndarray,
    neg_dst_idx: np.ndarray,
    *,
    use_edge_prior_features: bool,
    edge_feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(context_idx) == 0:
        return np.array([], dtype=np.float32), np.zeros((0, int(edge_feature_dim)), dtype=np.float32)

    table = candidates.iloc[context_idx].copy()
    sender = cells.iloc[src_idx[context_idx]].reset_index(drop=True)
    receiver = cells.iloc[neg_dst_idx].reset_index(drop=True)
    table = table.reset_index(drop=True)
    table["sender_id"] = sender["cell_id"].to_numpy()
    table["receiver_id"] = receiver["cell_id"].to_numpy()
    table["sender_cell_type"] = sender["cell_type"].to_numpy()
    table["receiver_cell_type"] = receiver["cell_type"].to_numpy()
    table["sender_niche"] = sender["niche"].to_numpy()
    table["receiver_niche"] = receiver["niche"].to_numpy()

    sx = sender["x"].to_numpy(dtype=float)
    sy = sender["y"].to_numpy(dtype=float)
    rx = receiver["x"].to_numpy(dtype=float)
    ry = receiver["y"].to_numpy(dtype=float)
    distance = np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2)
    table["distance"] = distance
    table["ligand_expr"] = _lookup_expression(expression, table["sender_id"], table["ligand"])
    table["receptor_expr"] = _lookup_expression(expression, table["receiver_id"], table["receptor"])
    table["range_rule_satisfied"] = distance <= pd.to_numeric(
        table.get("spatial_range", pd.Series(np.inf, index=table.index)),
        errors="coerce",
    ).fillna(np.inf).to_numpy(dtype=float)
    table["niche_rule_satisfied"] = _niche_rule(table)

    if use_edge_prior_features:
        edge_features = _edge_prior_features(table)
    else:
        edge_features = np.zeros((len(table), 0), dtype=np.float32)
    return distance.astype(np.float32), edge_features.astype(np.float32)


def _lr_identity_contrastive_indices(
    candidates: pd.DataFrame,
    pseudo_score: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 505)
    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression = np.log1p(ligand) * np.log1p(receptor)
    expression_present = (ligand > 0) & (receptor > 0)
    rule_like = _rule_like_mask(candidates)
    false_positive = (
        candidates["false_positive_high_expression"].astype(bool).to_numpy()
        if "false_positive_high_expression" in candidates
        else np.zeros(len(candidates), dtype=bool)
    )
    feasible_positive = expression_present & rule_like & ~false_positive
    hard_negative = false_positive | (~rule_like & (expression >= np.quantile(expression, 0.60)))
    if len(expression):
        hard_negative |= expression >= np.quantile(expression, 0.90)

    pos_parts = []
    neg_parts = []
    edge_key = candidates["sender_id"].astype(str) + "||" + candidates["receiver_id"].astype(str)
    for _, group in candidates.groupby(edge_key, sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        pos = idx[feasible_positive[idx]]
        neg = idx[hard_negative[idx] & ~feasible_positive[idx]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_order = np.argsort(-pseudo_score[pos])
        neg_order = np.argsort(-expression[neg])
        n_pairs = min(len(pos_order), len(neg_order), 2)
        if n_pairs == 0:
            continue
        pos_parts.append(pos[pos_order[:n_pairs]])
        neg_parts.append(neg[neg_order[:n_pairs]])

    if not pos_parts or not neg_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    pos_idx = np.concatenate(pos_parts).astype(np.int64)
    neg_idx = np.concatenate(neg_parts).astype(np.int64)
    if len(pos_idx) > max_pairs:
        keep = rng.choice(np.arange(len(pos_idx)), size=max(int(max_pairs), 1), replace=False)
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
    return pos_idx.astype(np.int64), neg_idx.astype(np.int64)


def _scale_contrastive_indices(
    candidates: pd.DataFrame,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair high-expression in-scale edges against high-expression out-of-scale edges.

    The in/out scale assignment is inferred from LR-specific expression-distance
    density and does not use benchmark truth or range-violation annotations.
    """

    rng = np.random.default_rng(seed + 515)
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(dtype=float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(dtype=float))
    expression = ligand * receptor
    scale_pref = _lr_scale_preference(candidates, expression)

    pos_parts = []
    neg_parts = []
    for _, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) < 8:
            continue
        local_expr = expression[idx]
        local_scale = scale_pref[idx]
        expr_high = local_expr >= np.quantile(local_expr, 0.60)
        pos_mask = expr_high & (local_scale >= np.quantile(local_scale, 0.70))
        neg_mask = expr_high & (local_scale <= np.quantile(local_scale, 0.30))
        pos = idx[pos_mask]
        neg = idx[neg_mask]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_order = np.argsort(-local_scale[pos_mask])
        neg_order = np.argsort(local_scale[neg_mask])
        n_pairs = min(len(pos), len(neg), max(int(max_pairs) // max(candidates["lr_id"].nunique(), 1), 1))
        pos_parts.append(pos[pos_order[:n_pairs]])
        neg_parts.append(neg[neg_order[:n_pairs]])

    if not pos_parts or not neg_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    pos_idx = np.concatenate(pos_parts).astype(np.int64)
    neg_idx = np.concatenate(neg_parts).astype(np.int64)
    if len(pos_idx) > max_pairs:
        keep = rng.choice(np.arange(len(pos_idx)), size=max(int(max_pairs), 1), replace=False)
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
    return pos_idx.astype(np.int64), neg_idx.astype(np.int64)


def _scale_alignment_weight(candidates: pd.DataFrame, expression: np.ndarray) -> np.ndarray:
    scale_pref = _lr_scale_preference(candidates, expression)
    expr_rank = np.zeros(len(candidates), dtype=float)
    for _, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        expr_rank[idx] = _rank01(expression[idx])
    return np.clip(expr_rank * (0.20 + 0.80 * scale_pref), 0.0, 1.0)


def _wavelet_response_scale_targets(
    h: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    lr_idx: torch.Tensor,
    weight: torch.Tensor,
    *,
    n_lr: int,
) -> torch.Tensor | None:
    if h.dim() != 3 or h.size(1) < 2:
        return None
    weight = weight.to(device=h.device, dtype=h.dtype).clamp_min(0.0)
    keep = weight > torch.quantile(weight, 0.60) if weight.numel() >= 8 else weight > 0
    if not bool(keep.any()):
        return None
    src = torch.nn.functional.normalize(h[src_idx[keep]], dim=-1)
    dst = torch.nn.functional.normalize(h[dst_idx[keep]], dim=-1)
    response = torch.relu((src * dst).sum(dim=-1))
    response = response * weight[keep].unsqueeze(-1)
    target = h.new_zeros((int(n_lr), h.size(1)))
    target.index_add_(0, lr_idx[keep], response)
    row_sum = target.sum(dim=1, keepdim=True)
    valid = row_sum.squeeze(-1) > 0
    if not bool(valid.any()):
        return None
    uniform = torch.full_like(target, 1.0 / float(h.size(1)))
    target = torch.where(valid.unsqueeze(-1), target / row_sum.clamp_min(1e-8), uniform)
    return target.detach()


def _default_scale_weight(model_name: str, value: float | None, default_for_scale_model: float) -> float:
    if value is not None:
        return float(value)
    return float(default_for_scale_model) if str(model_name) in SCALE_AWARE_BASELINE_NAMES else 0.0


def _scale_kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.kl_div(torch.log(pred.clamp_min(1e-8)), target, reduction="batchmean")


def _scale_diversity_loss(pred: torch.Tensor) -> torch.Tensor:
    entropy_each = -(pred * torch.log(pred.clamp_min(1e-8))).sum(dim=1).mean()
    mean_dist = pred.mean(dim=0)
    entropy_mean = -(mean_dist * torch.log(mean_dist.clamp_min(1e-8))).sum()
    return entropy_each - entropy_mean


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
    if getattr(scorer, "use_distance", True):
        parts.append(dist / scorer.distance_scale)
    if getattr(scorer, "edge_feature_dim", 0):
        parts.append(edge_features)
    base_input = torch.cat(parts, dim=-1)

    context_parts = [h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, e_lr]
    if getattr(scorer, "use_distance", True):
        context_parts.append(dist / scorer.distance_scale)
    if getattr(scorer, "edge_feature_dim", 0):
        context_parts.append(edge_features)
    context_input = torch.cat(context_parts, dim=-1)

    hidden = scorer.base_mlp(base_input)
    gate = scorer.context_gate(context_input)
    return scorer.out(hidden * gate).squeeze(-1)


def _lr_scale_target(
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    strategy: str,
    n_scales: int,
) -> torch.Tensor:
    strategy = str(strategy)
    if strategy in {"ranklocal", "pseudo_ranklocal", "mixture_ranklocal_responsibility"}:
        target_by_lr = _pseudo_rank_local_scale_targets(candidates, n_scales=n_scales)
    elif strategy in {"rankdist", "pseudo_rankdist"}:
        target_by_lr = _pseudo_rank_distance_scale_targets(candidates, n_scales=n_scales)
    elif strategy in {"distance", "pseudo_distance"}:
        target_by_lr = _pseudo_distance_scale_targets(candidates, n_scales=n_scales)
    else:
        raise ValueError(f"Unknown lr_scale_target_strategy: {strategy}")
    target = [
        target_by_lr.get(str(lr_id), np.full(n_scales, 1.0 / float(n_scales)))
        for lr_id in lr_pairs["lr_id"].astype(str)
    ]
    return torch.as_tensor(np.vstack(target), dtype=torch.float32)


def _pseudo_distance_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = np.nan_to_num(ligand * receptor, nan=0.0, posinf=0.0, neginf=0.0)
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    centers = np.linspace(0.0, 1.0, n_scales)
    width = max(1.0 / max(n_scales - 1, 1), 0.18)
    out: dict[str, np.ndarray] = {}
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
        target = np.exp(-0.5 * ((centers - weighted_d) / width) ** 2) + 0.02
        out[str(lr_id)] = target / target.sum()
    return out


def _pseudo_rank_distance_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    centers = _expression_distance_centers(candidates, quantile=0.60, power=1.0)
    return _ranked_center_targets(centers, n_scales=n_scales)


def _pseudo_rank_local_scale_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, np.ndarray]:
    centers = _expression_local_distance_centers(candidates, quantile=0.60, local_fraction=0.35)
    return _ranked_center_targets(centers, n_scales=n_scales)


def _ranked_center_targets(centers: dict[str, float], *, n_scales: int) -> dict[str, np.ndarray]:
    lr_ids = list(centers)
    values = np.array([centers[lr_id] for lr_id in lr_ids], dtype=float)
    order = np.argsort(np.argsort(values))
    rank_centers = order / max(len(order) - 1, 1)
    scale_centers = np.linspace(0.0, 1.0, n_scales)
    out: dict[str, np.ndarray] = {}
    for lr_id, center in zip(lr_ids, rank_centers):
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


def _expression_local_distance_centers(
    candidates: pd.DataFrame,
    *,
    quantile: float,
    local_fraction: float,
) -> dict[str, float]:
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
        if int(keep.sum()) < 4:
            keep = e >= np.quantile(e, max(0.55, quantile - 0.2))
        local_d = d[keep]
        local_w = np.maximum(e[keep], 1e-6)
        cutoff = np.quantile(local_d, local_fraction)
        local_keep = local_d <= cutoff
        if int(local_keep.sum()) == 0:
            out[str(lr_id)] = float(np.average(local_d, weights=local_w))
        else:
            out[str(lr_id)] = float(np.average(local_d[local_keep], weights=local_w[local_keep]))
    return out


def _build_lr_scale_output(
    lr_pairs: pd.DataFrame,
    scale_distribution: np.ndarray,
    target_distribution: np.ndarray | None,
    *,
    target_strategy: str,
) -> pd.DataFrame:
    idx = np.arange(scale_distribution.shape[1], dtype=float)
    rows = []
    for i, row in lr_pairs.reset_index(drop=True).iterrows():
        scale = scale_distribution[i]
        out = {
            "lr_id": row.get("lr_id", ""),
            "ligand": row.get("ligand", ""),
            "receptor": row.get("receptor", ""),
            "lr_pair": f"{row.get('ligand', '')}-{row.get('receptor', '')}",
            "mechanism": row.get("mechanism", ""),
            "target_strategy": target_strategy,
            "expected_scale": float((scale * idx).sum()),
            "local_weight": float(scale[:2].sum()),
            "broad_weight": float(scale[-2:].sum()),
        }
        if target_distribution is not None:
            target = target_distribution[i]
            out["target_expected_scale"] = float((target * idx).sum())
            out["target_local_weight"] = float(target[:2].sum())
            out["target_broad_weight"] = float(target[-2:].sum())
        for j in range(scale_distribution.shape[1]):
            out[f"scale_{j}"] = float(scale[j])
            if target_distribution is not None:
                out[f"target_scale_{j}"] = float(target_distribution[i, j])
        rows.append(out)
    return pd.DataFrame(rows)


def _control_contrastive_indices(
    candidates: pd.DataFrame,
    pseudo_score: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair curated plausible controls against explicit high-expression negatives on the same edge."""

    false_positive = (
        candidates["false_positive_high_expression"].fillna(False).astype(bool).to_numpy()
        if "false_positive_high_expression" in candidates
        else np.zeros(len(candidates), dtype=bool)
    )
    if not false_positive.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if "true_lr" in candidates:
        positive_control = candidates["true_lr"].fillna(False).astype(bool).to_numpy()
    elif "control_class" in candidates:
        control_class = candidates["control_class"].astype(str)
        positive_control = control_class.str.contains("positive", case=False, na=False).to_numpy()
    else:
        positive_control = np.ones(len(candidates), dtype=bool)

    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression = np.log1p(ligand) * np.log1p(receptor)
    expression_present = (ligand > 0) & (receptor > 0)
    rule_like = _rule_like_mask(candidates)
    positive_mask = positive_control & expression_present & rule_like & ~false_positive
    negative_mask = false_positive

    pos_parts = []
    neg_parts = []
    edge_key = candidates["sender_id"].astype(str) + "||" + candidates["receiver_id"].astype(str)
    for _, group in candidates.groupby(edge_key, sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        pos = idx[positive_mask[idx]]
        neg = idx[negative_mask[idx]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_order = np.argsort(-pseudo_score[pos])
        neg_order = np.argsort(-expression[neg])
        n_pairs = min(len(pos_order), len(neg_order), 2)
        if n_pairs == 0:
            continue
        pos_parts.append(pos[pos_order[:n_pairs]])
        neg_parts.append(neg[neg_order[:n_pairs]])

    if not pos_parts or not neg_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    pos_idx = np.concatenate(pos_parts).astype(np.int64)
    neg_idx = np.concatenate(neg_parts).astype(np.int64)
    if len(pos_idx) > max_pairs:
        rng = np.random.default_rng(seed + 606)
        keep = rng.choice(np.arange(len(pos_idx)), size=max(int(max_pairs), 1), replace=False)
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
    return pos_idx.astype(np.int64), neg_idx.astype(np.int64)


def _spatial_null_contrastive_indices(
    candidates: pd.DataFrame,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    pseudo_score: np.ndarray,
    *,
    max_pairs: int,
    n_cells: int,
    cells: pd.DataFrame,
    negative_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select plausible positives and pair them with distance/range-mismatched receivers."""

    ligand = candidates["ligand_expr"].to_numpy(dtype=float)
    receptor = candidates["receptor_expr"].to_numpy(dtype=float)
    expression_present = (ligand > 0) & (receptor > 0)
    rule_like = _rule_like_mask(candidates)
    positive_mask = expression_present & rule_like
    if "false_positive_high_expression" in candidates:
        positive_mask &= ~candidates["false_positive_high_expression"].astype(bool).to_numpy()
    idx = np.flatnonzero(positive_mask)
    if len(idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed + 707)
    keep_parts = []
    cap_per_lr = max(int(max_pairs) // max(candidates["lr_id"].nunique(), 1), 1)
    for _, group in candidates.iloc[idx].groupby("lr_id", sort=False):
        local = group.index.to_numpy(dtype=np.int64)
        order = np.argsort(-pseudo_score[local])[:cap_per_lr]
        keep_parts.append(local[order])
    keep = np.unique(np.concatenate(keep_parts)).astype(np.int64)
    if len(keep) > max_pairs:
        keep = rng.choice(keep, size=max(int(max_pairs), 1), replace=False)

    neg_dst = _spatial_null_negative_receivers(
        candidates,
        keep,
        src_idx,
        dst_idx,
        n_cells=n_cells,
        cells=cells,
        negative_mode=negative_mode,
        seed=seed,
    )
    same_dst = neg_dst == dst_idx[keep]
    if same_dst.any() and n_cells > 1:
        neg_dst[same_dst] = (neg_dst[same_dst] + 1) % int(n_cells)
    same_src = neg_dst == src_idx[keep]
    if same_src.any() and n_cells > 1:
        neg_dst[same_src] = (neg_dst[same_src] + 1) % int(n_cells)
    return keep.astype(np.int64), neg_dst.astype(np.int64)


def _spatial_null_negative_receivers(
    candidates: pd.DataFrame,
    keep_idx: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    *,
    n_cells: int,
    cells: pd.DataFrame,
    negative_mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 808)
    neg_dst = np.array(dst_idx[keep_idx], copy=True)
    all_receivers = np.arange(int(n_cells), dtype=np.int64)
    if negative_mode == "random_receiver":
        for mechanism, group in candidates.iloc[keep_idx].groupby("mechanism", sort=False):
            local_keep = group.index.to_numpy(dtype=np.int64)
            positions = np.flatnonzero(np.isin(keep_idx, local_keep))
            if len(positions) == 0:
                continue
            sampled = rng.choice(all_receivers, size=len(local_keep), replace=True)
            neg_dst[positions] = sampled
        return neg_dst.astype(np.int64)

    if negative_mode != "range_violation_receiver":
        raise ValueError(
            "Unknown spatial_null_negative_mode: "
            f"{negative_mode}. Expected random_receiver or range_violation_receiver."
        )

    coords = cells[["x", "y"]].to_numpy(dtype=float)
    niches = cells["niche"].astype(str).to_numpy() if "niche" in cells else np.array([""] * int(n_cells))
    for mechanism, group in candidates.iloc[keep_idx].groupby("mechanism", sort=False):
        local_keep = group.index.to_numpy(dtype=np.int64)
        positions = np.flatnonzero(np.isin(keep_idx, local_keep))
        if len(positions) == 0:
            continue
        mechanism = str(mechanism)
        spatial_range = pd.to_numeric(
            candidates.loc[local_keep, "spatial_range"]
            if "spatial_range" in candidates
            else pd.Series(np.nan, index=local_keep),
            errors="coerce",
        ).to_numpy(dtype=float)
        fallback = np.nanmedian(candidates.loc[local_keep, "distance"].to_numpy(dtype=float)) if len(local_keep) else 1.0
        spatial_range = np.nan_to_num(spatial_range, nan=max(float(fallback), 1.0), posinf=max(float(fallback), 1.0))
        active_niche = (
            candidates.loc[local_keep, "active_niche"].astype(str).to_numpy()
            if "active_niche" in candidates
            else np.array(["any"] * len(local_keep))
        )

        sampled = []
        for row_idx, row_range, row_niche in zip(local_keep, spatial_range, active_niche):
            src = int(src_idx[row_idx])
            source_coord = coords[src]
            distance = np.sqrt(((coords - source_coord) ** 2).sum(axis=1))
            row_range = max(float(row_range), 1e-6)
            if mechanism == "juxtacrine_contact":
                pool_mask = distance > 1.50 * row_range
            elif mechanism == "long_range_chemokine_niche":
                pool_mask = (distance > 1.20 * row_range) | (distance < 0.20 * row_range)
            else:
                pool_mask = distance > 1.15 * row_range

            row_niche = str(row_niche)
            if row_niche not in {"any", "", "nan", "None"}:
                pool_mask |= niches != row_niche
            pool_mask[src] = False
            pool = all_receivers[pool_mask]
            if len(pool) == 0:
                pool = all_receivers[all_receivers != src] if int(n_cells) > 1 else all_receivers
            sampled.append(int(rng.choice(pool)))
        neg_dst[positions] = np.asarray(sampled, dtype=np.int64)
    return neg_dst.astype(np.int64)


def _spatial_null_negative_features(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    spatial_null_idx: np.ndarray,
    src_idx: np.ndarray,
    neg_dst_idx: np.ndarray,
    *,
    use_edge_prior_features: bool,
    edge_feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(spatial_null_idx) == 0:
        return np.array([], dtype=np.float32), np.zeros((0, int(edge_feature_dim)), dtype=np.float32)

    table = candidates.iloc[spatial_null_idx].copy().reset_index(drop=True)
    sender = cells.iloc[src_idx[spatial_null_idx]].reset_index(drop=True)
    receiver = cells.iloc[neg_dst_idx].reset_index(drop=True)
    table["sender_id"] = sender["cell_id"].to_numpy()
    table["receiver_id"] = receiver["cell_id"].to_numpy()
    table["sender_cell_type"] = sender["cell_type"].to_numpy()
    table["receiver_cell_type"] = receiver["cell_type"].to_numpy()
    table["sender_niche"] = sender["niche"].to_numpy()
    table["receiver_niche"] = receiver["niche"].to_numpy()

    sx = sender["x"].to_numpy(dtype=float)
    sy = sender["y"].to_numpy(dtype=float)
    rx = receiver["x"].to_numpy(dtype=float)
    ry = receiver["y"].to_numpy(dtype=float)
    distance = np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2)
    table["distance"] = distance
    table["ligand_expr"] = _lookup_expression(expression, table["sender_id"], table["ligand"])
    table["receptor_expr"] = _lookup_expression(expression, table["receiver_id"], table["receptor"])
    spatial_range = pd.to_numeric(
        table.get("spatial_range", pd.Series(np.inf, index=table.index)),
        errors="coerce",
    ).fillna(np.inf).to_numpy(dtype=float)
    table["range_rule_satisfied"] = distance <= spatial_range
    table["niche_rule_satisfied"] = _niche_rule(table)

    if use_edge_prior_features:
        edge_features = _edge_prior_features(table)
    else:
        edge_features = np.zeros((len(table), 0), dtype=np.float32)
    return distance.astype(np.float32), edge_features.astype(np.float32)


def _lookup_expression(expr_index: pd.DataFrame, cell_ids: pd.Series, genes: pd.Series) -> np.ndarray:
    values = np.zeros(len(cell_ids), dtype=float)
    for gene in sorted(set(genes.astype(str))):
        mask = genes.astype(str).eq(gene).to_numpy()
        if gene in expr_index.columns:
            values[mask] = expr_index.loc[cell_ids[mask], gene].to_numpy(dtype=float)
    return values


def _niche_rule(candidates: pd.DataFrame) -> np.ndarray:
    active = candidates.get("active_niche", pd.Series(["any"] * len(candidates))).astype(str).to_numpy()
    sender = candidates["sender_niche"].astype(str).to_numpy()
    receiver = candidates["receiver_niche"].astype(str).to_numpy()
    unrestricted = (active == "any") | (active == "") | (active == "nan")
    return unrestricted | ((sender == active) & (receiver == active))


def _pairwise_tensor_distance(
    coords: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if coords.numel() == 0:
        return fallback
    src = coords[src_idx]
    dst = coords[dst_idx]
    distance = torch.sqrt(torch.sum((src - dst) ** 2, dim=1).clamp_min(1e-12))
    return torch.nan_to_num(distance, nan=0.0, posinf=float(fallback.max().item()) if fallback.numel() else 0.0)


def _indexed_pairwise_ranking_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return pos_logits.new_tensor(0.0)
    n_pairs = min(pos_logits.numel(), neg_logits.numel())
    return torch.relu(float(margin) - (pos_logits[:n_pairs] - neg_logits[:n_pairs])).mean()


def _knn_edge_index(coords: np.ndarray, *, k_nn: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n_cells = coords.shape[0]
    k_nn = min(max(int(k_nn), 1), max(n_cells - 1, 1))
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.sqrt((diff * diff).sum(axis=2))
        np.fill_diagonal(distance, np.inf)
        neighbors = np.argpartition(distance, kth=k_nn - 1, axis=1)[:, :k_nn]
    else:
        tree = cKDTree(coords)
        _, neighbors = tree.query(coords, k=k_nn + 1)
        neighbors = np.asarray(neighbors)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        neighbors = neighbors[:, 1 : k_nn + 1]
    src = np.repeat(np.arange(n_cells), k_nn)
    dst = neighbors.reshape(-1)
    return np.vstack([src, dst])


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (values - mean) / std


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    required_files = {
        "cells.csv",
        "expression.csv",
        "lr_pairs.csv",
        "candidate_interactions.csv",
    }
    if dataset_root.is_dir() and required_files.issubset({path.name for path in dataset_root.iterdir() if path.is_file()}):
        return [dataset_root]
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/spatialself",
        help="Directory for prediction and metric outputs.",
    )
    parser.add_argument("--model-name", default="spatialself_wavelet_scale")
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--diffusion-steps", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--lr-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--positive-quantile", type=float, default=0.9)
    parser.add_argument("--negative-quantile", type=float, default=0.2)
    parser.add_argument("--pseudo-label-strategy", default="mechanism_niche_expression_distance")
    parser.add_argument("--pseudo-label-flip-fraction", type=float, default=0.0)
    parser.add_argument("--soft-labels", action="store_true")
    parser.add_argument("--weak-positive-loss-weight", type=float, default=1.0)
    parser.add_argument("--weak-positive-expression-quantile", type=float, default=0.7)
    parser.add_argument(
        "--train-sampling-strategy",
        choices=("balanced_global", "balanced_per_lr"),
        default="balanced_per_lr",
    )
    parser.add_argument("--focal-loss-gamma", type=float, default=0.0)
    parser.add_argument("--focal-loss-alpha", type=float, default=0.5)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--feasibility-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.05)
    parser.add_argument("--context-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--context-contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--context-contrastive-negative-mode",
        choices=("random_receiver", "hard_receiver"),
        default="random_receiver",
    )
    parser.add_argument("--structural-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--structural-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--lr-identity-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr-identity-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--scale-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--scale-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--wavelet-scale-alignment-loss-weight", type=float, default=0.0)
    parser.add_argument("--scale-target-loss-weight", type=float, default=None)
    parser.add_argument("--scale-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--scale-responsibility-loss-weight", type=float, default=None)
    parser.add_argument("--scale-responsibility-margin", type=float, default=0.35)
    parser.add_argument("--scale-responsibility-edges", type=int, default=2048)
    parser.add_argument(
        "--lr-scale-target-strategy",
        choices=("ranklocal", "rankdist", "distance", "pseudo_ranklocal", "pseudo_rankdist", "pseudo_distance"),
        default="ranklocal",
    )
    parser.add_argument("--control-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--control-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--null-candidate-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatial-null-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatial-null-contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--spatial-null-negative-mode",
        choices=("random_receiver", "range_violation_receiver"),
        default="random_receiver",
    )
    parser.add_argument("--mechanism-guard-loss-weight", type=float, default=0.03)
    parser.add_argument("--background-guard-loss-weight", type=float, default=0.0)
    parser.add_argument("--background-guard-target-path", default="")
    parser.add_argument("--evaluation-truth-path", default="")
    parser.add_argument("--lr-cap-positive", type=int, default=50)
    parser.add_argument("--lr-cap-negative", type=int, default=200)
    parser.add_argument("--no-lr-embedding", action="store_true")
    parser.add_argument(
        "--lr-encoding-mode",
        choices=("learned", "fixed_random", "one_hot"),
        default="learned",
    )
    parser.add_argument("--no-distance", action="store_true")
    parser.add_argument("--no-spatial-graph", action="store_true")
    parser.add_argument("--no-expression-features", action="store_true")
    parser.add_argument("--no-cell-type-features", action="store_true")
    parser.add_argument("--no-niche-features", action="store_true")
    parser.add_argument("--no-coordinate-features", action="store_true")
    parser.add_argument("--use-edge-prior-features", dest="use_edge_prior_features", action="store_true", default=True)
    parser.add_argument("--no-edge-prior-features", dest="use_edge_prior_features", action="store_false")
    parser.add_argument("--fixed-uniform-range", action="store_true")
    parser.add_argument("--no-context-gate", action="store_true")
    parser.add_argument("--n-range-basis", type=int, default=6)
    parser.add_argument("--complex-expression-mode", choices=("pairwise", "geometric_mean"), default="pairwise")
    parser.add_argument("--complex-min-subunit-expression", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    output_dir = Path(args.output_dir)
    metrics = [
        run_spatialself_on_replicate(
            replicate_dir,
            output_dir,
            model_name=args.model_name,
            k_nn=args.k_nn,
            diffusion_steps=tuple(args.diffusion_steps),
            h_dim=args.h_dim,
            lr_dim=args.lr_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            bce_loss_weight=args.bce_loss_weight,
            positive_quantile=args.positive_quantile,
            negative_quantile=args.negative_quantile,
            pseudo_label_strategy=args.pseudo_label_strategy,
            pseudo_label_flip_fraction=args.pseudo_label_flip_fraction,
            soft_labels=args.soft_labels,
            weak_positive_loss_weight=args.weak_positive_loss_weight,
            weak_positive_expression_quantile=args.weak_positive_expression_quantile,
            train_sampling_strategy=args.train_sampling_strategy,
            focal_loss_gamma=args.focal_loss_gamma,
            focal_loss_alpha=args.focal_loss_alpha,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            feasibility_loss_weight=args.feasibility_loss_weight,
            hard_negative_loss_weight=args.hard_negative_loss_weight,
            context_contrastive_loss_weight=args.context_contrastive_loss_weight,
            context_contrastive_margin=args.context_contrastive_margin,
            context_contrastive_negative_mode=args.context_contrastive_negative_mode,
            structural_contrastive_loss_weight=args.structural_contrastive_loss_weight,
            structural_contrastive_margin=args.structural_contrastive_margin,
            lr_identity_contrastive_loss_weight=args.lr_identity_contrastive_loss_weight,
            lr_identity_contrastive_margin=args.lr_identity_contrastive_margin,
            scale_contrastive_loss_weight=args.scale_contrastive_loss_weight,
            scale_contrastive_margin=args.scale_contrastive_margin,
            wavelet_scale_alignment_loss_weight=args.wavelet_scale_alignment_loss_weight,
            scale_target_loss_weight=args.scale_target_loss_weight,
            scale_diversity_loss_weight=args.scale_diversity_loss_weight,
            scale_responsibility_loss_weight=args.scale_responsibility_loss_weight,
            scale_responsibility_margin=args.scale_responsibility_margin,
            scale_responsibility_edges=args.scale_responsibility_edges,
            lr_scale_target_strategy=args.lr_scale_target_strategy,
            control_contrastive_loss_weight=args.control_contrastive_loss_weight,
            control_contrastive_margin=args.control_contrastive_margin,
            null_candidate_loss_weight=args.null_candidate_loss_weight,
            spatial_null_contrastive_loss_weight=args.spatial_null_contrastive_loss_weight,
            spatial_null_contrastive_margin=args.spatial_null_contrastive_margin,
            spatial_null_negative_mode=args.spatial_null_negative_mode,
            mechanism_guard_loss_weight=args.mechanism_guard_loss_weight,
            background_guard_loss_weight=args.background_guard_loss_weight,
            background_guard_target_path=args.background_guard_target_path,
            evaluation_truth_path=args.evaluation_truth_path,
            lr_cap_positive=args.lr_cap_positive,
            lr_cap_negative=args.lr_cap_negative,
            use_lr_embedding=not args.no_lr_embedding,
            lr_encoding_mode=args.lr_encoding_mode,
            use_distance=not args.no_distance,
            use_spatial_graph=not args.no_spatial_graph,
            use_expression_features=not args.no_expression_features,
            use_cell_type_features=not args.no_cell_type_features,
            use_niche_features=not args.no_niche_features,
            use_coordinate_features=not args.no_coordinate_features,
            use_edge_prior_features=args.use_edge_prior_features,
            fixed_uniform_range=args.fixed_uniform_range,
            use_context_gate=not args.no_context_gate,
            n_range_basis=args.n_range_basis,
        complex_expression_mode=args.complex_expression_mode,
        complex_min_subunit_expression=args.complex_min_subunit_expression,
            max_train_samples=args.max_train_samples,
            seed=args.seed,
            device=args.device,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / f"{args.model_name}_metrics.csv", index=False)
    print(
        metrics_table[
            [
                "replicate",
                "method",
                "auroc",
                "aupr",
                "top_k_precision",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote SpatialSelf outputs to {output_dir}")


if __name__ == "__main__":
    main()
