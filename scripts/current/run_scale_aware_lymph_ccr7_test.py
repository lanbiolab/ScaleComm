#!/usr/bin/env python
"""Run the scale-aware diagnostic SpatialSelf on the lymph-node CCR7 panel.

This uses the rank-local scale target that separated LR spatial scales in the
Fig. 5 diagnostic. It is intentionally written as an experiment script so it
does not overwrite the frozen benchmark baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import run_spatialself as rs
from scripts.current import run_scale_gate_forcing_experiment as sg
from src.spitialself.evaluation import evaluate_predictions_auto


DATA_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/datasets/human_lymph_node_visium_phase6_expanded_replicate_00"
OUT_DIR = PROJECT_ROOT / "outputs/current_lymph_node_ccr7_scale_baseline"
PAIR = ("CCL19", "CCR7")
PAIR_LABEL = "CCL19-CCR7"


def mechanism_to_range_class(mechanism: str) -> str:
    mechanism = str(mechanism)
    if mechanism == "juxtacrine_contact":
        return "contact"
    if mechanism == "short_range_paracrine":
        return "short"
    if mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        return "medium"
    if mechanism == "long_range_chemokine_niche":
        return "long_niche"
    return "background"


def train_scale_aware(args: argparse.Namespace) -> dict[str, Path]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source_data"
    source.mkdir(exist_ok=True)

    cells = pd.read_csv(DATA_DIR / "cells.csv")
    expression = pd.read_csv(DATA_DIR / "expression.csv")
    lr_pairs = pd.read_csv(DATA_DIR / "lr_pairs.csv")
    candidates = pd.read_csv(DATA_DIR / "candidate_interactions.csv")
    lr_design = lr_pairs[["lr_id", "mechanism", "true_lr", "weak_signal", "false_positive_high_expression"]].copy()
    lr_design["range_class"] = lr_design["mechanism"].map(mechanism_to_range_class)

    data, tensors = rs._build_training_data(
        cells,
        expression,
        lr_pairs,
        candidates,
        k_nn=args.k_nn,
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

    model = sg.ScaleMixtureWaveletModel(
        in_dim=data.x.shape[1],
        h_dim=args.h_dim,
        lr_dim=args.lr_dim,
        n_lr=len(lr_pairs),
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        edge_feature_dim=int(tensors["edge_features"].shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_idx = tensors["train_idx"]
    labels = tensors["pseudo_label"][train_idx]
    sample_weight = tensors["sample_weight"][train_idx]
    aux_target = sg._scale_target(args.variant, lr_pairs, lr_design, candidates, n_scales=model.node_encoder.n_scales)

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
        loss = loss + sg._guard_losses(model, h, tensors)

        lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
        pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
        target = aux_target.to(device=h.device, dtype=pred_scale.dtype)
        loss = loss + args.scale_target_weight * sg._kl(pred_scale, target)
        loss = loss + args.diversity_weight * sg._diversity_loss(pred_scale)
        loss = loss + args.responsibility_weight * sg._scale_responsibility_loss(
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
            print(f"epoch {epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")

    model.eval()
    with torch.no_grad():
        h = model.encode_nodes(data)
        scores = []
        for start in range(0, len(candidates), args.score_batch_size):
            end = min(start + args.score_batch_size, len(candidates))
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
        {
            "candidate_id": candidates["candidate_id"],
            "method": f"spatialself_scale_{args.variant}",
            "score": np.concatenate(scores),
        }
    )
    prediction_path = out / f"human_lymph_node_visium_phase6_expanded__spatialself_scale_{args.variant}_predictions.csv"
    prediction.to_csv(prediction_path, index=False)

    lr_scale = build_lr_scale_table(lr_pairs, lr_design, scale, aux_target.numpy(), args.variant)
    lr_scale_path = source / "lr_scale_distribution.csv"
    lr_scale.to_csv(lr_scale_path, index=False)

    lr_rank = build_lr_rank_table(candidates, prediction)
    lr_rank_path = source / "lr_rank_summary.csv"
    lr_rank.to_csv(lr_rank_path, index=False)

    receiver = build_receiver_summary(cells, candidates, prediction)
    receiver_path = source / "ccl19_ccr7_receiver_summary.csv"
    receiver.to_csv(receiver_path, index=False)

    ccl19_summary = summarize_ccl19(lr_rank, receiver, lr_scale)
    ccl19_path = source / "ccl19_ccr7_scale_baseline_summary.csv"
    pd.DataFrame([ccl19_summary]).to_csv(ccl19_path, index=False)

    metrics = evaluate_predictions_auto(candidates, prediction)
    metrics.update(ccl19_summary)
    metrics.update(
        {
            "variant": args.variant,
            "epochs": int(args.epochs),
            "k_nn": int(args.k_nn),
            "max_train_samples": int(args.max_train_samples),
            "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        }
    )
    metrics_path = out / "scale_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest_path = out / "scale_baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "purpose": "Run scale-aware diagnostic SpatialSelf on the lymph-node CCL19-CCR7 positive-control panel.",
                "dataset": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                "variant": args.variant,
                "outputs": {
                    "predictions": str(prediction_path.relative_to(PROJECT_ROOT)),
                    "lr_scale": str(lr_scale_path.relative_to(PROJECT_ROOT)),
                    "lr_rank": str(lr_rank_path.relative_to(PROJECT_ROOT)),
                    "receiver_summary": str(receiver_path.relative_to(PROJECT_ROOT)),
                    "ccl19_summary": str(ccl19_path.relative_to(PROJECT_ROOT)),
                    "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
                },
                "claim_boundary": "Diagnostic scale-aware candidate baseline; not yet the frozen official baseline.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return {
        "prediction": prediction_path,
        "lr_scale": lr_scale_path,
        "lr_rank": lr_rank_path,
        "receiver": receiver_path,
        "summary": ccl19_path,
        "metrics": metrics_path,
        "manifest": manifest_path,
    }


def build_lr_scale_table(
    lr_pairs: pd.DataFrame,
    lr_design: pd.DataFrame,
    scale: np.ndarray,
    target: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    design = lr_design.set_index("lr_id")
    rows = []
    idx = np.arange(scale.shape[1], dtype=float)
    for i, row in lr_pairs.reset_index(drop=True).iterrows():
        lr_id = str(row["lr_id"])
        out = {
            "variant": variant,
            "lr_id": lr_id,
            "ligand": row["ligand"],
            "receptor": row["receptor"],
            "lr_pair": f"{row['ligand']}-{row['receptor']}",
            "mechanism": row["mechanism"],
            "range_class": design.loc[lr_id, "range_class"] if lr_id in design.index else "unknown",
            "true_lr": bool(row.get("true_lr", False)),
            "weak_signal": bool(row.get("weak_signal", False)),
            "false_positive_high_expression": bool(row.get("false_positive_high_expression", False)),
            "expected_scale": float((scale[i] * idx).sum()),
            "target_expected_scale": float((target[i] * idx).sum()),
            "local_weight": float(scale[i, :2].sum()),
            "broad_weight": float(scale[i, -2:].sum()),
        }
        for j in range(scale.shape[1]):
            out[f"scale_{j}"] = float(scale[i, j])
            out[f"target_scale_{j}"] = float(target[i, j])
        rows.append(out)
    return pd.DataFrame(rows)


def build_lr_rank_table(candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="left")
    table["score"] = table["score"].fillna(0.0)
    grouped = (
        table.groupby(["lr_id", "ligand", "receptor", "mechanism"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            n_edges=("score", "size"),
            n_positive_edges=("score", lambda x: int(np.sum(np.asarray(x) > 0))),
            truth_edges=("ground_truth", "sum"),
            weak_signal=("weak_signal", "max"),
            false_positive_high_expression=("false_positive_high_expression", "max"),
        )
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["lr_pair"] = grouped["ligand"].astype(str) + "-" + grouped["receptor"].astype(str)
    return grouped


def build_receiver_summary(cells: pd.DataFrame, candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="left")
    table = table[table["ligand"].eq(PAIR[0]) & table["receptor"].eq(PAIR[1])].copy()
    table["score"] = table["score"].fillna(0.0)
    incoming = (
        table.groupby("receiver_id", as_index=False)
        .agg(
            incoming_score=("score", "sum"),
            n_edges=("score", "size"),
            n_positive_edges=("score", lambda x: int(np.sum(np.asarray(x) > 0))),
            n_truth_edges=("ground_truth", "sum"),
        )
        .rename(columns={"receiver_id": "cell_id"})
    )
    out = cells.merge(incoming, on="cell_id", how="left")
    for col in ["incoming_score", "n_edges", "n_positive_edges", "n_truth_edges"]:
        out[col] = out[col].fillna(0.0)
    return out


def summarize_ccl19(lr_rank: pd.DataFrame, receiver: pd.DataFrame, lr_scale: pd.DataFrame) -> dict:
    ccl = lr_rank[lr_rank["lr_pair"].eq(PAIR_LABEL)].iloc[0]
    t = receiver[receiver["region"].eq("T cell zone")]["incoming_score"].to_numpy(float)
    o = receiver[~receiver["region"].eq("T cell zone")]["incoming_score"].to_numpy(float)
    top = receiver.sort_values("incoming_score", ascending=False).head(50)
    scale = lr_scale[lr_scale["lr_pair"].eq(PAIR_LABEL)].iloc[0]
    return {
        "ccl19_ccr7_rank": int(ccl["rank"]),
        "ccl19_ccr7_mean_score": float(ccl["mean_score"]),
        "ccl19_ccr7_truth_edges": int(ccl["truth_edges"]),
        "ccl19_ccr7_tzone_mean_incoming": float(np.mean(t)) if len(t) else np.nan,
        "ccl19_ccr7_other_mean_incoming": float(np.mean(o)) if len(o) else np.nan,
        "ccl19_ccr7_tzone_enrichment": float(np.mean(t) / max(np.mean(o), 1e-12)) if len(t) and len(o) else np.nan,
        "ccl19_ccr7_top50_tzone_fraction": float(top["region"].eq("T cell zone").mean()) if len(top) else np.nan,
        "ccl19_ccr7_top_receiver_region": str(receiver.sort_values("incoming_score", ascending=False).iloc[0]["region"]),
        "ccl19_ccr7_expected_scale": float(scale["expected_scale"]),
        "ccl19_ccr7_local_weight": float(scale["local_weight"]),
        "ccl19_ccr7_broad_weight": float(scale["broad_weight"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="mixture_ranklocal_responsibility")
    parser.add_argument("--output-dir", default=str(OUT_DIR / "mixture_ranklocal_responsibility"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--lr-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--scale-target-weight", type=float, default=1.1)
    parser.add_argument("--diversity-weight", type=float, default=0.45)
    parser.add_argument("--responsibility-weight", type=float, default=0.35)
    parser.add_argument("--responsibility-margin", type=float, default=0.35)
    parser.add_argument("--responsibility-edges", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_scale_aware(args)


if __name__ == "__main__":
    main()
