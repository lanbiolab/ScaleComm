#!/usr/bin/env python
"""Evaluate non-oracle LR scale targets before training ScaleGate variants.

The targets use only candidate expression and distance profiles. Benchmark
range classes are used only for audit metrics, not for target construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.current.run_scale_gate_forcing_experiment import (
    _pseudo_distance_scale_targets,
    _pseudo_hybrid_scale_targets,
    _pseudo_profile_scale_targets,
    _separation_metrics,
)


BENCH_200 = PROJECT_ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = PROJECT_ROOT / "outputs/current_fig5_lr_scale_mechanism/scale_target_quality_sweep"
SOURCE = OUT / "source_data"

DATASETS = {
    "lymph_scdesign3_200lr": BENCH_200 / "datasets/reference_lymph_node_scdesign3_200lr_replicate_00",
    "melanoma_scdesign3_200lr": BENCH_200 / "datasets/reference_melanoma_scdesign3_200lr_replicate_00",
}
RANGE_ORDER = ["contact", "short", "medium", "long_niche"]


def expression_distance_centers(
    candidates: pd.DataFrame,
    *,
    quantile: float,
    power: float,
) -> dict[str, float]:
    ligand = np.log1p(candidates["ligand_expr"].to_numpy(float))
    receptor = np.log1p(candidates["receptor_expr"].to_numpy(float))
    expr = np.nan_to_num(ligand * receptor, nan=0.0, posinf=0.0, neginf=0.0)
    dist = candidates["distance"].to_numpy(float)
    d95 = max(float(np.quantile(dist, 0.95)), 1e-6)
    out = {}
    for lr_id, group in candidates.groupby("lr_id", sort=False):
        idx = group.index.to_numpy()
        e = expr[idx]
        d = np.clip(dist[idx] / d95, 0.0, 1.0)
        if len(idx) < 5 or np.nanmax(e) <= 0:
            out[str(lr_id)] = 0.5
            continue
        q = np.quantile(e, quantile)
        w = np.clip(e - q, 0.0, None)
        if w.sum() <= 1e-8:
            keep = e >= np.quantile(e, max(0.55, quantile - 0.2))
            w = np.where(keep, np.maximum(e, 1e-6), 0.0)
        w = np.power(np.maximum(w, 1e-8), power)
        out[str(lr_id)] = float(np.average(d, weights=w))
    return out


def centers_to_targets(
    centers_by_lr: dict[str, float],
    *,
    n_scales: int,
    width: float,
    stretch: float = 1.0,
    rank_transform: bool = False,
) -> dict[str, np.ndarray]:
    ids = list(centers_by_lr)
    centers = np.array([centers_by_lr[lr_id] for lr_id in ids], dtype=float)
    if rank_transform and len(centers) > 1:
        order = np.argsort(np.argsort(centers))
        centers = order / max(len(centers) - 1, 1)
    centers = np.clip(0.5 + float(stretch) * (centers - 0.5), 0.0, 1.0)
    scale_centers = np.linspace(0.0, 1.0, n_scales)
    out = {}
    for lr_id, center in zip(ids, centers, strict=False):
        target = np.exp(-0.5 * ((scale_centers - center) / float(width)) ** 2) + 0.02
        out[str(lr_id)] = target / target.sum()
    return out


def profile_peak_targets(candidates: pd.DataFrame, *, n_scales: int, sharpen: float) -> dict[str, np.ndarray]:
    base = _pseudo_profile_scale_targets(candidates, n_scales=n_scales)
    out = {}
    for lr_id, vec in base.items():
        v = np.power(np.clip(np.asarray(vec, dtype=float), 1e-8, None), sharpen)
        out[str(lr_id)] = v / v.sum()
    return out


def blend_targets(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, weight_b: float, power: float = 1.0) -> dict[str, np.ndarray]:
    out = {}
    for lr_id in sorted(set(a) | set(b)):
        va = np.asarray(a.get(lr_id, np.full(6, 1 / 6)), dtype=float)
        vb = np.asarray(b.get(lr_id, np.full(len(va), 1 / len(va))), dtype=float)
        target = (1.0 - weight_b) * (va / va.sum()) + weight_b * (vb / vb.sum())
        target = np.power(np.clip(target, 1e-8, None), power)
        out[str(lr_id)] = target / target.sum()
    return out


def make_targets(candidates: pd.DataFrame, *, n_scales: int) -> dict[str, dict[str, np.ndarray]]:
    targets: dict[str, dict[str, np.ndarray]] = {
        "existing_distance": _pseudo_distance_scale_targets(candidates, n_scales=n_scales),
        "existing_profile": _pseudo_profile_scale_targets(candidates, n_scales=n_scales),
        "existing_hybrid": _pseudo_hybrid_scale_targets(candidates, n_scales=n_scales),
    }
    for q in (0.60, 0.70, 0.80, 0.88):
        for power in (1.0, 1.5, 2.0):
            centers = expression_distance_centers(candidates, quantile=q, power=power)
            for width in (0.12, 0.16, 0.20):
                for stretch in (1.0, 1.5, 2.0):
                    key = f"dist_q{int(q*100)}_p{power:g}_w{width:g}_s{stretch:g}"
                    targets[key] = centers_to_targets(centers, n_scales=n_scales, width=width, stretch=stretch)
            targets[f"rankdist_q{int(q*100)}_p{power:g}_w0.16"] = centers_to_targets(
                centers,
                n_scales=n_scales,
                width=0.16,
                rank_transform=True,
            )
    profile_sharp = profile_peak_targets(candidates, n_scales=n_scales, sharpen=1.8)
    targets["profile_sharp"] = profile_sharp
    for weight in (0.15, 0.30, 0.45):
        targets[f"distance_profile_blend{weight:g}"] = blend_targets(
            targets["existing_distance"],
            targets["existing_profile"],
            weight_b=weight,
            power=1.15,
        )
        targets[f"rankdistance_profile_blend{weight:g}"] = blend_targets(
            targets["rankdist_q80_p1.5_w0.16"],
            targets["existing_profile"],
            weight_b=weight,
            power=1.15,
        )
    return targets


def evaluate_target(
    *,
    dataset_id: str,
    target_name: str,
    target_by_lr: dict[str, np.ndarray],
    lr_pairs: pd.DataFrame,
    lr_design: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    class_map = lr_design.set_index("lr_id")["range_class"].to_dict()
    rows = []
    scale_index = np.arange(next(iter(target_by_lr.values())).shape[0], dtype=float)
    for lr_id in lr_pairs["lr_id"].astype(str):
        vec = np.asarray(target_by_lr.get(str(lr_id)), dtype=float)
        row = {
            "dataset_id": dataset_id,
            "target_name": target_name,
            "lr_id": str(lr_id),
            "range_class": class_map.get(str(lr_id), "unknown"),
            "expected_scale": float((vec * scale_index).sum()),
            "local_weight": float(vec[:2].sum()),
            "broad_weight": float(vec[-2:].sum()),
        }
        for i, value in enumerate(vec):
            row[f"scale_{i}"] = float(value)
        rows.append(row)
    lr_scale = pd.DataFrame(rows)
    lr_scale = lr_scale[lr_scale["range_class"].isin(RANGE_ORDER)].copy()
    metrics = _separation_metrics(lr_scale)
    class_mean = lr_scale.groupby("range_class")["expected_scale"].mean()
    metrics.update(
        {
            "dataset_id": dataset_id,
            "target_name": target_name,
            "contact_mean_scale": float(class_mean.get("contact", np.nan)),
            "short_mean_scale": float(class_mean.get("short", np.nan)),
            "medium_mean_scale": float(class_mean.get("medium", np.nan)),
            "long_mean_scale": float(class_mean.get("long_niche", np.nan)),
        }
    )
    return lr_scale, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--n-scales", type=int, default=6)
    args = parser.parse_args()

    SOURCE.mkdir(parents=True, exist_ok=True)
    lr_rows = []
    metric_rows = []
    for dataset_id in args.datasets:
        replicate_dir = DATASETS[dataset_id]
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        lr_design = pd.read_csv(replicate_dir / "lr_panel_design.csv")
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        targets = make_targets(candidates, n_scales=args.n_scales)
        for target_name, target_by_lr in targets.items():
            lr_scale, metrics = evaluate_target(
                dataset_id=dataset_id,
                target_name=target_name,
                target_by_lr=target_by_lr,
                lr_pairs=lr_pairs,
                lr_design=lr_design,
            )
            lr_rows.append(lr_scale)
            metric_rows.append(metrics)

    lr_table = pd.concat(lr_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    overall = (
        metrics.groupby("target_name", as_index=False)
        .agg(
            scale_eta2=("scale_eta2", "mean"),
            scale_spearman_range_order=("scale_spearman_range_order", "mean"),
            long_minus_contact_expected_scale=("long_minus_contact_expected_scale", "mean"),
            contact_mean_scale=("contact_mean_scale", "mean"),
            short_mean_scale=("short_mean_scale", "mean"),
            medium_mean_scale=("medium_mean_scale", "mean"),
            long_mean_scale=("long_mean_scale", "mean"),
        )
        .sort_values(["scale_eta2", "scale_spearman_range_order", "long_minus_contact_expected_scale"], ascending=False)
    )
    lr_table.to_csv(SOURCE / "scale_target_quality_lr_scale.csv", index=False)
    metrics.to_csv(SOURCE / "scale_target_quality_by_dataset.csv", index=False)
    overall.to_csv(SOURCE / "scale_target_quality_overall.csv", index=False)
    with (OUT / "scale_target_quality_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "purpose": "Non-oracle scale-target quality audit; range_class is used only for evaluation.",
                "datasets": args.datasets,
                "n_scales": args.n_scales,
                "source_data": {
                    "lr_scale": "source_data/scale_target_quality_lr_scale.csv",
                    "by_dataset": "source_data/scale_target_quality_by_dataset.csv",
                    "overall": "source_data/scale_target_quality_overall.csv",
                },
            },
            fh,
            indent=2,
        )
    print(overall.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {SOURCE}")


if __name__ == "__main__":
    main()
