#!/usr/bin/env python
"""Build cell-type-pair CCC modules for the fresh high-resolution tonsil subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal4-dir",
        default="outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset",
    )
    parser.add_argument("--operating-quantile", type=float, default=0.94)
    parser.add_argument("--cluster-distance-threshold", type=float, default=0.82)
    parser.add_argument("--fallback-n-modules", type=int, default=4)
    args = parser.parse_args()

    goal4_dir = PROJECT_ROOT / args.goal4_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(goal4_dir / "constructed_replicate/cells.csv")
    candidates = pd.read_csv(goal4_dir / "constructed_replicate/candidate_interactions.csv")
    lr_pairs = pd.read_csv(goal4_dir / "constructed_replicate/lr_pairs.csv")
    predictions = pd.read_csv(goal4_dir / "tonsil_fresh_spatialself_wavelet_predictions.csv")
    table = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    table["cell_type_pair"] = table["sender_cell_type"].astype(str) + "->" + table["receiver_cell_type"].astype(str)
    table["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in table.itertuples(index=False)]

    lr_activity = _lr_activity_profiles(table, lr_pairs, operating_quantile=float(args.operating_quantile))
    lr_activity = _assign_modules(
        lr_activity,
        threshold=float(args.cluster_distance_threshold),
        fallback_n_modules=int(args.fallback_n_modules),
    )
    module_summary = _module_summary(lr_activity, table, cells, operating_quantile=float(args.operating_quantile))
    pair_summary = _module_pair_summary(lr_activity)
    enrichment = _module_control_enrichment(lr_activity)
    overview = _overview(module_summary, lr_activity, args)

    lr_activity.to_csv(output_dir / "tonsil_ccc_module_lr_activity.csv", index=False)
    module_summary.to_csv(output_dir / "tonsil_ccc_module_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "tonsil_ccc_module_cell_type_pair_summary.csv", index=False)
    enrichment.to_csv(output_dir / "tonsil_ccc_module_control_enrichment.csv", index=False)
    pd.DataFrame([overview]).to_csv(output_dir / "tonsil_ccc_module_overview.csv", index=False)
    (output_dir / "tonsil_ccc_module_manifest.json").write_text(
        json.dumps(
            {
                "goal_id": "4",
                "analysis_label": "tonsil_ccc_modules_fresh_subset",
                "evidence_status": "module_validation_initial",
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "goal4_dir": args.goal4_dir,
                "output_dir": args.output_dir,
                "operating_quantile": float(args.operating_quantile),
                "cluster_distance_threshold": float(args.cluster_distance_threshold),
                "fallback_n_modules": int(args.fallback_n_modules),
                "analysis_scope": (
                    "High-resolution tonsil cell-type-pair CCC module diagnostic; no ground truth and "
                    "single 720-cell subset, so this is biological-structure support rather than final validation."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([overview]).to_string(index=False))
    print(module_summary.head(12).to_string(index=False))


def _lr_activity_profiles(table: pd.DataFrame, lr_pairs: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(table["score"].quantile(operating_quantile))
    top = table[table["score"] >= threshold].copy()
    all_pairs = sorted(table["cell_type_pair"].unique())
    top_counts = (
        top.groupby(["lr_id", "cell_type_pair"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=all_pairs, fill_value=0)
    )
    mean_scores = (
        table.groupby(["lr_id", "cell_type_pair"], sort=False)["score"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=all_pairs, fill_value=0.0)
    )
    top_fraction = top_counts.div(top_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    score_scaled = mean_scores.apply(_minmax, axis=1, result_type="expand")
    profile = 0.55 * top_fraction + 0.45 * score_scaled
    profile.columns = [f"profile__{column}" for column in profile.columns]

    lr_meta = lr_pairs.set_index("lr_id")[
        ["ligand", "receptor", "mechanism", "control_class", "spatial_range", "active_niche"]
    ].copy()
    out = lr_meta.join(profile, how="left").fillna(0.0).reset_index()
    out["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in out.itertuples(index=False)]
    out["n_top_edges"] = out["lr_id"].map(top_counts.sum(axis=1)).fillna(0).astype(int)
    out["dominant_cell_type_pair"] = [
        _dominant_profile_pair(row, all_pairs) for row in out.itertuples(index=False)
    ]
    out["cell_type_pair_concentration"] = [
        _profile_concentration(row, all_pairs) for row in out.itertuples(index=False)
    ]
    return out


def _assign_modules(activity: pd.DataFrame, *, threshold: float, fallback_n_modules: int) -> pd.DataFrame:
    profile_cols = [column for column in activity.columns if column.startswith("profile__")]
    if len(activity) <= 2 or not profile_cols:
        result = activity.copy()
        result["ccc_module"] = 1
        return result
    matrix = activity[profile_cols].to_numpy(dtype=float)
    if np.allclose(matrix, 0.0):
        result = activity.copy()
        result["ccc_module"] = 1
        return result
    distance = pdist(matrix, metric="cosine")
    distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0)
    tree = linkage(distance, method="average")
    clusters = fcluster(tree, t=float(threshold), criterion="distance")
    if len(set(clusters.tolist())) <= 1 and int(fallback_n_modules) > 1 and len(activity) >= int(fallback_n_modules):
        clusters = fcluster(tree, t=int(fallback_n_modules), criterion="maxclust")
    result = activity.copy()
    result["ccc_module"] = clusters.astype(int)
    return result


def _module_summary(activity: pd.DataFrame, table: pd.DataFrame, cells: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(table["score"].quantile(operating_quantile))
    top = table[table["score"] >= threshold].copy()
    cell_dispersion = _coordinate_dispersion(cells[["x", "y"]].to_numpy(dtype=float))
    rows = []
    for module_id, group in activity.groupby("ccc_module", sort=True):
        lr_ids = set(group["lr_id"].astype(str))
        top_group = top[top["lr_id"].astype(str).isin(lr_ids)]
        pair_counts = top_group["cell_type_pair"].value_counts()
        mechanisms = group["mechanism"].astype(str).value_counts()
        controls = group["control_class"].astype(str).value_counts()
        receiver_dispersion = _receiver_dispersion(top_group, cells)
        rows.append(
            {
                "ccc_module": int(module_id),
                "n_lr_pairs": int(len(group)),
                "n_top_edges": int(len(top_group)),
                "dominant_mechanism": mechanisms.index[0] if len(mechanisms) else "",
                "dominant_mechanism_fraction": float(mechanisms.iloc[0] / len(group)) if len(mechanisms) else 0.0,
                "n_positive_controls": int((group["control_class"] == "tonsil_positive_prior").sum()),
                "n_housekeeping_negative_controls": int((group["control_class"] == "housekeeping_negative_control").sum()),
                "dominant_cell_type_pair": pair_counts.index[0] if len(pair_counts) else "",
                "dominant_cell_type_pair_fraction": float(pair_counts.iloc[0] / len(top_group)) if len(pair_counts) else 0.0,
                "cell_type_pair_concentration_mean": float(group["cell_type_pair_concentration"].mean()),
                "receiver_spatial_concentration": float(1.0 - receiver_dispersion / cell_dispersion)
                if cell_dispersion > 0 and math.isfinite(receiver_dispersion)
                else 0.0,
                "top_lr_pairs": ";".join(
                    group.sort_values("n_top_edges", ascending=False)["canonical_pair"].head(8).astype(str)
                ),
                "control_classes": ";".join(f"{key}:{value}" for key, value in controls.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_positive_controls", "n_top_edges"], ascending=[False, False])


def _module_pair_summary(activity: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [column for column in activity.columns if column.startswith("profile__")]
    rows = []
    for module_id, group in activity.groupby("ccc_module", sort=True):
        means = group[profile_cols].mean(axis=0).sort_values(ascending=False)
        for rank, (column, value) in enumerate(means.head(12).items(), start=1):
            rows.append(
                {
                    "ccc_module": int(module_id),
                    "rank": int(rank),
                    "cell_type_pair": column.removeprefix("profile__"),
                    "module_profile_mean": float(value),
                }
            )
    return pd.DataFrame(rows)


def _module_control_enrichment(activity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = ["tonsil_positive_prior", "housekeeping_negative_control", "background_lr"]
    for module_id, group in activity.groupby("ccc_module", sort=True):
        in_module = activity["ccc_module"] == module_id
        for label in labels:
            has_label = activity["control_class"].astype(str) == label
            table = [
                [int((in_module & has_label).sum()), int((in_module & ~has_label).sum())],
                [int((~in_module & has_label).sum()), int((~in_module & ~has_label).sum())],
            ]
            odds, p_value = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "ccc_module": int(module_id),
                    "control_class": label,
                    "n_in_module": table[0][0],
                    "n_out_module": table[1][0],
                    "odds_ratio": float(odds),
                    "p_value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    result["bh_fdr"] = _bh_adjust(result["p_value"].to_numpy(dtype=float)) if not result.empty else []
    return result.sort_values(["bh_fdr", "p_value", "ccc_module"]).reset_index(drop=True)


def _overview(module_summary: pd.DataFrame, activity: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    positive_modules = module_summary[module_summary["n_positive_controls"] > 0]
    negative_modules = module_summary[module_summary["n_housekeeping_negative_controls"] > 0]
    best_positive = (
        positive_modules.sort_values(["n_positive_controls", "receiver_spatial_concentration"], ascending=[False, False])
        .iloc[0]
        .to_dict()
        if not positive_modules.empty
        else {}
    )
    return {
        "goal_id": "4",
        "analysis_label": "tonsil_ccc_modules_fresh_subset",
        "evidence_status": "module_validation_initial",
        "n_lr_pairs": int(len(activity)),
        "n_modules": int(module_summary["ccc_module"].nunique()) if not module_summary.empty else 0,
        "n_modules_with_positive_controls": int(len(positive_modules)),
        "n_modules_with_housekeeping_negatives": int(len(negative_modules)),
        "best_positive_module": int(best_positive.get("ccc_module", 0)) if best_positive else 0,
        "best_positive_module_positive_controls": int(best_positive.get("n_positive_controls", 0)) if best_positive else 0,
        "best_positive_module_dominant_pair": best_positive.get("dominant_cell_type_pair", ""),
        "best_positive_module_spatial_concentration": float(best_positive.get("receiver_spatial_concentration", math.nan))
        if best_positive
        else math.nan,
        "mean_module_cell_type_pair_concentration": float(module_summary["cell_type_pair_concentration_mean"].mean())
        if not module_summary.empty
        else math.nan,
        "operating_quantile": float(args.operating_quantile),
        "cluster_distance_threshold": float(args.cluster_distance_threshold),
        "interpretation": (
            "Initial high-resolution CCC module diagnostic; supports module-level structure if positive-control "
            "modules are spatially/cell-type concentrated, but remains single-subset and not ground truth."
        ),
    }


def _canonical_pair(ligand: str, receptor: str) -> str:
    return f"{ligand}-{receptor}"


def _dominant_profile_pair(row: object, all_pairs: list[str]) -> str:
    values = {pair: float(getattr(row, f"profile__{pair}", 0.0)) for pair in all_pairs}
    if not values:
        return ""
    return max(values, key=values.get)


def _profile_concentration(row: object, all_pairs: list[str]) -> float:
    values = np.array([float(getattr(row, f"profile__{pair}", 0.0)) for pair in all_pairs], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values / total
    entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    max_entropy = math.log(len(values)) if len(values) > 1 else 1.0
    return float(1.0 - entropy / max_entropy) if max_entropy > 0 else 0.0


def _receiver_dispersion(top_group: pd.DataFrame, cells: pd.DataFrame) -> float:
    if top_group.empty:
        return float("nan")
    coords = cells.set_index("cell_id").loc[top_group["receiver_id"].drop_duplicates(), ["x", "y"]].to_numpy(dtype=float)
    return _coordinate_dispersion(coords)


def _coordinate_dispersion(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if len(coords) <= 1:
        return 0.0
    center = coords.mean(axis=0)
    return float(np.sqrt(((coords - center) ** 2).sum(axis=1)).mean())


def _minmax(row: pd.Series) -> pd.Series:
    values = row.to_numpy(dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return pd.Series(np.zeros_like(values), index=row.index)
    return pd.Series((values - low) / (high - low), index=row.index)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


if __name__ == "__main__":
    main()
