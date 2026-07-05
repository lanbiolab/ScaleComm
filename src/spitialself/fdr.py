"""Permutation p-value and FDR helpers for CCC prediction tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_NULL_MODELS = (
    "score_permutation",
    "spatial_permutation",
    "receiver_permutation",
    "lr_identity_permutation",
    "within_domain_permutation",
    "within_region_permutation",
    "cell_type_preserving_permutation",
)


@dataclass(frozen=True)
class PermutationFDRConfig:
    n_permutations: int = 100
    random_seed: int = 2025
    score_column: str = "score"
    max_quantile_cells: int = 20_000_000


@dataclass(frozen=True)
class FormalFDRPanelConfig:
    """Configuration for a complete CCC FDR output panel.

    This sits above the low-level permutation helpers. It writes the candidate
    edge, LR-pair, cell-type-pair, significant discovery, summary, and manifest
    outputs that the method framework exposes to downstream scripts.
    """

    null_models: tuple[str, ...] = (
        "spatial_permutation",
        "receiver_permutation",
        "lr_identity_permutation",
    )
    n_permutations: int = 100
    random_seed: int = 2025
    score_column: str = "score"
    alpha: float = 0.05
    direct_group_top_fraction: float = 0.1
    panel_metadata: dict = field(default_factory=dict)
    claim_level: str = "formal_fdr_panel_not_manuscript_ready"


def run_formal_fdr_panel(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    candidate_path: Path | str = "",
    prediction_path: Path | str = "",
    config: FormalFDRPanelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full formal CCC FDR panel and write standard output tables."""

    config = config or FormalFDRPanelConfig()
    output_dir = Path(output_dir)
    candidate_path = Path(candidate_path) if candidate_path else Path("")
    prediction_path = Path(prediction_path) if prediction_path else Path("")
    output_dir.mkdir(parents=True, exist_ok=True)
    permutation_config = PermutationFDRConfig(
        n_permutations=int(config.n_permutations),
        random_seed=int(config.random_seed),
        score_column=str(config.score_column),
    )

    summary_rows: list[dict] = []
    group_rows: list[dict] = []
    fdr_paths: dict[str, str] = {}
    prediction_stem = prediction_path.stem if str(prediction_path) else "predictions"
    for null_model in config.null_models:
        edge_table = permutation_fdr(candidates, predictions, null_model=null_model, config=permutation_config)
        edge_path = output_dir / f"{prediction_stem}__{null_model}__candidate_edge_fdr.csv"
        edge_table.to_csv(edge_path, index=False)
        fdr_paths[f"{null_model}:candidate_edge"] = str(edge_path)

        lr_table = permutation_group_fdr(
            candidates,
            predictions,
            null_model=null_model,
            group_columns=["lr_id", "ligand", "receptor", "mechanism"],
            config=permutation_config,
            top_fraction=float(config.direct_group_top_fraction),
        )
        lr_path = output_dir / f"{prediction_stem}__{null_model}__lr_pair_direct_fdr.csv"
        lr_table.to_csv(lr_path, index=False)
        fdr_paths[f"{null_model}:lr_pair"] = str(lr_path)

        cell_type_columns = ["sender_cell_type", "receiver_cell_type", "lr_id", "ligand", "receptor", "mechanism"]
        cell_type_table = permutation_group_fdr(
            candidates,
            predictions,
            null_model=null_model,
            group_columns=cell_type_columns,
            config=permutation_config,
            top_fraction=float(config.direct_group_top_fraction),
        )
        cell_type_path = output_dir / f"{prediction_stem}__{null_model}__cell_type_pair_direct_fdr.csv"
        cell_type_table.to_csv(cell_type_path, index=False)
        fdr_paths[f"{null_model}:cell_type_pair"] = str(cell_type_path)

        significant_edges = edge_table[edge_table["bh_fdr"].astype(float) <= float(config.alpha)].copy()
        sig_edge_path = output_dir / f"{prediction_stem}__{null_model}__significant_spatial_edges.csv"
        significant_edges.to_csv(sig_edge_path, index=False)
        fdr_paths[f"{null_model}:significant_spatial_edges"] = str(sig_edge_path)

        significant_lr = lr_table[lr_table["group_bh_fdr"].astype(float) <= float(config.alpha)].copy()
        sig_lr_path = output_dir / f"{prediction_stem}__{null_model}__significant_lr_pairs.csv"
        significant_lr.to_csv(sig_lr_path, index=False)
        fdr_paths[f"{null_model}:significant_lr_pairs"] = str(sig_lr_path)

        significant_ct = cell_type_table[cell_type_table["group_bh_fdr"].astype(float) <= float(config.alpha)].copy()
        sig_ct_path = output_dir / f"{prediction_stem}__{null_model}__significant_cell_type_pairs.csv"
        significant_ct.to_csv(sig_ct_path, index=False)
        fdr_paths[f"{null_model}:significant_cell_type_pairs"] = str(sig_ct_path)

        summary = summarize_fdr(edge_table, alpha=float(config.alpha))
        summary.update(summarize_fdr_with_truth(candidates, edge_table, alpha=float(config.alpha)))
        summary.update(
            {
                "dataset": config.panel_metadata.get("dataset", ""),
                "replicate": config.panel_metadata.get("replicate", candidate_path.parent.name if str(candidate_path) else ""),
                "method": _method_name(predictions, prediction_path),
                "candidate_path": str(candidate_path),
                "prediction_path": str(prediction_path),
                "candidate_edge_fdr_path": str(edge_path),
                "significant_spatial_edges_path": str(sig_edge_path),
                "n_significant_lr_pairs_direct": int(len(significant_lr)),
                "n_significant_cell_type_pairs_direct": int(len(significant_ct)),
                "lr_pair_direct_fdr_path": str(lr_path),
                "cell_type_pair_direct_fdr_path": str(cell_type_path),
                "n_permutations": int(config.n_permutations),
                "alpha": float(config.alpha),
                "direct_group_top_fraction": float(config.direct_group_top_fraction),
            }
        )
        summary_rows.append(summary)

        group_rows.append(
            _formal_group_summary_row(
                lr_table,
                significant_lr,
                "lr_pair",
                null_model,
                lr_path,
                config.panel_metadata,
                predictions,
                prediction_path,
                float(config.alpha),
            )
        )
        group_rows.append(
            _formal_group_summary_row(
                cell_type_table,
                significant_ct,
                "cell_type_pair",
                null_model,
                cell_type_path,
                config.panel_metadata,
                predictions,
                prediction_path,
                float(config.alpha),
            )
        )

    summary_table = pd.DataFrame(summary_rows)
    group_summary_table = pd.DataFrame(group_rows)
    summary_path = output_dir / "goal5_formal_fdr_summary.csv"
    group_summary_path = output_dir / "goal5_formal_fdr_group_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    group_summary_table.to_csv(group_summary_path, index=False)
    manifest = {
        "candidate_path": str(candidate_path),
        "prediction_path": str(prediction_path),
        "output_dir": str(output_dir),
        "null_models": list(config.null_models),
        "n_permutations": int(config.n_permutations),
        "alpha": float(config.alpha),
        "direct_group_top_fraction": float(config.direct_group_top_fraction),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "fdr_paths": fdr_paths,
        "claim_level": str(config.claim_level),
        "framework_module": "src.spitialself.fdr.run_formal_fdr_panel",
    }
    (output_dir / "goal5_formal_fdr_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary_table, group_summary_table


def permutation_fdr(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    null_model: str,
    config: PermutationFDRConfig | None = None,
) -> pd.DataFrame:
    """Compute candidate-level empirical p-values under a score-permutation null.

    This is a method-agnostic post-processing null: it does not retrain or rerun
    the predictor. Instead, it breaks a chosen biological/spatial association by
    shuffling observed scores within conservative strata. That gives us a common
    FDR schema for SpatialSelf and baselines; model-rerun nulls can be added
    later with the same output columns.
    """

    config = config or PermutationFDRConfig()
    merged = _merge_candidate_predictions(candidates, predictions, config.score_column)
    scores = _clean_scores(merged[config.score_column].to_numpy())
    rng = np.random.default_rng(config.random_seed)

    exceed_count = np.zeros(len(merged), dtype=np.int32)
    null_sum = np.zeros(len(merged), dtype=float)
    null_sum_sq = np.zeros(len(merged), dtype=float)
    keep_quantile_samples = config.n_permutations * len(merged) <= config.max_quantile_cells
    null_q95_samples = (
        np.empty((config.n_permutations, len(merged)), dtype=np.float32)
        if keep_quantile_samples
        else None
    )

    groups = _null_groups(merged, null_model)
    for perm_idx in range(config.n_permutations):
        null_scores = _permute_scores_by_group(scores, groups, rng)
        exceed_count += null_scores >= scores
        null_sum += null_scores
        null_sum_sq += null_scores * null_scores
        if null_q95_samples is not None:
            null_q95_samples[perm_idx, :] = null_scores.astype(np.float32)

    p_values = (exceed_count + 1.0) / (config.n_permutations + 1.0)
    fdr = bh_adjust(p_values)
    null_mean = null_sum / max(config.n_permutations, 1)
    null_var = np.maximum(null_sum_sq / max(config.n_permutations, 1) - null_mean * null_mean, 0.0)

    result = merged[["candidate_id"]].copy()
    if "lr_id" in merged.columns:
        result["lr_id"] = merged["lr_id"]
    result["method"] = merged.get("method", pd.Series(["unknown"] * len(merged))).to_numpy()
    result["null_model"] = null_model
    result["score"] = scores
    result["empirical_p_value"] = p_values
    result["bh_fdr"] = fdr
    result["null_mean"] = null_mean
    result["null_std"] = np.sqrt(null_var)
    if null_q95_samples is None:
        result["null_q95"] = np.nan
    else:
        result["null_q95"] = np.quantile(null_q95_samples, 0.95, axis=0)
    result["n_permutations"] = int(config.n_permutations)
    return result


def _formal_group_summary_row(
    group_table: pd.DataFrame,
    significant_table: pd.DataFrame,
    group_level: str,
    null_model: str,
    group_path: Path,
    panel_metadata: dict,
    predictions: pd.DataFrame,
    prediction_path: Path,
    alpha: float,
) -> dict:
    if group_table.empty:
        min_fdr = float("nan")
        top_group = ""
    else:
        ranked = group_table.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False])
        min_fdr = float(ranked["group_bh_fdr"].min())
        top_group = json.dumps(ranked.iloc[0].to_dict(), ensure_ascii=True, default=str)
    return {
        "dataset": panel_metadata.get("dataset", ""),
        "replicate": panel_metadata.get("replicate", ""),
        "method": _method_name(predictions, prediction_path),
        "null_model": null_model,
        "group_level": group_level,
        "alpha": float(alpha),
        "n_groups": int(len(group_table)),
        "n_significant_groups": int(len(significant_table)),
        "min_group_bh_fdr": min_fdr,
        "top_group": top_group,
        "group_fdr_path": str(group_path),
    }


def _method_name(predictions: pd.DataFrame, prediction_path: Path) -> str:
    if "method" in predictions.columns and predictions["method"].notna().any():
        return str(predictions["method"].dropna().iloc[0])
    name = prediction_path.stem
    for suffix in ("__predictions", "_predictions"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def summarize_fdr(fdr_table: pd.DataFrame, *, alpha: float = 0.05) -> dict:
    """Summarize one candidate-level FDR table."""

    required = {"bh_fdr", "empirical_p_value", "score", "null_model"}
    missing = required - set(fdr_table.columns)
    if missing:
        raise ValueError(f"FDR table missing columns: {sorted(missing)}")

    significant = fdr_table["bh_fdr"].astype(float) <= alpha
    summary = {
        "null_model": str(fdr_table["null_model"].iloc[0]) if len(fdr_table) else "unknown",
        "alpha": float(alpha),
        "n_candidates": int(len(fdr_table)),
        "n_significant_edges": int(significant.sum()),
        "min_p_value": float(fdr_table["empirical_p_value"].min()) if len(fdr_table) else float("nan"),
        "min_bh_fdr": float(fdr_table["bh_fdr"].min()) if len(fdr_table) else float("nan"),
        "score_mean": float(fdr_table["score"].mean()) if len(fdr_table) else float("nan"),
    }
    if "lr_id" in fdr_table.columns:
        summary["n_significant_lr_pairs"] = int(fdr_table.loc[significant, "lr_id"].nunique())
    return summary


def summarize_fdr_with_truth(
    candidates: pd.DataFrame,
    fdr_table: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict:
    """Summarize FDR calibration when reference labels are available."""

    required = {
        "candidate_id",
        "ground_truth",
        "weak_signal",
        "false_positive_high_expression",
    }
    missing = required - set(candidates.columns)
    if missing:
        p_values = fdr_table["empirical_p_value"].astype(float).to_numpy()
        return {
            "truth_diagnostics_available": False,
            "realized_fdr": float("nan"),
            "realized_precision": float("nan"),
            "realized_recall": float("nan"),
            "realized_weak_signal_recall": float("nan"),
            "realized_false_positive_high_expression_call_rate": float("nan"),
            "positive_control_min_p": float("nan"),
            "negative_control_min_p": float("nan"),
            "p_value_ks_uniform_stat": _ks_uniform_statistic(p_values),
            "truth_diagnostics_note": "missing_columns:" + ",".join(sorted(missing)),
        }
    merged = candidates[list(required)].merge(
        fdr_table[["candidate_id", "bh_fdr", "empirical_p_value"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    significant = merged["bh_fdr"].astype(float) <= alpha
    truth = merged["ground_truth"].astype(int).to_numpy()
    weak = merged["weak_signal"].astype(bool).to_numpy()
    fp_high = merged["false_positive_high_expression"].astype(bool).to_numpy()
    sig = significant.to_numpy()
    p_values = merged["empirical_p_value"].astype(float).to_numpy()

    return {
        "truth_diagnostics_available": True,
        "realized_fdr": _safe_divide(int((sig & (truth == 0)).sum()), int(sig.sum())),
        "realized_precision": _safe_divide(int((sig & (truth == 1)).sum()), int(sig.sum())),
        "realized_recall": _safe_divide(int((sig & (truth == 1)).sum()), int((truth == 1).sum())),
        "realized_weak_signal_recall": _safe_divide(
            int((sig & (truth == 1) & weak).sum()), int(((truth == 1) & weak).sum())
        ),
        "realized_false_positive_high_expression_call_rate": _safe_divide(
            int((sig & fp_high).sum()), int(fp_high.sum())
        ),
        "positive_control_min_p": float(np.min(p_values[truth == 1])) if (truth == 1).any() else float("nan"),
        "negative_control_min_p": float(np.min(p_values[truth == 0])) if (truth == 0).any() else float("nan"),
        "p_value_ks_uniform_stat": _ks_uniform_statistic(p_values),
    }


def summarize_group_fdr(
    candidates: pd.DataFrame,
    fdr_table: pd.DataFrame,
    *,
    group_columns: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate candidate-level FDR results to LR or cell-type-pair groups.

    The group p-value uses a Simes combination of candidate-level empirical
    p-values within each group, followed by BH adjustment across groups.
    """

    required_fdr = {"candidate_id", "empirical_p_value", "bh_fdr", "score", "null_model"}
    missing_fdr = required_fdr - set(fdr_table.columns)
    if missing_fdr:
        raise ValueError(f"FDR table missing columns: {sorted(missing_fdr)}")
    missing_group = set(group_columns) - set(candidates.columns)
    if missing_group:
        raise ValueError(f"Candidate table missing group columns: {sorted(missing_group)}")

    fdr_columns = [column for column in fdr_table.columns if column not in group_columns]
    merged = candidates[["candidate_id", *group_columns]].merge(
        fdr_table[fdr_columns],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and FDR table")

    rows = []
    for group_key, group in merged.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        p_values = group["empirical_p_value"].astype(float).to_numpy()
        edge_significant = group["bh_fdr"].astype(float) <= alpha
        row = {column: value for column, value in zip(group_columns, group_key)}
        row.update(
            {
                "null_model": str(group["null_model"].iloc[0]),
                "method": str(group["method"].iloc[0]) if "method" in group.columns else "unknown",
                "n_candidate_edges": int(len(group)),
                "n_significant_edges": int(edge_significant.sum()),
                "mean_score": float(group["score"].astype(float).mean()),
                "max_score": float(group["score"].astype(float).max()),
                "min_edge_p_value": float(np.min(p_values)),
                "group_p_value": _simes_p_value(p_values),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["group_bh_fdr"] = bh_adjust(result["group_p_value"].to_numpy(dtype=float))
    result["group_significant"] = result["group_bh_fdr"] <= alpha
    return result.sort_values(["group_bh_fdr", "group_p_value", "max_score"], ascending=[True, True, False])


def permutation_group_fdr(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    null_model: str,
    group_columns: list[str],
    config: PermutationFDRConfig | None = None,
    top_fraction: float = 0.05,
) -> pd.DataFrame:
    """Compute direct group-level empirical p-values under a permutation null.

    Candidate-edge BH can be overly granular for CCC inference because the
    manuscript-level claim is usually an LR-pair, cell-type-pair, or pathway
    program. This test aggregates the top-scoring tail within each group, then
    compares that group statistic with score permutations under the same null.
    """

    config = config or PermutationFDRConfig()
    merged = _merge_candidate_predictions(candidates, predictions, config.score_column)
    missing_group = set(group_columns) - set(merged.columns)
    if missing_group:
        raise ValueError(f"Candidate table missing group columns: {sorted(missing_group)}")

    scores = _clean_scores(merged[config.score_column].to_numpy())
    group_indices, group_keys = _group_indices_with_keys(merged, group_columns)
    observed = np.array(
        [_top_tail_mean(scores[index], top_fraction=top_fraction) for index in group_indices],
        dtype=float,
    )
    null_score_groups = _null_groups(merged, null_model)
    rng = np.random.default_rng(config.random_seed)
    exceed_count = np.zeros(len(group_indices), dtype=np.int32)
    null_sum = np.zeros(len(group_indices), dtype=float)
    null_sum_sq = np.zeros(len(group_indices), dtype=float)

    for _ in range(config.n_permutations):
        null_scores = _permute_scores_by_group(scores, null_score_groups, rng)
        null_stats = np.array(
            [_top_tail_mean(null_scores[index], top_fraction=top_fraction) for index in group_indices],
            dtype=float,
        )
        exceed_count += null_stats >= observed
        null_sum += null_stats
        null_sum_sq += null_stats * null_stats

    p_values = (exceed_count + 1.0) / (config.n_permutations + 1.0)
    result = pd.DataFrame(
        [{column: value for column, value in zip(group_columns, key)} for key in group_keys]
    )
    result["null_model"] = null_model
    result["method"] = (
        str(merged["method"].iloc[0]) if "method" in merged.columns and len(merged) else "unknown"
    )
    result["group_statistic"] = observed
    result["group_p_value"] = p_values
    result["group_bh_fdr"] = bh_adjust(p_values)
    result["group_significant"] = result["group_bh_fdr"] <= 0.05
    result["n_candidate_edges"] = [int(len(index)) for index in group_indices]
    result["null_mean"] = null_sum / max(config.n_permutations, 1)
    null_var = np.maximum(null_sum_sq / max(config.n_permutations, 1) - result["null_mean"].to_numpy() ** 2, 0.0)
    result["null_std"] = np.sqrt(null_var)
    result["top_fraction"] = float(top_fraction)
    result["n_permutations"] = int(config.n_permutations)
    return result.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False])


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _ks_uniform_statistic(p_values: np.ndarray) -> float:
    p_values = np.sort(np.asarray(p_values, dtype=float))
    p_values = p_values[np.isfinite(p_values)]
    if p_values.size == 0:
        return float("nan")
    n = p_values.size
    empirical = np.arange(1, n + 1) / n
    return float(np.max(np.maximum(np.abs(empirical - p_values), np.abs((np.arange(n) / n) - p_values))))


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _simes_p_value(p_values: np.ndarray) -> float:
    p_values = np.sort(np.asarray(p_values, dtype=float))
    p_values = p_values[np.isfinite(p_values)]
    if p_values.size == 0:
        return float("nan")
    n = p_values.size
    simes = np.min(p_values * n / np.arange(1, n + 1))
    return float(np.clip(simes, 0.0, 1.0))


def _merge_candidate_predictions(
    candidates: pd.DataFrame, predictions: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    if "candidate_id" not in candidates.columns:
        raise ValueError("Candidate table missing column: candidate_id")
    required_prediction = {"candidate_id", score_column}
    missing_prediction = required_prediction - set(predictions.columns)
    if missing_prediction:
        raise ValueError(f"Prediction table missing columns: {sorted(missing_prediction)}")

    prediction_columns = ["candidate_id", score_column]
    if "method" in predictions.columns:
        prediction_columns.append("method")
    merged = candidates.merge(
        predictions[prediction_columns], on="candidate_id", how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between candidates and predictions")
    return merged


def _clean_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _null_groups(table: pd.DataFrame, null_model: str) -> list[np.ndarray]:
    if null_model == "score_permutation":
        return [np.arange(len(table))]
    if null_model == "spatial_permutation":
        return _group_indices(table, ["lr_id"])
    if null_model == "receiver_permutation":
        return _group_indices(table, ["lr_id", "sender_cell_type"], fallback=["lr_id"])
    if null_model == "lr_identity_permutation":
        return _group_indices(table, ["sender_id", "receiver_id"], fallback=["sender_id"])
    if null_model in {"within_domain_permutation", "within_region_permutation"}:
        return _group_indices(
            table,
            ["lr_id", "sender_niche", "receiver_niche"],
            fallback=["lr_id"],
        )
    if null_model == "cell_type_preserving_permutation":
        return _group_indices(
            table,
            ["lr_id", "sender_cell_type", "receiver_cell_type"],
            fallback=["lr_id"],
        )
    raise ValueError(f"Unsupported null model: {null_model}")


def _group_indices(
    table: pd.DataFrame, columns: list[str], *, fallback: list[str] | None = None
) -> list[np.ndarray]:
    active_columns = [col for col in columns if col in table.columns]
    if not active_columns and fallback:
        active_columns = [col for col in fallback if col in table.columns]
    if not active_columns:
        return [np.arange(len(table))]

    groups = [
        group_index.to_numpy(dtype=int)
        for _, group_index in table.reset_index().groupby(active_columns, sort=False)["index"]
    ]
    if all(len(group) > 1 for group in groups):
        return groups

    fallback_columns = [col for col in (fallback or []) if col in table.columns]
    if fallback_columns and fallback_columns != active_columns:
        return _group_indices(table, fallback_columns)
    return [np.arange(len(table))]


def _group_indices_with_keys(table: pd.DataFrame, columns: list[str]) -> tuple[list[np.ndarray], list[tuple[object, ...]]]:
    grouped = table.reset_index().groupby(columns, dropna=False, sort=False)["index"]
    indices: list[np.ndarray] = []
    keys: list[tuple[object, ...]] = []
    for group_key, group_index in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys.append(group_key)
        indices.append(group_index.to_numpy(dtype=int))
    return indices, keys


def _top_tail_mean(values: np.ndarray, *, top_fraction: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    fraction = float(np.clip(top_fraction, 1.0 / values.size, 1.0))
    k = max(1, int(np.ceil(values.size * fraction)))
    if k >= values.size:
        return float(np.mean(values))
    partition = np.partition(values, values.size - k)
    return float(np.mean(partition[-k:]))


def _permute_scores_by_group(
    scores: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    null_scores = scores.copy()
    for index in groups:
        if len(index) < 2:
            continue
        null_scores[index] = rng.permutation(scores[index])
    return null_scores
