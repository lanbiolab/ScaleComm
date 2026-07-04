#!/usr/bin/env python
"""Control-separation FDR for Phase 7 Goal 5 model-rerun outputs.

Absolute LR-pair FDR can mark both true positive controls and high-expression
negative controls significant when the whole model score scale shifts above
rerun nulls. This diagnostic asks a narrower, pre-specified question:

    Is the best positive control separated from the best high-expression
    negative control more strongly in observed data than under rerun nulls?

It is supporting evidence for false-positive suppression, not a replacement for
strict all-null LR-level FDR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"


DATASET_RUNS = {
    "slide_tags_tonsil_n1200": {
        "source_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    "slide_tags_tonsil_n1200_hardtrain": {
        "source_dir": GOAL5_DIR / "slide_tags_tonsil_n1200_hardtrain_model_rerun_fdr_20perm",
        "prefix": "slide_tags_tonsil_n1200",
    },
    "post_xenium_colon": {
        "source_dir": GOAL5_DIR / "post_xenium_colon_model_rerun_fdr_50perm",
        "prefix": "post_xenium_colon",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_RUNS), required=True)
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    spec = DATASET_RUNS[args.dataset]
    source_dir = Path(args.source_dir) if args.source_dir else Path(spec["source_dir"])
    prefix = str(args.prefix or spec["prefix"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = pd.read_csv(source_dir / f"{prefix}_model_rerun_observed_lr_stats.csv")
    null = pd.read_csv(source_dir / f"{prefix}_model_rerun_null_lr_stats.csv")
    status = _read_optional_csv(source_dir / f"{prefix}_model_rerun_status.csv")

    observed_sep = _observed_separation(observed)
    null_sep = _null_separation(null)
    fdr = _separation_fdr(observed_sep, null_sep, alpha=float(args.alpha))

    observed_path = output_dir / f"{args.dataset}_control_separation_observed.csv"
    null_path = output_dir / f"{args.dataset}_control_separation_null.csv"
    fdr_path = output_dir / f"{args.dataset}_control_separation_fdr.csv"
    manifest_path = output_dir / f"{args.dataset}_control_separation_manifest.json"
    observed_sep.to_csv(observed_path, index=False)
    null_sep.to_csv(null_path, index=False)
    fdr.to_csv(fdr_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "source_dir": _rel(source_dir),
        "output_dir": _rel(output_dir),
        "prefix": prefix,
        "alpha": float(args.alpha),
        "n_completed_reruns": int(status["status"].eq("completed").sum()) if not status.empty else "",
        "separation_success": bool(fdr["separation_status"].eq("positive_separated_from_negative").all()),
        "claim_boundary": (
            "Control-separation FDR tests whether positive controls outrank high-expression negatives under the "
            "same model-rerun nulls. It supports false-positive suppression but does not replace strict LR-level "
            "all-null FDR for each individual prediction."
        ),
        "outputs": {
            "observed": _rel(observed_path),
            "null": _rel(null_path),
            "fdr": _rel(fdr_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(observed_sep.to_string(index=False))
    print(fdr.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _observed_separation(observed: pd.DataFrame) -> pd.DataFrame:
    positives = observed[_positive_mask(observed)].copy()
    negatives = observed[_negative_mask(observed)].copy()
    if positives.empty or negatives.empty:
        raise ValueError("Observed stats must contain at least one positive and one high-expression negative control.")
    best_positive = positives.sort_values("observed_or_null_statistic", ascending=False).iloc[0]
    best_negative = negatives.sort_values("observed_or_null_statistic", ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {
                "best_positive": _control_name(best_positive),
                "best_negative": _control_name(best_negative),
                "best_positive_statistic": float(best_positive["observed_or_null_statistic"]),
                "best_negative_statistic": float(best_negative["observed_or_null_statistic"]),
                "positive_minus_negative": float(
                    best_positive["observed_or_null_statistic"] - best_negative["observed_or_null_statistic"]
                ),
            }
        ]
    )


def _null_separation(null: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (null_model, permutation), group in null.groupby(["null_model", "permutation"], sort=False):
        positives = group[_positive_mask(group)]
        negatives = group[_negative_mask(group)]
        if positives.empty or negatives.empty:
            continue
        best_positive = positives.sort_values("observed_or_null_statistic", ascending=False).iloc[0]
        best_negative = negatives.sort_values("observed_or_null_statistic", ascending=False).iloc[0]
        rows.append(
            {
                "null_model": null_model,
                "permutation": int(permutation),
                "best_positive": _control_name(best_positive),
                "best_negative": _control_name(best_negative),
                "best_positive_statistic": float(best_positive["observed_or_null_statistic"]),
                "best_negative_statistic": float(best_negative["observed_or_null_statistic"]),
                "positive_minus_negative": float(
                    best_positive["observed_or_null_statistic"] - best_negative["observed_or_null_statistic"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _separation_fdr(observed_sep: pd.DataFrame, null_sep: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    observed_value = float(observed_sep.iloc[0]["positive_minus_negative"])
    rows = []
    for null_model, group in null_sep.groupby("null_model", sort=False):
        values = pd.to_numeric(group["positive_minus_negative"], errors="coerce").dropna().to_numpy(dtype=float)
        n_perm = int(len(values))
        if n_perm:
            p_value = (float((values >= observed_value).sum()) + 1.0) / (float(n_perm) + 1.0)
            null_mean = float(np.mean(values))
            null_q95 = float(np.quantile(values, 0.95))
        else:
            p_value = np.nan
            null_mean = np.nan
            null_q95 = np.nan
        rows.append(
            {
                "null_model": null_model,
                "observed_positive_minus_negative": observed_value,
                "null_mean": null_mean,
                "null_q95": null_q95,
                "empirical_p_value": p_value,
                "n_completed_permutations": n_perm,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bh_fdr"] = _bh_adjust(pd.to_numeric(out["empirical_p_value"], errors="coerce").to_numpy(dtype=float))
    out["separation_status"] = np.where(
        pd.to_numeric(out["bh_fdr"], errors="coerce") <= float(alpha),
        "positive_separated_from_negative",
        "not_significant",
    )
    return out


def _positive_mask(frame: pd.DataFrame) -> pd.Series:
    false_positive = frame.get("false_positive_high_expression", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if "true_lr" in frame:
        return frame["true_lr"].fillna(False).astype(bool) & ~false_positive
    return frame.get("control_class", pd.Series("", index=frame.index)).astype(str).str.contains("positive", case=False, na=False)


def _negative_mask(frame: pd.DataFrame) -> pd.Series:
    if "false_positive_high_expression" in frame:
        return frame["false_positive_high_expression"].fillna(False).astype(bool)
    return frame.get("control_class", pd.Series("", index=frame.index)).astype(str).str.contains("negative", case=False, na=False)


def _control_name(row: pd.Series) -> str:
    value = row.get("control_name", "")
    if pd.isna(value) or str(value) == "":
        return str(row.get("lr_id", ""))
    return str(value)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    out = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return out
    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[valid] = tmp
    return out


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
