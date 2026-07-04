#!/usr/bin/env python
"""Assemble revised Phase 2 runtime and scalability evidence tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_revised/goal7_runtime_scalability")
    parser.add_argument(
        "--method-profile",
        default="outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv",
    )
    parser.add_argument(
        "--highres-run-summary",
        default="outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_run_summary.csv",
    )
    parser.add_argument(
        "--highres-summary",
        default="outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_summary.csv",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    method_profile = _method_profile(PROJECT_ROOT / args.method_profile)
    highres_runs = _highres_runs(PROJECT_ROOT / args.highres_run_summary)
    highres_summary = _highres_summary(PROJECT_ROOT / args.highres_summary)
    claim_summary = _claim_summary(method_profile, highres_runs, highres_summary)

    method_profile.to_csv(output_dir / "goal7_method_runtime_profile.csv", index=False)
    highres_runs.to_csv(output_dir / "goal7_highres_scaling_runs.csv", index=False)
    pd.DataFrame([highres_summary]).to_csv(output_dir / "goal7_highres_scaling_summary.csv", index=False)
    pd.DataFrame([claim_summary]).to_csv(output_dir / "goal7_runtime_scalability_claims.csv", index=False)
    manifest = {
        "analysis_label": "phase2_revised_goal6_runtime_scalability_support",
        "canonical_goal": "goal6_manuscript_evidence_figure_convergence",
        "historical_output_family": "goal7_runtime_scalability",
        "evidence_status": "supporting_runtime_scalability_not_final_100k_benchmark",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "inputs": {
            "method_profile": args.method_profile,
            "highres_run_summary": args.highres_run_summary,
            "highres_summary": args.highres_summary,
        },
        "outputs": {
            "method_runtime_profile": str((output_dir / "goal7_method_runtime_profile.csv").relative_to(PROJECT_ROOT)),
            "highres_scaling_runs": str((output_dir / "goal7_highres_scaling_runs.csv").relative_to(PROJECT_ROOT)),
            "highres_scaling_summary": str((output_dir / "goal7_highres_scaling_summary.csv").relative_to(PROJECT_ROOT)),
            "claims": str((output_dir / "goal7_runtime_scalability_claims.csv").relative_to(PROJECT_ROOT)),
        },
        "limitations": [
            "Method runtime profile is inherited from Phase 1 full-slide sparse semi-synthetic profiling.",
            "High-resolution scaling is tonsil subset scaling up to 1,800 cells and 1.64M candidate edges.",
            "This is not a final 50k/100k Xenium or Visium HD full-dataset benchmark.",
        ],
    }
    (output_dir / "goal7_runtime_scalability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame([claim_summary]).to_string(index=False))
    print(f"Wrote revised runtime/scalability outputs to {output_dir}")


def _method_profile(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    keep = [
        "method",
        "status",
        "n_cells_or_spots",
        "n_candidate_edges",
        "n_lr_pairs",
        "wall_clock_seconds",
        "peak_cpu_memory_mb",
        "peak_gpu_memory_mb",
        "gpu_monitor_status",
        "runtime_per_epoch_seconds",
        "detail",
        "command",
    ]
    available = [column for column in keep if column in table.columns]
    table = table[available].copy()
    table["analysis_scope"] = "phase1_full_slide_sparse_runtime_supporting_evidence"
    table["evidence_status"] = table["status"].map(
        lambda value: "completed_runtime_profile" if str(value) == "completed" else "status_only"
    )
    return table


def _highres_runs(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    keep = [
        "analysis_label",
        "evidence_status",
        "n_cells",
        "n_lr_pairs",
        "n_candidate_edges",
        "epochs",
        "max_train_samples",
        "wall_clock_seconds",
        "train_and_inference_seconds",
        "peak_rss_mb",
        "n_positive_controls_top_decile",
        "n_negative_controls_top_decile",
        "best_positive_rank_percentile",
        "best_negative_rank_percentile",
        "output_dir",
    ]
    table = table[[column for column in keep if column in table.columns]].copy()
    table["analysis_scope"] = "tonsil_subset_high_resolution_scaling_supporting_evidence"
    return table


def _highres_summary(path: Path) -> dict:
    row = pd.read_csv(path).iloc[0].to_dict()
    row["analysis_scope"] = "tonsil_subset_high_resolution_scaling_supporting_evidence"
    return row


def _claim_summary(method_profile: pd.DataFrame, highres_runs: pd.DataFrame, highres_summary: dict) -> dict:
    completed = method_profile[method_profile["status"].astype(str) == "completed"].copy()
    spatialself = completed[completed["method"].astype(str).str.startswith("spatialself")].copy()
    commot = completed[completed["method"].astype(str) == "commot"].copy()
    fastest_spatialself = spatialself.sort_values("wall_clock_seconds").iloc[0] if len(spatialself) else {}
    edge_prior = completed[completed["method"].astype(str) == "spatialself_wavelet_edge_prior"].copy()
    return {
        "analysis_label": "phase2_revised_goal6_runtime_scalability_support",
        "canonical_goal": "goal6_manuscript_evidence_figure_convergence",
        "historical_output_family": "goal7_runtime_scalability",
        "evidence_status": "supporting_runtime_scalability_not_final_100k_benchmark",
        "n_profiled_methods_completed": int(len(completed)),
        "n_profiled_methods_status_only": int(len(method_profile) - len(completed)),
        "profile_n_cells_or_spots": int(completed["n_cells_or_spots"].max()) if len(completed) else 0,
        "profile_n_candidate_edges": int(completed["n_candidate_edges"].max()) if len(completed) else 0,
        "spatialself_edge_prior_wall_clock_seconds": float(edge_prior["wall_clock_seconds"].iloc[0]) if len(edge_prior) else float("nan"),
        "spatialself_edge_prior_peak_cpu_memory_mb": float(edge_prior["peak_cpu_memory_mb"].iloc[0]) if len(edge_prior) else float("nan"),
        "fastest_spatialself_method": fastest_spatialself.get("method", ""),
        "fastest_spatialself_wall_clock_seconds": float(fastest_spatialself.get("wall_clock_seconds", float("nan"))) if len(spatialself) else float("nan"),
        "commot_wall_clock_seconds": float(commot["wall_clock_seconds"].iloc[0]) if len(commot) else float("nan"),
        "commot_peak_cpu_memory_mb": float(commot["peak_cpu_memory_mb"].iloc[0]) if len(commot) else float("nan"),
        "spatialself_edge_prior_speedup_vs_commot": _safe_divide(
            float(commot["wall_clock_seconds"].iloc[0]) if len(commot) else float("nan"),
            float(edge_prior["wall_clock_seconds"].iloc[0]) if len(edge_prior) else float("nan"),
        ),
        "highres_n_runs": int(highres_summary.get("n_runs", 0)),
        "highres_n_cells_max": int(highres_summary.get("n_cells_max", 0)),
        "highres_n_candidate_edges_max": int(highres_summary.get("n_candidate_edges_max", 0)),
        "highres_largest_run_wall_clock_seconds": float(highres_summary.get("largest_run_wall_clock_seconds", float("nan"))),
        "highres_largest_run_peak_rss_mb": float(highres_summary.get("largest_run_peak_rss_mb", float("nan"))),
        "highres_largest_run_negative_controls_top_decile": int(highres_summary.get("largest_run_negative_controls_top_decile", -1)),
        "highres_runtime_seconds_per_candidate_edge": float(highres_summary.get("runtime_seconds_per_candidate_edge", float("nan"))),
        "interpretation": (
            "Supporting runtime evidence: SpatialSelf edge-prior profiles on a 4,035-cell/332k-edge "
            "real-background benchmark and tonsil subset scaling reaches 1,800 cells/1.64M edges. "
            "This supports scalability wiring but is not a final 50k/100k high-resolution benchmark."
        ),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator != denominator or numerator != numerator or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
