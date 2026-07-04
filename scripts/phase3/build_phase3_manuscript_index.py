#!/usr/bin/env python
"""Build Phase 3 manuscript source-table and figure scaffold indexes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_COLUMNS = [
    "source_table_id",
    "phase3_goal",
    "path",
    "exists",
    "n_rows",
    "intended_figures",
    "supported_claim",
    "limitation",
]

FIGURE_COLUMNS = [
    "figure_id",
    "title",
    "source_table_ids",
    "source_paths",
    "plotting_script",
    "figure_output",
    "qa_status",
    "supported_claim",
    "limitation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase3_end_to_end.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / (args.output_dir or config.get("output_dir", "outputs/phase3/end_to_end"))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = _source_rows(output_dir)
    figure_rows = _figure_rows(config, source_rows, output_dir)

    source_path = output_dir / "phase3_manuscript_source_table_index.csv"
    figure_path = output_dir / "phase3_figure_scaffold_index.csv"
    _write_csv(source_path, source_rows, SOURCE_COLUMNS)
    _write_csv(figure_path, figure_rows, FIGURE_COLUMNS)

    summary = {
        "source_table_index_csv": _rel(source_path),
        "figure_scaffold_index_csv": _rel(figure_path),
        "n_source_tables": len(source_rows),
        "n_existing_source_tables": sum(1 for row in source_rows if row["exists"] is True),
        "n_figures": len(figure_rows),
        "n_figures_with_existing_sources": sum(1 for row in figure_rows if row["qa_status"] == "source_tables_available_plot_pending"),
        "claim_level": config.get("claim_level", ""),
    }
    summary_path = output_dir / "phase3_manuscript_index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 manuscript source table index: {source_path}")
    print(f"Wrote Phase 3 figure scaffold index: {figure_path}")
    print(json.dumps(summary, indent=2))


def _source_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = [
        _source_row(
            "ST1_dataset_manifest",
            "goal1_dataset_manifest",
            output_dir / "phase3_dataset_manifest.csv",
            "Fig1;Fig2;Fig3;Fig7;Fig8",
            "Unified data entry points are frozen for Phase 3.",
            "Optional Xenium/Visium HD placeholders are not populated yet.",
        ),
        _source_row(
            "ST2_prediction_index",
            "goal2_main_model_and_baselines",
            output_dir / "phase3_prediction_index.csv",
            "Fig4;Fig8",
            "Main model and retained baselines have a unified prediction/status index.",
            "High-resolution CellNEST is missing; DeepTalk/NICHES are status rows.",
        ),
        _source_row(
            "ST3_metric_index",
            "goal3_unified_evaluation",
            output_dir / "phase3_metric_index.csv",
            "Fig2;Fig3;Fig4;Fig5;Fig8",
            "Truth/control/runtime metrics are normalized for comparison.",
            "Results are first-pass and include adapter-light baseline caveats.",
        ),
        _source_row(
            "ST4_fdr_index",
            "goal4_fdr_permutation",
            output_dir / "phase3_fdr_index.csv",
            "Fig6",
            "FDR artifacts are attached for real-background SpatialSelf 100-perm panel.",
            "Controlled/high-resolution FDR remain not run; 500-perm final calibration pending.",
        ),
        _source_row(
            "ST5_biological_validation_index",
            "goal5_biological_validation",
            output_dir / "phase3_biological_validation_index.csv",
            "Fig7",
            "Positive/negative controls and downstream validation are indexed across real contexts.",
            "Curated downstream coverage is limited and not a unique-discovery claim.",
        ),
        _source_row(
            "ST6_audit_summary",
            "phase3_audit",
            output_dir / "audit/phase3_audit_summary.csv",
            "Fig1;Supplement",
            "Phase 3 gates are auditable from generated artifacts.",
            "Audit checks scaffold completeness, not final manuscript strength.",
        ),
    ]
    return rows


def _source_row(
    source_table_id: str,
    goal: str,
    path: Path,
    figures: str,
    claim: str,
    limitation: str,
) -> dict[str, Any]:
    return {
        "source_table_id": source_table_id,
        "phase3_goal": goal,
        "path": _rel(path),
        "exists": path.exists(),
        "n_rows": _count_rows(path),
        "intended_figures": figures,
        "supported_claim": claim,
        "limitation": limitation,
    }


def _figure_rows(config: dict[str, Any], sources: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    figure_sources = {
        "Fig1": ["ST1_dataset_manifest", "ST6_audit_summary"],
        "Fig2": ["ST1_dataset_manifest", "ST3_metric_index"],
        "Fig3": ["ST1_dataset_manifest", "ST3_metric_index"],
        "Fig4": ["ST2_prediction_index", "ST3_metric_index"],
        "Fig5": ["ST3_metric_index"],
        "Fig6": ["ST4_fdr_index"],
        "Fig7": ["ST5_biological_validation_index", "ST3_metric_index"],
        "Fig8": ["ST1_dataset_manifest", "ST2_prediction_index", "ST3_metric_index"],
    }
    claims = {
        "Fig1": "SpatialSelf manuscript pipeline has a defined data-to-evidence scaffold.",
        "Fig2": "Controlled simulator v2 benchmark can report truth-level metrics.",
        "Fig3": "Real-background semi-synthetic benchmarks can report truth/control metrics.",
        "Fig4": "SpatialSelf and retained baselines are compared under one metric schema.",
        "Fig5": "Weak-signal versus false-positive tradeoff can be read from unified metrics.",
        "Fig6": "Permutation/FDR module is wired to representative real-background predictions.",
        "Fig7": "Biological validation framework is indexed across lymph node, tonsil, melanoma, and high-resolution tonsil.",
        "Fig8": "Runtime/scalability evidence can be sourced from prediction and metric indexes.",
    }
    limitations = {
        "Fig1": "Overview only; not a performance claim.",
        "Fig2": "Only replicate_00 is frozen in Phase 3 manifest; replicate expansion pending.",
        "Fig3": "Semi-synthetic truth is injected, not native biological ground truth.",
        "Fig4": "Some baselines are adapter-light; DeepTalk/NICHES are status-only.",
        "Fig5": "Weak-signal claim remains a tradeoff claim, not universal highest recall.",
        "Fig6": "100 permutations are representative; final 200-500 permutation panel pending.",
        "Fig7": "Validation is framework-level; curated downstream coverage is limited.",
        "Fig8": "High-resolution panel lacks CellNEST and full-platform scale.",
    }
    source_by_id = {row["source_table_id"]: row for row in sources}
    rows: list[dict[str, Any]] = []
    for fig in config.get("figures") or []:
        figure_id = fig.get("figure_id", "")
        source_ids = figure_sources.get(figure_id, [])
        source_paths = [source_by_id[source_id]["path"] for source_id in source_ids if source_id in source_by_id]
        sources_available = all(source_by_id.get(source_id, {}).get("exists") is True for source_id in source_ids)
        rows.append(
            {
                "figure_id": figure_id,
                "title": fig.get("title", ""),
                "source_table_ids": ";".join(source_ids),
                "source_paths": ";".join(source_paths),
                "plotting_script": _rel(PROJECT_ROOT / "scripts/phase3/figures/plot_phase3_scaffold_figures.py"),
                "figure_output": _rel(output_dir / "figures" / f"{figure_id.lower()}_phase3_scaffold.pdf"),
                "qa_status": "source_tables_available_plot_pending" if sources_available else "missing_source_tables",
                "supported_claim": claims.get(figure_id, ""),
                "limitation": limitations.get(figure_id, ""),
            }
        )
    return rows


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 3 config loading.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
