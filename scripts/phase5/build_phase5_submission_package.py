#!/usr/bin/env python
"""Build the Phase 5 submission manuscript package index."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


FIGURE_PACKAGE_COLUMNS = [
    "figure_id",
    "manuscript_slot",
    "title",
    "primary_claim",
    "reviewer_concern",
    "limitation",
    "source_tables",
    "source_figure_id",
    "source_pdf",
    "source_svg",
    "source_tiff",
    "packaged_pdf",
    "packaged_svg",
    "packaged_tiff",
    "status",
]

SOURCE_DATA_COLUMNS = [
    "figure_id",
    "source_table",
    "exists",
    "role",
]

PANEL_SOURCE_COLUMNS = [
    "figure_id",
    "panel_id",
    "panel_conclusion",
    "source_tables",
    "reviewer_concern",
    "limitation",
    "status",
]

QA_COLUMNS = [
    "artifact",
    "requirement",
    "status",
    "evidence",
    "detail",
]


MAIN_FIGURES = [
    {
        "figure_id": "Fig1",
        "manuscript_slot": "fig1_method_overview",
        "title": "Method overview",
        "primary_claim": "SpatialSelf combines spatial topology, graph-wavelet propagation, LR identity, expression, distance, self-supervised scoring, and permutation/FDR calibration.",
        "reviewer_concern": "The method should not be a generic expression-distance graph model.",
        "limitation": "Schematic overview; performance claims require benchmark, ablation, FDR, and validation figures.",
        "source": "phase4",
        "source_figure_id": "Fig1",
        "source_tables": [
            "outputs/phase4/manuscript_evidence_lock/phase4_evidence_index.csv",
            "outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_summary.csv",
        ],
    },
    {
        "figure_id": "Fig2",
        "manuscript_slot": "fig2_controlled_benchmark",
        "title": "Controlled mechanism benchmark",
        "primary_claim": "The controlled simulator tests contact, paracrine, long-range niche, weak-signal, and high-expression false-positive mechanisms.",
        "reviewer_concern": "The benchmark should not be aligned only with high expression and short distance.",
        "limitation": "Truth is injected and synthetic.",
        "source": "phase4",
        "source_figure_id": "Fig2",
        "source_tables": [
            "outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/formal_benchmark_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/formal_lr_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/formal_validation_summary.csv",
        ],
    },
    {
        "figure_id": "Fig3",
        "manuscript_slot": "fig3_real_background_benchmark",
        "title": "Real-background semi-synthetic benchmark",
        "primary_claim": "Real-background benchmarks use lymph-node, tonsil, and melanoma-like spatial expression backgrounds while keeping injected LR truth available.",
        "reviewer_concern": "Simulation should use realistic spatial structure and expression noise.",
        "limitation": "Injected truth is not native real-tissue ground truth.",
        "source": "phase4",
        "source_figure_id": "Fig3",
        "source_tables": [
            "outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/formal_real_background_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/formal_injection_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/formal_validation_summary.csv",
        ],
    },
    {
        "figure_id": "Fig4",
        "manuscript_slot": "fig4_baseline_comparison",
        "title": "Baseline comparison",
        "primary_claim": "SpatialSelf supports a metric-specific weak-signal and false-positive tradeoff against retained baselines.",
        "reviewer_concern": "The manuscript needs traditional, spatial, and deep-learning baseline comparisons.",
        "limitation": "No global-best claim; DeepTalk and NICHES remain status methods.",
        "source": "phase4",
        "source_figure_id": "Fig4",
        "source_tables": [
            "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_metric_summary.csv",
            "outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_claims.csv",
            "outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_baseline_status_table.csv",
        ],
    },
    {
        "figure_id": "Fig5",
        "manuscript_slot": "fig5_ablation_weak_signal",
        "title": "Ablation and weak-signal mechanism",
        "primary_claim": "LR identity and graph-wavelet modeling contribute to controlled-benchmark performance.",
        "reviewer_concern": "The gain should not be attributed only to model capacity or pseudo-label bias.",
        "limitation": "Current formal ablation is a 10-epoch coverage panel.",
        "source": "phase4",
        "source_figure_id": "Fig5",
        "source_tables": [
            "outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_metrics.csv",
        ],
    },
    {
        "figure_id": "Fig6",
        "manuscript_slot": "fig6_fdr",
        "title": "Permutation/FDR calibration",
        "primary_claim": "SpatialSelf outputs can be calibrated with spatial, receiver, and LR-identity null models.",
        "reviewer_concern": "The method needs p-values and FDR rather than only scores.",
        "limitation": "Strongest current evidence is group-level in the key lymph-node panel; edge-level discoveries are limited.",
        "source": "phase4",
        "source_figure_id": "Fig6",
        "source_tables": [
            "outputs/phase4/manuscript_evidence_lock/goal3_fdr_500perm_lymph_node/goal5_formal_fdr_summary.csv",
            "outputs/phase4/manuscript_evidence_lock/goal3_fdr_500perm_lymph_node/goal5_formal_fdr_group_summary.csv",
        ],
    },
    {
        "figure_id": "Fig7",
        "manuscript_slot": "fig7_lymph_node_validation",
        "title": "Lymph node validation",
        "primary_claim": "Lymph-node positive controls rank above high-expression negatives and key downstream evidence supports biological plausibility.",
        "reviewer_concern": "Real data validation should not be limited to a selected example.",
        "limitation": "Key lymph-node validation does not imply global cross-tissue downstream validation.",
        "source": "phase5",
        "source_figure_id": "Fig7",
        "source_tables": [
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_tissue_summary.csv",
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_method_summary.csv",
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_lr_panel.csv",
            "outputs/phase4/manuscript_evidence_lock/goal5_downstream_key_lymph_node/phase4_downstream_key_lymph_node_summary.csv",
        ],
    },
    {
        "figure_id": "Fig8",
        "manuscript_slot": "fig8_tonsil_or_melanoma_validation",
        "title": "Tonsil and melanoma validation",
        "primary_claim": "Tonsil and melanoma panels preserve positive/negative control separation with low high-expression false-positive rates.",
        "reviewer_concern": "The method needs high-resolution or heterogeneous-tissue validation beyond lymph node.",
        "limitation": "Tonsil and melanoma currently use proxy/niche validation rather than key curated downstream validation.",
        "source": "phase5",
        "source_figure_id": "Fig8",
        "source_tables": [
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_tissue_summary.csv",
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_method_summary.csv",
            "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_lr_panel.csv",
            "outputs/phase5/manuscript_submission_lock/goal2_cross_tissue_scope/phase5_cross_tissue_validation_scope.csv",
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase5_submission_manuscript_lock.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs/phase5/manuscript_submission_lock")
    package_dir = PROJECT_ROOT / (args.output_dir or str(output_root / "main_figures"))
    package_dir.mkdir(parents=True, exist_ok=True)

    figure_rows = _build_figure_package(package_dir)
    source_rows = _build_source_index()
    panel_rows = _build_panel_source_index()
    qa_rows = _build_qa(config, figure_rows, source_rows, panel_rows)

    figure_index_path = package_dir / "phase5_main_figure_package_index.csv"
    source_index_path = package_dir / "phase5_main_figure_source_data_index.csv"
    panel_index_path = package_dir / "phase5_main_figure_panel_source_index.csv"
    qa_path = package_dir / "phase5_results_methods_qa.csv"
    summary_path = package_dir / "phase5_submission_package_summary.json"

    _write_csv(figure_index_path, figure_rows, FIGURE_PACKAGE_COLUMNS)
    _write_csv(source_index_path, source_rows, SOURCE_DATA_COLUMNS)
    _write_csv(panel_index_path, panel_rows, PANEL_SOURCE_COLUMNS)
    _write_csv(qa_path, qa_rows, QA_COLUMNS)

    summary = {
        "config": _rel(config_path),
        "package_dir": _rel(package_dir),
        "figure_index": _rel(figure_index_path),
        "source_data_index": _rel(source_index_path),
        "panel_source_index": _rel(panel_index_path),
        "results_methods_qa": _rel(qa_path),
        "n_main_figures": len(figure_rows),
        "n_complete_figures": sum(row["status"] == "complete" for row in figure_rows),
        "n_source_tables": len(source_rows),
        "n_existing_source_tables": sum(row["exists"] == "true" for row in source_rows),
        "n_panel_source_rows": len(panel_rows),
        "n_complete_panel_source_rows": sum(row["status"] == "complete" for row in panel_rows),
        "qa_status_counts": _counts(row["status"] for row in qa_rows),
        "package_complete": all(row["status"] == "complete" for row in qa_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 5 submission package: {package_dir}")
    print(json.dumps(summary, indent=2))


def _build_figure_package(package_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in MAIN_FIGURES:
        source_paths = _source_figure_paths(spec["source"], spec["source_figure_id"])
        packaged_paths: dict[str, Path] = {}
        for ext, source_path in source_paths.items():
            packaged_path = package_dir / f"{spec['figure_id'].lower()}_phase5_main.{ext}"
            packaged_paths[ext] = packaged_path
            if source_path.exists():
                shutil.copy2(source_path, packaged_path)
        status = "complete" if all(path.exists() and packaged_paths[ext].exists() for ext, path in source_paths.items()) else "missing_export"
        rows.append(
            {
                "figure_id": spec["figure_id"],
                "manuscript_slot": spec["manuscript_slot"],
                "title": spec["title"],
                "primary_claim": spec["primary_claim"],
                "reviewer_concern": spec["reviewer_concern"],
                "limitation": spec["limitation"],
                "source_tables": ";".join(spec["source_tables"]),
                "source_figure_id": spec["source_figure_id"],
                "source_pdf": _rel(source_paths["pdf"]),
                "source_svg": _rel(source_paths["svg"]),
                "source_tiff": _rel(source_paths["tiff"]),
                "packaged_pdf": _rel(packaged_paths["pdf"]),
                "packaged_svg": _rel(packaged_paths["svg"]),
                "packaged_tiff": _rel(packaged_paths["tiff"]),
                "status": status,
            }
        )
    return rows


def _build_source_index() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in MAIN_FIGURES:
        for table in spec["source_tables"]:
            path = PROJECT_ROOT / table
            rows.append(
                {
                    "figure_id": spec["figure_id"],
                    "source_table": table,
                    "exists": str(path.exists()).lower(),
                    "role": spec["primary_claim"],
                }
            )
    return rows


def _build_panel_source_index() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in MAIN_FIGURES:
        source_tables = ";".join(spec["source_tables"])
        for panel_id, conclusion in [
            ("a", spec["primary_claim"]),
            ("b", f"Reviewer-facing control: {spec['reviewer_concern']}"),
            ("c", f"Boundary: {spec['limitation']}"),
        ]:
            rows.append(
                {
                    "figure_id": spec["figure_id"],
                    "panel_id": panel_id,
                    "panel_conclusion": conclusion,
                    "source_tables": source_tables,
                    "reviewer_concern": spec["reviewer_concern"],
                    "limitation": spec["limitation"],
                    "status": "complete"
                    if source_tables and all((PROJECT_ROOT / table).exists() for table in spec["source_tables"])
                    else "missing_source",
                }
            )
    return rows


def _build_qa(
    config: dict[str, Any],
    figure_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    panel_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase5/manuscript_submission_lock")
    results_path = PROJECT_ROOT / ((config.get("source_artifacts") or {}).get("phase5") or {}).get(
        "results_manuscript_draft", ""
    )
    methods_path = PROJECT_ROOT / ((config.get("source_artifacts") or {}).get("phase5") or {}).get(
        "methods_manuscript_draft", ""
    )
    claim_path = PROJECT_ROOT / ((config.get("source_artifacts") or {}).get("phase5") or {}).get("claim_boundary", "")
    results = _read_text(results_path)
    methods = _read_text(methods_path)
    claim_text = _read_text(claim_path)
    required_figures = config.get("required_main_figures") or []
    figure_slots = {row["manuscript_slot"] for row in figure_rows}
    figure_ids = {row["figure_id"] for row in figure_rows}
    rows = [
        _qa(
            "main_figure_package",
            "all required main figures are packaged in PDF/SVG/TIFF",
            len(figure_rows) == 8 and all(row["status"] == "complete" for row in figure_rows),
            _rel(output_dir / "main_figures" / "phase5_main_figure_package_index.csv"),
            f"figures={sorted(figure_ids)};complete={sum(row['status'] == 'complete' for row in figure_rows)}/8",
        ),
        _qa(
            "main_figure_package",
            "all required manuscript slots are represented",
            set(required_figures).issubset(figure_slots),
            _rel(output_dir / "main_figures" / "phase5_main_figure_package_index.csv"),
            f"missing={sorted(set(required_figures) - figure_slots)}",
        ),
        _qa(
            "source_data",
            "all figure source tables exist",
            bool(source_rows) and all(row["exists"] == "true" for row in source_rows),
            _rel(output_dir / "main_figures" / "phase5_main_figure_source_data_index.csv"),
            f"existing={sum(row['exists'] == 'true' for row in source_rows)}/{len(source_rows)}",
        ),
        _qa(
            "source_data",
            "each main-figure panel has a source-table mapping",
            _panels_ready(panel_rows),
            _rel(output_dir / "main_figures" / "phase5_main_figure_panel_source_index.csv"),
            f"panels={len(panel_rows)};complete={sum(row['status'] == 'complete' for row in panel_rows)};figures={len({row['figure_id'] for row in panel_rows})}",
        ),
        _qa(
            "results",
            "Results manuscript has seven sections, supported claims, limitations, and eight figure mentions",
            _results_ready(results),
            _rel(results_path),
            f"sections={_results_section_count(results)}/7;claims={results.count('Supported claim:')};limitations={results.count('Limitation:')};figures={_figure_mention_count(results)}",
        ),
        _qa(
            "results",
            "Results manuscript follows claim boundary guardrails",
            _claim_guardrails_present(results),
            _rel(results_path),
            "requires no global-best claim, no complete real ground truth claim, and explicit edge-level/group-level boundary",
        ),
        _qa(
            "methods",
            "Methods manuscript covers benchmark, graph-wavelet, pseudo-label, FDR, baselines, validation, and reproducibility",
            _methods_ready(methods),
            _rel(methods_path),
            f"topics={_methods_topic_count(methods)}/9;has_lr_coexp_exclusion={'LR-Coexp' in methods};has_500perm={'500 permutations' in methods}",
        ),
        _qa(
            "methods",
            "Methods final wording states pair-level LR controls without multi-subunit complex modeling",
            "pair-level LR controls without multi-subunit complex modeling" in methods,
            _rel(methods_path),
            "this closes the external LR database/complex-handling wording decision for Phase 5",
        ),
        _qa(
            "claim_boundary",
            "CLAIM_BOUNDARY.md contains allowed and disallowed claim sets",
            "Claims We Can Make" in claim_text and "Claims We Cannot Make" in claim_text,
            _rel(claim_path),
            "claim boundary is used as a Results wording gate",
        ),
    ]
    return rows


def _panels_ready(rows: list[dict[str, str]]) -> bool:
    return (
        len(rows) >= 24
        and len({row.get("figure_id") for row in rows}) == 8
        and all(row.get("source_tables") for row in rows)
        and all(row.get("status") == "complete" for row in rows)
    )


def _source_figure_paths(source: str, figure_id: str) -> dict[str, Path]:
    idx = figure_id.lower().replace("fig", "")
    if source == "phase4":
        stem = f"fig{idx}_phase4_manuscript"
        base = PROJECT_ROOT / "outputs/phase4/manuscript_evidence_lock/figures"
    elif source == "phase5":
        stem = f"fig{idx}_phase5_real_validation"
        base = PROJECT_ROOT / "outputs/phase5/manuscript_submission_lock/figures"
    else:
        raise ValueError(f"Unknown figure source: {source}")
    return {ext: base / f"{stem}.{ext}" for ext in ["pdf", "svg", "tiff"]}


def _results_ready(text: str) -> bool:
    return (
        _results_section_count(text) == 7
        and text.count("Supported claim:") >= 7
        and text.count("Limitation:") >= 7
        and _figure_mention_count(text) >= 8
        and "Source evidence:" in text
    )


def _results_section_count(text: str) -> int:
    sections = [
        "SpatialSelf Models CCC",
        "Realistic Simulations",
        "Weak-Signal / False-Positive",
        "Ablation Studies",
        "Permutation Testing",
        "Lymph Node",
        "High-Resolution Tonsil and Melanoma",
    ]
    return sum(section in text for section in sections)


def _figure_mention_count(text: str) -> int:
    return sum(1 for idx in range(1, 9) if f"Fig. {idx}" in text or f"Figure {idx}" in text)


def _claim_guardrails_present(text: str) -> bool:
    lowered = text.lower()
    disallowed_phrases = [
        "spatialself is the best",
        "complete native real-tissue ground truth",
        "establishes global cross-tissue downstream superiority",
        "proves global cross-tissue downstream superiority",
        "produces many significant single edges",
    ]
    required_phrases = [
        "not globally best",
        "not native complete real-tissue ground truth",
        "group-level",
        "edge-level",
        "does not establish global cross-tissue downstream superiority",
    ]
    return all(phrase not in lowered for phrase in disallowed_phrases) and all(
        phrase in lowered for phrase in required_phrases
    )


def _methods_ready(text: str) -> bool:
    return (
        _methods_topic_count(text) == 9
        and "LR-Coexp is excluded" in text
        and "DeepTalk" in text
        and "NICHES" in text
        and "K_NN" in text
        and "K_pos" in text
        and "K_neg" in text
        and "500 permutations" in text
        and "conda run -n Benchmark" in text
    )


def _methods_topic_count(text: str) -> int:
    topics = [
        "Controlled CCC Simulator",
        "Real-Background Semi-Synthetic Benchmark",
        "LR Database and Candidate Universe",
        "Spatial Graph, KNN, and Graph-Wavelet Range Handling",
        "Self-Supervised Pseudo-Labels",
        "Permutation-Based FDR",
        "Baseline Comparison",
        "Real-Sample and Real-Background Validation",
        "Reproducibility Commands",
    ]
    return sum(topic in text for topic in topics)


def _qa(artifact: str, requirement: str, ok: bool, evidence: str, detail: str) -> dict[str, str]:
    return {
        "artifact": artifact,
        "requirement": requirement,
        "status": "complete" if ok else "needs_action",
        "evidence": evidence,
        "detail": detail,
    }


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 5 package building.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
