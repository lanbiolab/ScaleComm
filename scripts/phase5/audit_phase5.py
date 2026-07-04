#!/usr/bin/env python
"""Audit Phase 5 submission-manuscript lock progress."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AuditRow:
    goal_id: str
    requirement: str
    status: str
    evidence: str
    detail: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase5_submission_manuscript_lock.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs/phase5/manuscript_submission_lock")
    output_dir = PROJECT_ROOT / (args.output_dir or str(output_root / "audit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[AuditRow] = []
    rows.extend(_audit_static(config_path, config))
    rows.extend(_audit_index(config, output_root))

    csv_path = output_dir / "phase5_audit_summary.csv"
    json_path = output_dir / "phase5_audit_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal_id", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    counts = _counts(row.status for row in rows)
    print(f"Phase 5 audit CSV:  {csv_path}")
    print(f"Phase 5 audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("\nRows needing action:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")


def _audit_static(config_path: Path, config: dict[str, Any]) -> list[AuditRow]:
    required_goals = {
        "goal1_claim_boundary",
        "goal2_real_biological_validation",
        "goal3_main_figure_package",
        "goal4_results_draft",
        "goal5_methods_reproducibility",
        "goal6_real_negative_controls",
    }
    goals = set(config.get("phase5_goals") or [])
    sources = config.get("source_artifacts") or {}
    p5 = sources.get("phase5") or {}
    paths = {
        "claim_boundary": PROJECT_ROOT / p5.get("claim_boundary", ""),
        "results_draft": PROJECT_ROOT / p5.get("results_draft", ""),
        "methods_reproducibility": PROJECT_ROOT / p5.get("methods_reproducibility", ""),
        "figure_plan": PROJECT_ROOT / p5.get("figure_plan", ""),
    }
    excluded = set(config.get("excluded_baselines") or [])
    return [
        AuditRow("phase5", "Phase 5 config exists", _exists(config_path), _rel(config_path)),
        AuditRow(
            "phase5",
            "config declares exactly the six Phase 5 goals",
            "complete" if goals == required_goals else "partial",
            _rel(config_path),
            f"goals={sorted(goals)}",
        ),
        AuditRow(
            "phase5_goal1_claim_boundary",
            "CLAIM_BOUNDARY.md exists",
            _exists(paths["claim_boundary"]),
            _rel(paths["claim_boundary"]),
        ),
        AuditRow(
            "phase5_goal3_main_figure_package",
            "main figure plan exists",
            _exists(paths["figure_plan"]),
            _rel(paths["figure_plan"]),
        ),
        AuditRow(
            "phase5_goal4_results_draft",
            "Results draft exists",
            _exists(paths["results_draft"]),
            _rel(paths["results_draft"]),
        ),
        AuditRow(
            "phase5_goal5_methods_reproducibility",
            "Methods reproducibility checklist exists",
            _exists(paths["methods_reproducibility"]),
            _rel(paths["methods_reproducibility"]),
        ),
        AuditRow(
            "phase5_goal5_methods_reproducibility",
            "LR-Coexp remains excluded",
            "complete" if "lr_coexp" in excluded else "partial",
            _rel(config_path),
            f"excluded={sorted(excluded)}",
        ),
    ]


def _audit_index(config: dict[str, Any], output_root: Path) -> list[AuditRow]:
    index_path = output_root / "phase5_evidence_index.csv"
    summary_path = output_root / "phase5_evidence_index.summary.json"
    rows = [
        AuditRow("phase5", "Phase 5 evidence index exists", _exists(index_path), _rel(index_path)),
        AuditRow("phase5", "Phase 5 evidence summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not index_path.exists():
        return rows
    table = _read_csv(index_path)
    required = set(config.get("phase5_goals") or [])
    goals = {row.get("goal_id", "") for row in table}
    complete_rows = [row for row in table if row.get("status") == "complete"]
    incomplete_rows = [row for row in table if row.get("status") != "complete"]
    rows.extend(
        [
            AuditRow(
                "phase5",
                "evidence index covers all Phase 5 goals",
                "complete" if required.issubset(goals) else "partial",
                _rel(index_path),
                f"missing={sorted(required - goals)}",
            ),
            AuditRow(
                "phase5",
                "Phase 5 evidence rows are all complete",
                "complete" if not incomplete_rows else "partial",
                _rel(index_path),
                f"complete_rows={len(complete_rows)};incomplete_rows={len(incomplete_rows)}",
            ),
        ]
    )
    rows.extend(_audit_goal_rows(index_path, table))
    return rows


def _audit_goal_rows(index_path: Path, table: list[dict[str, str]]) -> list[AuditRow]:
    by_area = {(row.get("goal_id", ""), row.get("evidence_area", "")): row for row in table}
    checks = [
        (
            "goal1_claim_boundary",
            "claim boundary is formal and complete",
            by_area.get(("goal1_claim_boundary", "claim_boundary_document")),
            {"formal"},
            {"complete"},
        ),
        (
            "goal2_real_biological_validation",
            "three target tissues have control-ranking validation",
            by_area.get(("goal2_real_biological_validation", "three_tissue_control_ranking")),
            {"formal"},
            {"complete"},
        ),
        (
            "goal2_real_biological_validation",
            "cross-tissue downstream/niche scope is not overstated",
            by_area.get(("goal2_real_biological_validation", "downstream_cross_tissue_scope")),
            {"representative", "formal"},
            {"needs_review", "complete"},
        ),
        (
            "goal3_main_figure_package",
            "main figure package has source-linked plan",
            by_area.get(("goal3_main_figure_package", "main_figure_plan")),
            {"representative", "formal"},
            {"complete", "needs_polish"},
        ),
        (
            "goal4_results_draft",
            "Results draft is figure-linked and complete",
            by_area.get(("goal4_results_draft", "results_skeleton")),
            {"formal"},
            {"complete"},
        ),
        (
            "goal5_methods_reproducibility",
            "Methods reproducibility draft is complete",
            by_area.get(("goal5_methods_reproducibility", "methods_checklist")),
            {"formal"},
            {"complete"},
        ),
        (
            "goal6_real_negative_controls",
            "three-tissue positive/negative controls are formal",
            by_area.get(("goal6_real_negative_controls", "positive_negative_control_ranking")),
            {"formal"},
            {"complete"},
        ),
    ]
    out: list[AuditRow] = []
    for goal_id, requirement, row, tiers, statuses in checks:
        if not row:
            out.append(AuditRow(goal_id, requirement, "missing", _rel(index_path), "evidence row absent"))
            continue
        status = "complete" if row.get("tier") in tiers and row.get("status") in statuses else "partial"
        out.append(
            AuditRow(
                goal_id,
                requirement,
                status,
                _rel(index_path),
                f"tier={row.get('tier')};status={row.get('status')};current={row.get('current_value')};target={row.get('target_value')}",
            )
        )
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _exists(path: Path) -> str:
    return "complete" if path.exists() else "missing"


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 5 audit.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
