#!/usr/bin/env python
"""Audit Phase 4 manuscript evidence-lock progress."""

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
    parser.add_argument("--config", default="configs/experiments/phase4_manuscript_evidence_lock.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path) if config_path.exists() else {}
    output_dir = PROJECT_ROOT / (args.output_dir or str(Path(config.get("output_dir", "outputs/phase4/manuscript_evidence_lock")) / "audit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[AuditRow] = []
    rows.extend(_audit_static_contract(config_path, config))
    rows.extend(_audit_evidence_index(config))

    csv_path = output_dir / "phase4_audit_summary.csv"
    json_path = output_dir / "phase4_audit_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal_id", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    counts = _counts(row.status for row in rows)
    print(f"Phase 4 audit CSV:  {csv_path}")
    print(f"Phase 4 audit JSON: {json_path}")
    print("Status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("\nRows needing action:")
    for row in rows:
        if row.status != "complete":
            print(f"  {row.goal_id} | {row.status} | {row.requirement} | {row.detail}")


def _audit_static_contract(config_path: Path, config: dict[str, Any]) -> list[AuditRow]:
    docs_path = PROJECT_ROOT / "docs/PHASE4_MANUSCRIPT_EVIDENCE_LOCK.md"
    index_script = PROJECT_ROOT / "scripts/phase4/build_phase4_evidence_index.py"
    audit_script = PROJECT_ROOT / "scripts/phase4/audit_phase4.py"
    required_goals = {
        "goal1_formalize_benchmarks",
        "goal2_strengthen_baselines",
        "goal3_upgrade_fdr",
        "goal4_weak_signal_evidence",
        "goal5_real_validation",
        "goal6_ablation_lock",
        "goal7_manuscript_figures",
    }
    goals = set(config.get("phase4_goals") or [])
    retained = config.get("retained_methods") or {}
    excluded = set(retained.get("excluded_methods") or [])
    gates = config.get("submission_gates") or {}
    nulls = set(gates.get("required_null_models") or [])
    text = docs_path.read_text(encoding="utf-8") if docs_path.exists() else ""
    return [
        AuditRow("phase4", "Phase 4 config exists", _exists(config_path), _rel(config_path)),
        AuditRow("phase4", "Phase 4 evidence-lock document exists", _exists(docs_path), _rel(docs_path)),
        AuditRow("phase4", "Phase 4 evidence index builder exists", _exists(index_script), _rel(index_script)),
        AuditRow("phase4", "Phase 4 audit script exists", _exists(audit_script), _rel(audit_script)),
        AuditRow(
            "phase4",
            "config declares exactly the seven Phase 4 goals",
            "complete" if goals == required_goals else "partial",
            _rel(config_path),
            f"goals={sorted(goals)}",
        ),
        AuditRow(
            "phase4_goal2_strengthen_baselines",
            "LR-Coexp remains excluded",
            "complete" if "lr_coexp" in excluded else "partial",
            _rel(config_path),
            f"excluded={sorted(excluded)}",
        ),
        AuditRow(
            "phase4_goal3_upgrade_fdr",
            "FDR gate requires three fixed null models and >=200 formal permutations",
            "complete"
            if nulls == {"spatial_permutation", "receiver_permutation", "lr_identity_permutation"}
            and int(gates.get("formal_fdr_permutations_min", 0)) >= 200
            else "partial",
            _rel(config_path),
            f"nulls={sorted(nulls)};formal_min={gates.get('formal_fdr_permutations_min')}",
        ),
        AuditRow(
            "phase4",
            "document contains wording guardrail against global dominance",
            "complete" if "global dominance" in text.lower() and "not allowed" in text.lower() else "partial",
            _rel(docs_path),
        ),
    ]


def _audit_evidence_index(config: dict[str, Any]) -> list[AuditRow]:
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase4/manuscript_evidence_lock")
    index_path = output_dir / "phase4_evidence_index.csv"
    summary_path = output_dir / "phase4_evidence_index.summary.json"
    rows = [
        AuditRow("phase4", "Phase 4 evidence index exists", _exists(index_path), _rel(index_path)),
        AuditRow("phase4", "Phase 4 evidence summary exists", _exists(summary_path), _rel(summary_path)),
    ]
    if not index_path.exists():
        return rows

    table = _read_csv(index_path)
    goals = {row.get("goal_id", "") for row in table}
    required = set(config.get("phase4_goals") or [])
    non_complete = [row for row in table if row.get("status") != "complete"]
    final_rows = [row for row in table if row.get("tier") == "final"]
    formal_or_final = [row for row in table if row.get("tier") in {"formal", "final"}]
    rows.extend(
        [
            AuditRow(
                "phase4",
                "evidence index covers all Phase 4 goals",
                "complete" if required.issubset(goals) else "partial",
                _rel(index_path),
                f"missing={sorted(required - goals)}",
            ),
            AuditRow(
                "phase4",
                "index intentionally distinguishes complete evidence from remaining actions",
                "complete" if non_complete else "partial",
                _rel(index_path),
                f"rows_needing_action={len(non_complete)}",
            ),
            AuditRow(
                "phase4",
                "some evidence has reached formal/final tier",
                "complete" if formal_or_final else "partial",
                _rel(index_path),
                f"formal_or_final_rows={len(formal_or_final)}",
            ),
            AuditRow(
                "phase4",
                "final manuscript completion is not falsely claimed",
                "complete" if not final_rows or non_complete else "partial",
                _rel(index_path),
                f"final_rows={len(final_rows)};non_complete={len(non_complete)}",
            ),
        ]
    )
    rows.extend(_audit_goal_specific(index_path, table))
    return rows


def _audit_goal_specific(index_path: Path, table: list[dict[str, str]]) -> list[AuditRow]:
    by_area = {(row.get("goal_id", ""), row.get("evidence_area", "")): row for row in table}
    checks = [
        (
            "goal1_formalize_benchmarks",
            "controlled simulator reaches Phase 4 replicate gate",
            by_area.get(("goal1_formalize_benchmarks", "controlled_simulator_v2_replicates")),
        ),
        (
            "goal1_formalize_benchmarks",
            "real-background benchmark reaches Phase 4 replicate gate",
            by_area.get(("goal1_formalize_benchmarks", "real_background_replicates")),
        ),
        (
            "goal2_strengthen_baselines",
            "replicated baseline panel is formal",
            by_area.get(("goal2_strengthen_baselines", "replicated_baseline_panel")),
        ),
        (
            "goal3_upgrade_fdr",
            "FDR reaches final 500-permutation target",
            by_area.get(("goal3_upgrade_fdr", "fdr_permutation_depth")),
        ),
        (
            "goal4_weak_signal_evidence",
            "weak-signal claim remains tradeoff-gated",
            by_area.get(("goal4_weak_signal_evidence", "raw_weak_recall_guardrail")),
        ),
        (
            "goal5_real_validation",
            "downstream validation is no longer coverage-limited",
            by_area.get(("goal5_real_validation", "downstream_validation")),
        ),
        (
            "goal6_ablation_lock",
            "Phase 4 formal ablation panel exists",
            by_area.get(("goal6_ablation_lock", "required_ablation_coverage")),
        ),
        (
            "goal7_manuscript_figures",
            "scaffold figures are converted to final manuscript figures",
            by_area.get(("goal7_manuscript_figures", "figure_source_scaffold")),
        ),
    ]
    rows: list[AuditRow] = []
    for goal_id, requirement, row in checks:
        if not row:
            rows.append(AuditRow(goal_id, requirement, "missing", _rel(index_path), "evidence row absent"))
            continue
        expected_complete = requirement in {
            "replicated baseline panel is formal",
            "weak-signal claim remains tradeoff-gated",
        }
        if expected_complete:
            status = "complete" if row.get("status") == "complete" else "partial"
        else:
            status = "complete" if row.get("status") == "complete" and row.get("tier") in {"formal", "final"} else "partial"
        rows.append(
            AuditRow(
                goal_id,
                requirement,
                status,
                _rel(index_path),
                f"tier={row.get('tier')};status={row.get('status')};current={row.get('current_value')};target={row.get('target_value')}",
            )
        )
    return rows


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
        raise RuntimeError("PyYAML is required for Phase 4 audit.") from exc
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
