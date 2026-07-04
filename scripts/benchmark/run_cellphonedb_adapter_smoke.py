#!/usr/bin/env python3
"""Build a minimal custom CellPhoneDB database and run a smoke analysis."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from cellphonedb.src.core.methods import cpdb_analysis_method
from cellphonedb.utils import db_utils


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def build_minimal_db(input_dir: Path, output_dir: Path) -> Path:
    lr = pd.read_csv(input_dir / "lr_pairs.csv")
    genes = sorted(set(lr["ligand"]).union(lr["receptor"]))
    db_dir = output_dir / "minimal_cpdb_inputs"
    db_dir.mkdir(parents=True, exist_ok=True)

    gene_rows = []
    protein_rows = []
    for gene in genes:
        uid = f"UP_{_safe_id(gene)}"
        gene_rows.append(
            {
                "ensembl": f"ENSG_FAKE_{_safe_id(gene)}",
                "gene_name": gene,
                "hgnc_symbol": gene,
                "uniprot": uid,
            }
        )
        protein_rows.append(
            {
                "protein_name": gene,
                "tags": "",
                "tags_reason": "",
                "tags_description": "",
                "uniprot": uid,
                "receptor": bool(gene in set(lr["receptor"])),
                "receptor_desc": "",
                "other": False,
                "other_desc": "",
                "secreted_highlight": bool(gene in set(lr["ligand"])),
                "secreted_desc": "",
                "transmembrane": bool(gene in set(lr["receptor"])),
                "secreted": bool(gene in set(lr["ligand"])),
                "peripheral": False,
                "integrin": False,
            }
        )

    interaction_rows = []
    for _, row in lr.iterrows():
        interaction_rows.append(
            {
                "partner_a": f"UP_{_safe_id(row['ligand'])}",
                "partner_b": f"UP_{_safe_id(row['receptor'])}",
                "source": "custom_benchmark_lr",
                "annotation_strategy": "curated",
            }
        )

    complex_cols = [
        "complex_name",
        "uniprot_1",
        "uniprot_2",
        "uniprot_3",
        "uniprot_4",
        "pdb_structure",
        "pdb_id",
        "stoichiometry",
        "comments_complex",
        "receptor",
        "receptor_desc",
        "other",
        "other_desc",
        "secreted_highlight",
        "secreted_desc",
        "transmembrane",
        "secreted",
        "peripheral",
        "integrin",
    ]

    pd.DataFrame(gene_rows).to_csv(db_dir / "gene_input.csv", index=False)
    pd.DataFrame(protein_rows).to_csv(db_dir / "protein_input.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(db_dir / "interaction_input.csv", index=False)
    # CellPhoneDB 5.0.1's DB builder assumes complex_input has at least
    # one row when applying over complex subunit columns. Add a harmless
    # dummy complex that is not used by any interaction.
    dummy_uniprot = protein_rows[0]["uniprot"]
    dummy_complex = {col: "" for col in complex_cols}
    dummy_complex.update(
        {
            "complex_name": "DUMMY_UNUSED_COMPLEX",
            "uniprot_1": dummy_uniprot,
            "uniprot_2": dummy_uniprot,
            "uniprot_3": dummy_uniprot,
            "uniprot_4": dummy_uniprot,
            "pdb_structure": "",
            "pdb_id": "",
            "stoichiometry": "",
            "comments_complex": "unused dummy row for CellPhoneDB database builder",
            "receptor": False,
            "other": False,
            "secreted_highlight": False,
            "transmembrane": False,
            "secreted": False,
            "peripheral": False,
            "integrin": False,
        }
    )
    pd.DataFrame([dummy_complex], columns=complex_cols).to_csv(
        db_dir / "complex_input.csv", index=False
    )
    pd.DataFrame(columns=["receptor_id", "TF_symbol"]).to_csv(
        db_dir / "transcription_factor_input.csv", index=False
    )

    db_utils.create_db(str(db_dir))
    db_files = sorted(db_dir.glob("cellphonedb_*.zip"), key=lambda p: p.stat().st_mtime)
    if not db_files:
        raise RuntimeError("CellPhoneDB create_db did not produce a zip file")
    return db_files[-1]


def prepare_inputs(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    expr = pd.read_csv(input_dir / "expression.csv")
    cells = pd.read_csv(input_dir / "cells.csv")

    counts = expr.set_index("cell_id").T
    counts.index.name = "gene_name"
    counts_path = output_dir / "counts_gene_name.csv"
    counts.to_csv(counts_path)

    meta = cells[["cell_id", "cell_type"]].rename(columns={"cell_id": "Cell"})
    meta_path = output_dir / "meta.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "cellphonedb_smoke_status.csv"
    try:
        cpdb_zip = build_minimal_db(args.input_dir, args.output_dir)
        counts_path, meta_path = prepare_inputs(args.input_dir, args.output_dir)
        result_dir = args.output_dir / "cpdb_results"
        result_dir.mkdir(exist_ok=True)
        res = cpdb_analysis_method.call(
            cpdb_file_path=str(cpdb_zip),
            meta_file_path=str(meta_path),
            counts_file_path=str(counts_path),
            counts_data="gene_name",
            output_path=str(result_dir),
            threshold=0.0,
            score_interactions=True,
            threads=1,
            output_suffix="smoke",
        )
        status = {
            "method": "CellPhoneDB",
            "status": "ran_minimal_custom_database_smoke",
            "input_ok": True,
            "run_ok": True,
            "output_level": "cell_type_pair_lr_score",
            "comparable_candidate_edge_ranking": False,
            "reason": "CellPhoneDB API ran using a minimal custom CPDB database built from benchmark LR pairs; this avoids official DB download but is smoke-only, not the official full database baseline.",
            "output_path": str(result_dir),
            "log_path": "",
            "cpdb_file_path": str(cpdb_zip),
            "result_keys": "|".join(sorted(res.keys())),
        }
    except Exception as exc:  # pragma: no cover - smoke script
        status = {
            "method": "CellPhoneDB",
            "status": "failed_minimal_custom_database_smoke",
            "input_ok": True,
            "run_ok": False,
            "output_level": "not_reached",
            "comparable_candidate_edge_ranking": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "output_path": "",
            "log_path": "",
            "cpdb_file_path": "",
            "result_keys": "",
        }
    pd.DataFrame([status]).to_csv(status_path, index=False)
    print(pd.DataFrame([status]).to_string(index=False))


if __name__ == "__main__":
    main()
