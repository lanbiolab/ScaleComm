#!/usr/bin/env python
"""Build stage-wise SpatialSelf inputs from Cadinu 2024 mouse colon MERFISH.

The builder uses one representative MERFISH slice per disease stage by default
and writes the standard project replicate schema:
cells.csv, expression.csv, lr_pairs.csv, candidate_interactions.csv.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_STAGE_FILES = {
    "Healthy_D0": "082421_D0_m6_1_slice_3.csv",
    "DSS3": "092421_D3_m3_1_slice_4.csv",
    "DSS9": "100221_D9_m5_2_slice_3.csv",
    "DSS21": "082421_D21_m2_1_slice_2.csv",
}

STAGE_ORDER = {"Healthy_D0": 0, "DSS3": 3, "DSS9": 9, "DSS21": 21}


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path
    output_dir: Path
    max_cells_per_stage: int
    max_lr_pairs: int
    max_candidates_per_lr: int
    max_candidates_per_sender_lr: int
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/optional/mouse_colon_ibd_merfish/raw")
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mouse_colon_ibd_merfish/stagewise_replicates",
    )
    parser.add_argument("--max-cells-per-stage", type=int, default=900)
    parser.add_argument("--max-lr-pairs", type=int, default=260)
    parser.add_argument("--max-candidates-per-lr", type=int, default=2400)
    parser.add_argument("--max-candidates-per-sender-lr", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    cfg = BuildConfig(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        max_cells_per_stage=int(args.max_cells_per_stage),
        max_lr_pairs=int(args.max_lr_pairs),
        max_candidates_per_lr=int(args.max_candidates_per_lr),
        max_candidates_per_sender_lr=int(args.max_candidates_per_sender_lr),
        seed=int(args.seed),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    _check_required_files(cfg.raw_dir)
    lr_pairs_all = load_lr_pairs(cfg.raw_dir, cfg.max_lr_pairs)
    build_rows = []
    for stage, filename in DEFAULT_STAGE_FILES.items():
        replicate_dir = cfg.output_dir / f"mouse_colon_ibd_merfish_{stage}"
        replicate_dir.mkdir(parents=True, exist_ok=True)
        print(f"building {stage} from {filename}")
        cells, expression = build_stage_cells_expression(
            raw_dir=cfg.raw_dir,
            stage=stage,
            filename=filename,
            max_cells=cfg.max_cells_per_stage,
            seed=cfg.seed + STAGE_ORDER[stage],
        )
        lr_pairs = select_stage_lr_pairs(lr_pairs_all, expression, cfg.max_lr_pairs)
        candidates = build_candidates(
            cells,
            expression,
            lr_pairs,
            dataset_key=f"mouse_colon_ibd_merfish_{stage}",
            max_candidates_per_lr=cfg.max_candidates_per_lr,
            max_candidates_per_sender_lr=cfg.max_candidates_per_sender_lr,
            seed=cfg.seed + 100 + STAGE_ORDER[stage],
        )
        cells.to_csv(replicate_dir / "cells.csv", index=False)
        expression.to_csv(replicate_dir / "expression.csv", index=False)
        lr_pairs.to_csv(replicate_dir / "lr_pairs.csv", index=False)
        candidates.to_csv(replicate_dir / "candidate_interactions.csv", index=False)
        metadata = {
            "dataset": "Cadinu 2024 mouse colon IBD MERFISH",
            "doi": "10.5061/dryad.rjdfn2zh3",
            "primary_article": "10.1016/j.cell.2024.03.013",
            "stage": stage,
            "stage_day": STAGE_ORDER[stage],
            "source_slice_file": filename,
            "builder": "scripts/current/build_mouse_colon_ibd_merfish_replicates.py",
            "claim_boundary": "Representative-slice pilot for stage-wise communication trajectories; not a full 1.35M-cell atlas.",
            "n_cells": int(len(cells)),
            "n_expression_genes": int(expression.shape[1] - 1),
            "n_lr_pairs": int(len(lr_pairs)),
            "n_candidates": int(len(candidates)),
        }
        (replicate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        build_rows.append({"stage": stage, "replicate_dir": str(replicate_dir), **metadata})
    pd.DataFrame(build_rows).to_csv(cfg.output_dir / "stagewise_replicate_manifest.csv", index=False)
    print(pd.DataFrame(build_rows)[["stage", "n_cells", "n_expression_genes", "n_lr_pairs", "n_candidates"]])


def _check_required_files(raw_dir: Path) -> None:
    required = ["cell_properties.csv", "gene_names.csv", "ligand_receptor_pair_masterlist.csv", *DEFAULT_STAGE_FILES.values()]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Cadinu MERFISH files under "
            f"{raw_dir}: {missing}. Run scripts/current/download_mouse_colon_ibd_merfish.py "
            "or place the Dryad files there manually."
        )


def load_lr_pairs(raw_dir: Path, max_lr_pairs: int) -> pd.DataFrame:
    lr_raw = pd.read_csv(raw_dir / "ligand_receptor_pair_masterlist.csv")
    lr_raw.columns = [str(col).strip() for col in lr_raw.columns]
    ligand_col = _first_existing(lr_raw, ["Ligand", "ligand"])
    receptor_col = _first_existing(lr_raw, ["Receptor", "receptor"])
    category_col = _first_existing(lr_raw, ["Category", "category"], required=False)
    subcategory_col = _first_existing(lr_raw, ["Sub-category", "Sub_category", "subcategory"], required=False)
    lr = lr_raw[[ligand_col, receptor_col]].rename(columns={ligand_col: "ligand", receptor_col: "receptor"}).copy()
    lr["ligand"] = lr["ligand"].astype(str).str.strip()
    lr["receptor"] = lr["receptor"].astype(str).str.strip()
    lr = lr[(lr["ligand"] != "") & (lr["receptor"] != "")].drop_duplicates(["ligand", "receptor"])
    lr["biological_category"] = (
        lr_raw.loc[lr.index, category_col].astype(str).str.strip() if category_col else "other"
    )
    lr["subcategory"] = (
        lr_raw.loc[lr.index, subcategory_col].astype(str).str.strip() if subcategory_col else ""
    )
    lr["mechanism"] = [
        infer_mechanism(category, subcategory, ligand, receptor)
        for category, subcategory, ligand, receptor in zip(
            lr["biological_category"], lr["subcategory"], lr["ligand"], lr["receptor"]
        )
    ]
    lr["spatial_range"] = lr["mechanism"].map(
        {
            "juxtacrine_contact": 35.0,
            "short_medium_paracrine": 180.0,
            "long_range_chemokine_niche": 420.0,
        }
    ).fillna(180.0)
    lr["active_niche"] = "any"
    lr["control_name"] = lr["ligand"] + "-" + lr["receptor"]
    lr["control_class"] = "real_lr"
    lr["false_positive_high_expression"] = False
    lr["true_lr"] = True
    lr["source_count"] = 1
    lr["selected_reason"] = "Cadinu2024_KEGG_masterlist"
    priority = lr["mechanism"].map({"long_range_chemokine_niche": 0, "short_medium_paracrine": 1, "juxtacrine_contact": 2})
    lr = lr.assign(_priority=priority).sort_values(["_priority", "ligand", "receptor"]).drop(columns="_priority")
    lr = lr.head(int(max_lr_pairs)).reset_index(drop=True)
    lr.insert(0, "lr_id", [f"cadinu_lr_{idx:04d}_{row.ligand}_{row.receptor}" for idx, row in lr.iterrows()])
    return lr[
        [
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
            "spatial_range",
            "active_niche",
            "control_name",
            "control_class",
            "biological_category",
            "false_positive_high_expression",
            "true_lr",
            "source_count",
            "selected_reason",
        ]
    ]


def build_stage_cells_expression(
    *,
    raw_dir: Path,
    stage: str,
    filename: str,
    max_cells: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    slice_id = Path(filename).stem
    props = pd.read_csv(raw_dir / "cell_properties.csv", low_memory=False)
    slice_col = _first_existing(props, ["Slice_ID", "slice_id"])
    cell_id_col = _first_existing(props, ["cell_IDs", "cell_name", "cell_id", "Cell_ID"])
    sample_col = _first_existing(props, ["Sample_type"], required=False)
    props = props[props[slice_col].astype(str) == slice_id].copy()
    if props.empty:
        raise ValueError(f"No cell_properties rows found for Slice_ID={slice_id}")
    props["_cell_id"] = props[cell_id_col].astype(str)

    rna = pd.read_csv(raw_dir / filename, low_memory=False)
    gene_col = _first_existing(rna, ["gene", "Gene"])
    cell_col = _first_existing(rna, ["cell_name", "cell"], required=False)
    if cell_col is None:
        raise ValueError(f"{filename} does not contain cell_name or cell column")
    rna = rna[rna[cell_col].notna()].copy()
    rna[cell_col] = rna[cell_col].astype(str)
    if "is_noise" in rna.columns:
        rna = rna[~rna["is_noise"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if cell_col == "cell":
        rna = rna[rna[cell_col] != "0"]

    valid_cells = set(props["_cell_id"]) & set(rna[cell_col])
    if not valid_cells:
        # Some exports use numeric cell in RNA and string cell_IDs in properties.
        props["_cell_id"] = props["_cell_id"].str.replace(r"^cell_", "", regex=True)
        valid_cells = set(props["_cell_id"]) & set(rna[cell_col])
    props = props[props["_cell_id"].isin(valid_cells)].copy()
    if props.empty:
        raise ValueError(f"No overlap between {filename} RNA cell IDs and cell_properties")
    props = stratified_sample(props, max_cells=max_cells, rng=rng)
    selected_cells = props["_cell_id"].astype(str).tolist()
    rna = rna[rna[cell_col].isin(selected_cells)].copy()

    expression = (
        rna.groupby([cell_col, gene_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(selected_cells, fill_value=0)
        .reset_index()
        .rename(columns={cell_col: "cell_id"})
    )
    # Ensure coordinates and annotations follow selected_cells order.
    props = props.set_index("_cell_id").loc[selected_cells].reset_index()
    x_col = _first_existing(props, ["x", "X"])
    y_col = _first_existing(props, ["y", "Y"])
    cells = pd.DataFrame(
        {
            "cell_id": selected_cells,
            "x": pd.to_numeric(props[x_col], errors="coerce").to_numpy(float),
            "y": pd.to_numeric(props[y_col], errors="coerce").to_numpy(float),
            "cell_type": props[_first_existing(props, ["Tier3", "Tier2", "Tier1"], required=False)].fillna("unknown").astype(str)
            if any(col in props.columns for col in ["Tier3", "Tier2", "Tier1"])
            else "unknown",
            "region": props[_first_existing(props, ["Sample_type"], required=False)].fillna(stage).astype(str)
            if sample_col
            else stage,
            "niche": props[_first_existing(props, ["Leiden_neigh", "Tier2", "Tier1"], required=False)].fillna("unknown").astype(str)
            if any(col in props.columns for col in ["Leiden_neigh", "Tier2", "Tier1"])
            else "unknown",
        }
    )
    cells["library_size"] = expression.drop(columns=["cell_id"]).sum(axis=1).to_numpy(float)
    cells["stage"] = stage
    cells["stage_day"] = STAGE_ORDER[stage]
    cells["slice_id"] = slice_id
    return cells, expression


def stratified_sample(props: pd.DataFrame, *, max_cells: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(props) <= max_cells:
        return props.copy()
    strata_col = _first_existing(props, ["Tier1", "Tier2", "Leiden_neigh"], required=False)
    if strata_col is None:
        return props.sample(n=max_cells, random_state=int(rng.integers(0, 1_000_000))).copy()
    parts = []
    grouped = list(props.groupby(strata_col, dropna=False))
    base = max(8, max_cells // max(len(grouped), 1))
    remaining = max_cells
    for _, sub in grouped:
        take = min(len(sub), base, remaining)
        if take > 0:
            parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
            remaining -= take
    sampled = pd.concat(parts) if parts else props.iloc[:0].copy()
    if len(sampled) < max_cells:
        rest = props.drop(index=sampled.index)
        if not rest.empty:
            sampled = pd.concat(
                [
                    sampled,
                    rest.sample(n=min(max_cells - len(sampled), len(rest)), random_state=int(rng.integers(0, 1_000_000))),
                ]
            )
    return sampled.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def select_stage_lr_pairs(lr_pairs: pd.DataFrame, expression: pd.DataFrame, max_lr_pairs: int) -> pd.DataFrame:
    genes = set(expression.columns) - {"cell_id"}
    lr = lr_pairs[lr_pairs["ligand"].isin(genes) & lr_pairs["receptor"].isin(genes)].copy()
    if lr.empty:
        raise ValueError("No ligand-receptor pairs are covered by the selected MERFISH gene panel")
    return lr.head(int(max_lr_pairs)).reset_index(drop=True)


def build_candidates(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    dataset_key: str,
    max_candidates_per_lr: int,
    max_candidates_per_sender_lr: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expr = expression.set_index("cell_id")
    cell_ids = cells["cell_id"].astype(str).tolist()
    coords = cells[["x", "y"]].to_numpy(float)
    tree = cKDTree(coords)
    rows = []
    for lr_row in lr_pairs.itertuples(index=False):
        radius = float(lr_row.spatial_range)
        ligand_values = expr[lr_row.ligand].reindex(cell_ids).to_numpy(float)
        receptor_values = expr[lr_row.receptor].reindex(cell_ids).to_numpy(float)
        sender_order = np.argsort(-(np.log1p(ligand_values)))
        lr_rows = []
        for sender_idx in sender_order:
            if ligand_values[sender_idx] <= 0:
                continue
            receiver_idx = tree.query_ball_point(coords[sender_idx], r=radius)
            receiver_idx = [idx for idx in receiver_idx if idx != sender_idx and receptor_values[idx] > 0]
            if not receiver_idx:
                continue
            receiver_idx = sorted(receiver_idx, key=lambda idx: np.linalg.norm(coords[sender_idx] - coords[idx]))
            receiver_idx = receiver_idx[: int(max_candidates_per_sender_lr)]
            for dst_idx in receiver_idx:
                distance = float(np.linalg.norm(coords[sender_idx] - coords[dst_idx]))
                lr_rows.append(
                    {
                        "sender_id": cell_ids[sender_idx],
                        "receiver_id": cell_ids[dst_idx],
                        "lr_id": lr_row.lr_id,
                        "ligand": lr_row.ligand,
                        "receptor": lr_row.receptor,
                        "mechanism": lr_row.mechanism,
                        "distance": distance,
                        "sender_cell_type": cells.iloc[sender_idx]["cell_type"],
                        "receiver_cell_type": cells.iloc[dst_idx]["cell_type"],
                        "sender_niche": cells.iloc[sender_idx]["niche"],
                        "receiver_niche": cells.iloc[dst_idx]["niche"],
                        "weak_signal": bool(ligand_values[sender_idx] <= np.quantile(ligand_values[ligand_values > 0], 0.35)),
                        "false_positive_high_expression": False,
                        "true_lr": True,
                        "type_rule_satisfied": True,
                        "range_rule_satisfied": distance <= radius,
                        "niche_rule_satisfied": True,
                        "ligand_expr": float(ligand_values[sender_idx]),
                        "receptor_expr": float(receptor_values[dst_idx]),
                        "coexpression_distance_score": float(
                            np.log1p(ligand_values[sender_idx])
                            * np.log1p(receptor_values[dst_idx])
                            / (1.0 + distance / max(radius, 1.0))
                        ),
                    }
                )
        if len(lr_rows) > max_candidates_per_lr:
            keep = rng.choice(len(lr_rows), size=int(max_candidates_per_lr), replace=False)
            lr_rows = [lr_rows[int(idx)] for idx in keep]
        rows.extend(lr_rows)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError(f"No candidates generated for {dataset_key}")
    candidates.insert(0, "candidate_id", [f"{dataset_key}_cand_{idx:09d}" for idx in range(len(candidates))])
    return candidates


def infer_mechanism(category: str, subcategory: str, ligand: str, receptor: str) -> str:
    text = f"{category} {subcategory} {ligand} {receptor}".lower()
    if any(key in text for key in ["chemokine", "ccr", "cxcr", "ackr", "xcr"]):
        return "long_range_chemokine_niche"
    if any(key in text for key in ["notch", "cadherin", "adhesion", "integrin", "junction"]):
        return "juxtacrine_contact"
    return "short_medium_paracrine"


def _first_existing(df: pd.DataFrame, names: list[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"None of columns {names} found in {list(df.columns)[:20]}")
    return None


if __name__ == "__main__":
    main()
