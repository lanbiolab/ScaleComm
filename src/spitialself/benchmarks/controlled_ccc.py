"""Controlled ground-truth CCC benchmark generator.

The generator intentionally separates ground-truth communication from simple
"high expression plus short distance" heuristics. Ground truth is assigned by
mechanism-specific rules: direct contact, short/medium paracrine range, and
longer-range chemokine activity restricted to spatial niches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ControlledCCCConfig:
    n_cells: int = 720
    n_background_genes: int = 32
    random_seed: int = 2025
    replicates: int = 5
    coordinate_noise: float = 0.012
    contact_range: float = 0.055
    paracrine_range: float = 0.18
    chemokine_range: float = 0.42
    max_candidate_range: float = 0.45
    negative_edge_fraction: float = 0.35
    candidate_generation: str = "full_radius"
    max_candidates_per_lr: int | None = None
    max_candidates_per_sender_lr: int = 48
    output_dir: str = "outputs/phase1/exp01_controlled_benchmark/datasets"


@dataclass(frozen=True)
class LRSpec:
    lr_id: str
    ligand: str
    receptor: str
    mechanism: str
    sender_cell_type: str
    receiver_cell_type: str
    spatial_range: float
    expression_strength: float
    background_expression: float
    weak_signal: bool = False
    false_positive_high_expression: bool = False
    active_niche: str = "any"
    true_lr: bool = True


CELL_TYPES = ("T_cell", "B_cell", "FDC", "Stromal", "Myeloid", "Endothelial")
NICHES = ("T_zone", "B_follicle", "Germinal_center", "Boundary")


LR_SPECS = (
    LRSpec(
        "DLL4_NOTCH1_contact",
        "DLL4",
        "NOTCH1",
        "juxtacrine_contact",
        "Endothelial",
        "T_cell",
        0.055,
        3.4,
        0.08,
    ),
    LRSpec(
        "JAG1_NOTCH2_contact_weak",
        "JAG1",
        "NOTCH2",
        "juxtacrine_contact",
        "Stromal",
        "B_cell",
        0.055,
        1.35,
        0.06,
        weak_signal=True,
    ),
    LRSpec(
        "IL6_IL6R_paracrine",
        "IL6",
        "IL6R",
        "short_medium_paracrine",
        "Myeloid",
        "T_cell",
        0.18,
        2.8,
        0.12,
    ),
    LRSpec(
        "TGFB1_TGFBR2_paracrine_weak",
        "TGFB1",
        "TGFBR2",
        "short_medium_paracrine",
        "Stromal",
        "Myeloid",
        0.18,
        1.25,
        0.12,
        weak_signal=True,
    ),
    LRSpec(
        "CCL19_CCR7_chemokine_Tzone",
        "CCL19",
        "CCR7",
        "long_range_chemokine_niche",
        "Stromal",
        "T_cell",
        0.42,
        2.4,
        0.10,
        active_niche="T_zone",
    ),
    LRSpec(
        "CXCL13_CXCR5_chemokine_GC_weak",
        "CXCL13",
        "CXCR5",
        "long_range_chemokine_niche",
        "FDC",
        "B_cell",
        0.42,
        1.15,
        0.10,
        weak_signal=True,
        active_niche="Germinal_center",
    ),
    LRSpec(
        "ALB_EGFR_false_positive_high_expression",
        "ALB",
        "EGFR",
        "false_positive_high_expression",
        "Stromal",
        "Endothelial",
        0.42,
        4.5,
        2.4,
        false_positive_high_expression=True,
        true_lr=False,
    ),
    LRSpec(
        "ACTB_ITGB1_false_positive_high_expression",
        "ACTB",
        "ITGB1",
        "false_positive_high_expression",
        "B_cell",
        "T_cell",
        0.42,
        4.2,
        2.7,
        false_positive_high_expression=True,
        true_lr=False,
    ),
)


def generate_controlled_ccc_benchmark(config: ControlledCCCConfig) -> list[Path]:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    replicate_dirs: list[Path] = []
    for replicate in range(config.replicates):
        rng = np.random.default_rng(config.random_seed + replicate)
        rep_dir = out_root / f"replicate_{replicate:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        cells = _make_cells(config, rng)
        lr_pairs = _make_lr_table()
        expression = _make_expression(cells, lr_pairs, config, rng)
        candidates = _make_candidate_interactions(cells, expression, lr_pairs, config, rng)
        truth_tables = _make_truth_tables(candidates, lr_pairs)

        cells.to_csv(rep_dir / "cells.csv", index=False)
        expression.to_csv(rep_dir / "expression.csv", index=False)
        lr_pairs.to_csv(rep_dir / "lr_pairs.csv", index=False)
        candidates.to_csv(rep_dir / "candidate_interactions.csv", index=False)
        truth_tables["ground_truth_table"].to_csv(rep_dir / "ground_truth_table.csv", index=False)
        truth_tables["truth_lr_pairs"].to_csv(rep_dir / "truth_lr_pairs.csv", index=False)
        truth_tables["truth_cell_type_pairs"].to_csv(rep_dir / "truth_cell_type_pairs.csv", index=False)
        truth_tables["truth_spatial_niches"].to_csv(rep_dir / "truth_spatial_niches.csv", index=False)

        metadata = {
            "benchmark": "controlled_ground_truth_ccc",
            "benchmark_version": "v2_mechanism_layered_truth_tables",
            "replicate": replicate,
            "config": asdict(config),
            "schema": {
                "cells": "one row per simulated cell/spot with coordinates, cell type, niche, library size",
                "expression": "wide cell x gene count matrix; first column is cell_id",
                "lr_pairs": "one row per LR pair with mechanism and ground-truth rule metadata",
                "candidate_interactions": "one row per sender-receiver-LR candidate with binary ground_truth label",
                "ground_truth_table": "candidate-level truth table with true_edge, true LR, cell-type-pair, niche, weak/strong, and mechanism labels",
                "truth_lr_pairs": "LR-pair-level truth summary for positive, weak, and high-expression negative controls",
                "truth_cell_type_pairs": "cell-type-pair-by-LR truth summary",
                "truth_spatial_niches": "spatial-niche-by-LR truth summary",
            },
            "label_definition": (
                "Ground truth is mechanism-gated by sender/receiver cell type, spatial range, "
                "and, for chemokine-like pairs, active niche membership. False-positive pairs "
                "are deliberately high-expression but non-communicating."
            ),
            "counts": {
                "cells": int(len(cells)),
                "lr_pairs": int(len(lr_pairs)),
                "candidate_interactions": int(len(candidates)),
                "truth_edges": int(truth_tables["ground_truth_table"]["true_edge"].sum()),
                "truth_lr_positive_pairs": int(truth_tables["truth_lr_pairs"]["true_lr_pair"].sum()),
                "truth_cell_type_lr_pairs": int(
                    truth_tables["truth_cell_type_pairs"]["true_cell_type_pair"].sum()
                ),
                "truth_spatial_niche_lr_pairs": int(
                    truth_tables["truth_spatial_niches"]["true_spatial_niche"].sum()
                ),
                "positive_interactions": int(candidates["ground_truth"].sum()),
                "weak_positive_interactions": int(
                    candidates.loc[candidates["weak_signal"], "ground_truth"].sum()
                ),
            },
        }
        with (rep_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        replicate_dirs.append(rep_dir)

    manifest = pd.DataFrame(
        {
            "replicate": list(range(config.replicates)),
            "path": [str(path) for path in replicate_dirs],
        }
    )
    manifest.to_csv(out_root / "manifest.csv", index=False)
    return replicate_dirs


def _make_cells(config: ControlledCCCConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_cells
    centers = {
        "T_zone": np.array([0.28, 0.68]),
        "B_follicle": np.array([0.68, 0.70]),
        "Germinal_center": np.array([0.72, 0.38]),
        "Boundary": np.array([0.38, 0.34]),
    }
    niche_probs = np.array([0.34, 0.25, 0.22, 0.19])
    niches = rng.choice(NICHES, size=n, p=niche_probs)

    coords = np.zeros((n, 2), dtype=float)
    for i, niche in enumerate(niches):
        radius = 0.11 if niche != "Boundary" else 0.17
        coords[i] = centers[niche] + rng.normal(0.0, radius, size=2)

    coords += rng.normal(0.0, config.coordinate_noise, size=coords.shape)
    coords = np.clip(coords, 0.02, 0.98)

    cell_types = [_sample_cell_type(niche, rng) for niche in niches]
    library_size = rng.lognormal(mean=9.0, sigma=0.35, size=n).astype(int)

    return pd.DataFrame(
        {
            "cell_id": [f"cell_{i:04d}" for i in range(n)],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cell_type": cell_types,
            "niche": niches,
            "library_size": library_size,
        }
    )


def _sample_cell_type(niche: str, rng: np.random.Generator) -> str:
    probs = {
        "T_zone": [0.56, 0.08, 0.02, 0.19, 0.10, 0.05],
        "B_follicle": [0.10, 0.58, 0.08, 0.12, 0.07, 0.05],
        "Germinal_center": [0.08, 0.48, 0.25, 0.08, 0.07, 0.04],
        "Boundary": [0.22, 0.16, 0.02, 0.20, 0.30, 0.10],
    }[niche]
    return str(rng.choice(CELL_TYPES, p=probs))


def _make_lr_table() -> pd.DataFrame:
    return pd.DataFrame(asdict(spec) for spec in LR_SPECS)


def _make_truth_tables(candidates: pd.DataFrame, lr_pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lr_meta = lr_pairs[
        [
            "lr_id",
            "spatial_range",
            "expression_strength",
            "background_expression",
            "active_niche",
            "true_lr",
        ]
    ].copy()
    truth = candidates.merge(lr_meta, on="lr_id", how="left", validate="many_to_one")
    truth = truth.rename(
        columns={
            "ground_truth": "true_edge",
            "mechanism": "mechanism_type",
            "sender_niche": "sender_spatial_niche",
            "receiver_niche": "receiver_spatial_niche",
        }
    )
    truth["true_edge"] = truth["true_edge"].astype(int)
    truth["true_lr_pair"] = truth["true_lr"].astype(bool) & ~truth[
        "false_positive_high_expression"
    ].astype(bool)
    truth["true_cell_type_pair"] = truth["type_rule_satisfied"].astype(bool) & truth[
        "true_lr_pair"
    ].astype(bool)
    truth["true_spatial_niche"] = truth["niche_rule_satisfied"].astype(bool) & truth[
        "true_lr_pair"
    ].astype(bool)
    truth["weak_or_strong"] = np.where(truth["weak_signal"].astype(bool), "weak", "strong")
    truth.loc[truth["false_positive_high_expression"].astype(bool), "weak_or_strong"] = (
        "false_positive_high_expression"
    )

    ground_truth_table = truth[
        [
            "candidate_id",
            "sender_id",
            "receiver_id",
            "lr_id",
            "ligand",
            "receptor",
            "mechanism_type",
            "sender_cell_type",
            "receiver_cell_type",
            "sender_spatial_niche",
            "receiver_spatial_niche",
            "active_niche",
            "distance",
            "spatial_range",
            "expression_strength",
            "background_expression",
            "weak_signal",
            "weak_or_strong",
            "false_positive_high_expression",
            "true_lr_pair",
            "true_cell_type_pair",
            "true_spatial_niche",
            "type_rule_satisfied",
            "range_rule_satisfied",
            "niche_rule_satisfied",
            "true_edge",
        ]
    ].copy()

    truth_lr_pairs = (
        truth.groupby(
            [
                "lr_id",
                "ligand",
                "receptor",
                "mechanism_type",
                "weak_signal",
                "weak_or_strong",
                "false_positive_high_expression",
                "sender_cell_type",
                "receiver_cell_type",
                "spatial_range",
                "active_niche",
            ],
            dropna=False,
        )
        .agg(
            candidate_edges=("candidate_id", "count"),
            true_edges=("true_edge", "sum"),
            true_cell_type_candidates=("true_cell_type_pair", "sum"),
            true_spatial_niche_candidates=("true_spatial_niche", "sum"),
            positive_median_distance=("distance", lambda values: _median_where(values, truth.loc[values.index, "true_edge"])),
        )
        .reset_index()
    )
    truth_lr_pairs["true_lr_pair"] = (
        (truth_lr_pairs["true_edges"] > 0)
        & ~truth_lr_pairs["false_positive_high_expression"].astype(bool)
    )

    truth_cell_type_pairs = (
        truth.groupby(
            [
                "lr_id",
                "mechanism_type",
                "weak_or_strong",
                "sender_cell_type",
                "receiver_cell_type",
            ],
            dropna=False,
        )
        .agg(candidate_edges=("candidate_id", "count"), true_edges=("true_edge", "sum"))
        .reset_index()
    )
    truth_cell_type_pairs["true_cell_type_pair"] = truth_cell_type_pairs["true_edges"] > 0

    truth_spatial_niches = (
        truth.groupby(
            [
                "lr_id",
                "mechanism_type",
                "weak_or_strong",
                "sender_spatial_niche",
                "receiver_spatial_niche",
                "active_niche",
            ],
            dropna=False,
        )
        .agg(candidate_edges=("candidate_id", "count"), true_edges=("true_edge", "sum"))
        .reset_index()
    )
    truth_spatial_niches["true_spatial_niche"] = truth_spatial_niches["true_edges"] > 0

    return {
        "ground_truth_table": ground_truth_table,
        "truth_lr_pairs": truth_lr_pairs,
        "truth_cell_type_pairs": truth_cell_type_pairs,
        "truth_spatial_niches": truth_spatial_niches,
    }


def _median_where(values: pd.Series, mask: pd.Series) -> float:
    selected = values[mask.astype(bool)]
    if selected.empty:
        return float("nan")
    return float(selected.median())


def _make_expression(
    cells: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    config: ControlledCCCConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    lr_genes = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]))
    background_genes = [f"BG{i:03d}" for i in range(config.n_background_genes)]
    genes = lr_genes + background_genes

    expr = pd.DataFrame({"cell_id": cells["cell_id"]})
    size_factor = cells["library_size"].to_numpy() / np.median(cells["library_size"])

    for gene in genes:
        base = rng.uniform(0.02, 0.16)
        mean = np.full(len(cells), base, dtype=float)

        for spec in LR_SPECS:
            if gene == spec.ligand:
                mask = cells["cell_type"].eq(spec.sender_cell_type).to_numpy()
                mean[mask] += spec.background_expression + spec.expression_strength
            if gene == spec.receptor:
                mask = cells["cell_type"].eq(spec.receiver_cell_type).to_numpy()
                mean[mask] += spec.background_expression + 0.7 * spec.expression_strength

        if gene.startswith("BG"):
            cell_type_effect = rng.uniform(0.0, 0.35, size=len(CELL_TYPES))
            for idx, cell_type in enumerate(CELL_TYPES):
                mean[cells["cell_type"].eq(cell_type).to_numpy()] += cell_type_effect[idx]

        dropout_prob = np.clip(np.exp(-mean) * 0.42, 0.04, 0.70)
        counts = rng.poisson(mean * size_factor * 3.0)
        counts[rng.random(len(cells)) < dropout_prob] = 0
        expr[gene] = counts.astype(int)

    return expr


def _make_candidate_interactions(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    config: ControlledCCCConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    coords = cells[["x", "y"]].to_numpy()
    if config.candidate_generation not in {"full_radius", "sparse_sampled"}:
        raise ValueError(
            "candidate_generation must be either 'full_radius' or 'sparse_sampled'"
        )
    distances = None
    if config.candidate_generation == "full_radius":
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((delta * delta).sum(axis=2))
        np.fill_diagonal(distances, np.inf)

    rows = []
    expr_index = expression.set_index("cell_id")
    for lr in lr_pairs.itertuples(index=False):
        max_range = float(lr.spatial_range)
        search_range = min(float(config.max_candidate_range), max(config.contact_range, max_range))
        if config.candidate_generation == "sparse_sampled":
            within = _sample_sparse_controlled_pairs(
                cells,
                coords,
                lr,
                search_range=search_range,
                config=config,
                rng=rng,
            )
        else:
            assert distances is not None
            within = np.argwhere(distances <= search_range)
        lr_rows = []
        for sender_idx, receiver_idx in within:
            sender = cells.iloc[int(sender_idx)]
            receiver = cells.iloc[int(receiver_idx)]
            if distances is None:
                distance = float(np.linalg.norm(coords[int(sender_idx)] - coords[int(receiver_idx)]))
            else:
                distance = float(distances[sender_idx, receiver_idx])

            type_ok = (
                sender["cell_type"] == lr.sender_cell_type
                and receiver["cell_type"] == lr.receiver_cell_type
            )
            range_ok = distance <= max_range
            niche_ok = lr.active_niche == "any" or (
                sender["niche"] == lr.active_niche and receiver["niche"] == lr.active_niche
            )
            mechanism_ok = bool(lr.true_lr and type_ok and range_ok and niche_ok)

            contact_penalty = 1.0
            if lr.mechanism == "juxtacrine_contact":
                mechanism_ok = mechanism_ok and distance <= config.contact_range
                contact_penalty = 1.0 if distance <= config.contact_range else 0.0

            probability = _signal_probability(lr.mechanism, distance, max_range)
            if bool(lr.weak_signal):
                probability *= 0.52
            label = bool(mechanism_ok and rng.random() < probability)

            ligand_expr = float(expr_index.loc[sender["cell_id"], lr.ligand])
            receptor_expr = float(expr_index.loc[receiver["cell_id"], lr.receptor])
            heuristic_score = np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)
            if lr.false_positive_high_expression:
                label = False

            keep_negative = rng.random() < config.negative_edge_fraction
            if label or keep_negative:
                lr_rows.append(
                    {
                        "sender_id": sender["cell_id"],
                        "receiver_id": receiver["cell_id"],
                        "lr_id": lr.lr_id,
                        "ligand": lr.ligand,
                        "receptor": lr.receptor,
                        "mechanism": lr.mechanism,
                        "distance": distance,
                        "sender_cell_type": sender["cell_type"],
                        "receiver_cell_type": receiver["cell_type"],
                        "sender_niche": sender["niche"],
                        "receiver_niche": receiver["niche"],
                        "ground_truth": int(label),
                        "weak_signal": bool(lr.weak_signal),
                        "false_positive_high_expression": bool(
                            lr.false_positive_high_expression
                        ),
                        "type_rule_satisfied": bool(type_ok),
                        "range_rule_satisfied": bool(range_ok),
                        "niche_rule_satisfied": bool(niche_ok),
                        "ligand_expr": ligand_expr,
                        "receptor_expr": receptor_expr,
                        "coexpression_distance_score": float(heuristic_score * contact_penalty),
                    }
                )
        rows.extend(_cap_lr_rows(lr_rows, config.max_candidates_per_lr, rng))

    candidates = pd.DataFrame(rows)
    candidates.insert(0, "candidate_id", [f"cand_{i:07d}" for i in range(len(candidates))])
    return candidates


def _sample_sparse_controlled_pairs(
    cells: pd.DataFrame,
    coords: np.ndarray,
    lr,
    *,
    search_range: float,
    config: ControlledCCCConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bounded candidate set while enriching mechanism-valid pairs."""
    all_idx = np.arange(len(cells), dtype=np.int64)
    sender_type = cells["cell_type"].eq(lr.sender_cell_type).to_numpy()
    receiver_type = cells["cell_type"].eq(lr.receiver_cell_type).to_numpy()
    if lr.active_niche == "any":
        sender_niche = np.ones(len(cells), dtype=bool)
        receiver_niche = np.ones(len(cells), dtype=bool)
    else:
        sender_niche = cells["niche"].eq(lr.active_niche).to_numpy()
        receiver_niche = cells["niche"].eq(lr.active_niche).to_numpy()

    max_pairs = config.max_candidates_per_lr
    valid_budget = None if max_pairs is None else max(int(max_pairs * 0.55), 1)
    background_budget = None if max_pairs is None else max(int(max_pairs * 1.35), 1)
    valid_pairs = _sample_radius_pairs(
        coords,
        np.flatnonzero(sender_type & sender_niche),
        np.flatnonzero(receiver_type & receiver_niche),
        radius=float(lr.spatial_range),
        max_per_sender=config.max_candidates_per_sender_lr,
        max_pairs=valid_budget,
        rng=rng,
    )
    background_pairs = _sample_radius_pairs(
        coords,
        all_idx,
        all_idx,
        radius=search_range,
        max_per_sender=config.max_candidates_per_sender_lr,
        max_pairs=background_budget,
        rng=rng,
    )
    if len(valid_pairs) == 0:
        pairs = background_pairs
    elif len(background_pairs) == 0:
        pairs = valid_pairs
    else:
        pairs = np.vstack([valid_pairs, background_pairs])
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(pairs, axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def _sample_radius_pairs(
    coords: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    *,
    radius: float,
    max_per_sender: int,
    max_pairs: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(sender_indices) == 0 or len(receiver_indices) == 0 or radius <= 0:
        return np.empty((0, 2), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("sparse_sampled candidate generation requires scipy") from exc

    receiver_indices = np.asarray(receiver_indices, dtype=np.int64)
    tree = cKDTree(coords[receiver_indices])
    rows: list[tuple[int, int]] = []
    sender_order = np.asarray(sender_indices, dtype=np.int64).copy()
    rng.shuffle(sender_order)
    per_sender = max(int(max_per_sender), 1)
    for sender_idx in sender_order:
        local = tree.query_ball_point(coords[int(sender_idx)], r=float(radius))
        if not local:
            continue
        receiver_global = receiver_indices[np.asarray(local, dtype=np.int64)]
        receiver_global = receiver_global[receiver_global != int(sender_idx)]
        if len(receiver_global) == 0:
            continue
        if len(receiver_global) > per_sender:
            receiver_global = rng.choice(receiver_global, size=per_sender, replace=False)
        rows.extend((int(sender_idx), int(receiver_idx)) for receiver_idx in receiver_global)
        if max_pairs is not None and len(rows) >= max_pairs:
            rows = rows[: int(max_pairs)]
            break
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _cap_lr_rows(
    rows: list[dict],
    max_candidates_per_lr: int | None,
    rng: np.random.Generator,
) -> list[dict]:
    if max_candidates_per_lr is None or len(rows) <= max_candidates_per_lr:
        return rows
    positives = [row for row in rows if int(row["ground_truth"]) == 1]
    negatives = [row for row in rows if int(row["ground_truth"]) == 0]
    cap = int(max_candidates_per_lr)
    if len(positives) >= cap:
        keep = rng.choice(len(positives), size=cap, replace=False)
        return [positives[int(i)] for i in keep]
    remaining = cap - len(positives)
    if len(negatives) > remaining:
        keep = rng.choice(len(negatives), size=remaining, replace=False)
        negatives = [negatives[int(i)] for i in keep]
    return positives + negatives


def _signal_probability(mechanism: str, distance: float, spatial_range: float) -> float:
    if mechanism == "juxtacrine_contact":
        return 0.92
    if mechanism == "short_medium_paracrine":
        return float(0.18 + 0.68 * np.exp(-distance / max(spatial_range, 1e-6)))
    if mechanism == "long_range_chemokine_niche":
        # Long-range signals should not be reducible to nearest-neighbor contact.
        peak = 0.5 * spatial_range
        width = 0.28 * spatial_range
        return float(0.20 + 0.62 * np.exp(-((distance - peak) ** 2) / (2 * width**2)))
    return 0.0
