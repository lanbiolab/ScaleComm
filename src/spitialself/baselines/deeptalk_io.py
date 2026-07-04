"""DeepTalk input preparation for generated CCC benchmark replicates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_deeptalk_inputs(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
    *,
    max_pca_components: int = 128,
) -> pd.DataFrame:
    """Create per-LR DeepTalk prediction inputs from a benchmark replicate.

    DeepTalk prediction expects one directory per LR pair containing `test.txt`,
    `ent2id.txt`, `rel2id.txt`, `edgelist.txt`, and node embeddings. This
    function prepares those files and writes candidate maps so model outputs can
    later be projected back to `candidate_id`.
    """

    _check_columns(cells, {"cell_id", "x", "y"}, "cells")
    _check_columns(expression, {"cell_id"}, "expression")
    _check_columns(
        candidates,
        {"candidate_id", "sender_id", "receiver_id", "ligand", "receptor", "lr_id"},
        "candidates",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_order = cells["cell_id"].astype(str).to_list()
    cell_to_node = {cell_id: str(index) for index, cell_id in enumerate(cell_order)}

    expr = expression.set_index("cell_id").reindex(cell_order)
    if expr.isnull().any().any():
        missing = expr.index[expr.isnull().any(axis=1)].to_list()[:5]
        raise ValueError(f"Expression table missing cells present in cells.csv: {missing}")

    expr.T.to_csv(output_dir / "st_data_genes_by_cells.tsv", sep="\t")
    cells[["cell_id", "x", "y"]].to_csv(output_dir / "st_location.csv", index=False)
    cells[["x", "y"]].to_csv(output_dir / "st_location.txt", sep=" ", index=False, header=False)

    manifest_rows = []
    for lr_id, subset in candidates.groupby("lr_id", sort=False):
        ligand = str(subset["ligand"].iloc[0])
        receptor = str(subset["receptor"].iloc[0])
        relation = _safe_relation_name(ligand, receptor)
        lr_dir = output_dir / "data" / relation
        lr_dir.mkdir(parents=True, exist_ok=True)

        line_rows = []
        map_rows = []
        for line_index, row in enumerate(subset.reset_index(drop=True).itertuples(index=False)):
            sender = cell_to_node.get(str(row.sender_id))
            receiver = cell_to_node.get(str(row.receiver_id))
            if sender is None or receiver is None:
                continue
            # DeepTalk's prediction graph loader only materializes edges when the
            # fourth token is absent or equal to "1". Prediction labels here are
            # placeholders for graph construction rather than benchmark truth, so
            # we emit "1" to preserve the candidate graph for inference.
            line_rows.append((relation.lower(), sender, receiver, "1"))
            map_rows.append(
                {
                    "line_index": line_index,
                    "candidate_id": row.candidate_id,
                    "sender_id": row.sender_id,
                    "receiver_id": row.receiver_id,
                    "lr_id": lr_id,
                    "ligand": ligand,
                    "receptor": receptor,
                }
            )

        if not line_rows:
            continue
        pd.DataFrame(line_rows).to_csv(lr_dir / "test.txt", sep=" ", index=False, header=False)
        pd.DataFrame(map_rows).to_csv(lr_dir / "candidate_map.csv", index=False)
        _write_deeptalk_graph_files(lr_dir, line_rows)
        _write_pca_embedding(expr, lr_dir / "data_pca_pre.emd", max_pca_components=max_pca_components)

        manifest_rows.append(
            {
                "lr_id": lr_id,
                "ligand": ligand,
                "receptor": receptor,
                "deeptalk_relation": relation,
                "n_candidates": len(line_rows),
                "lr_dir": str(lr_dir),
                "status": "input_ready_model_required",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "deeptalk_input_manifest.csv", index=False)
    with (output_dir / "deeptalk_input_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, indent=2, ensure_ascii=True)
    return manifest


def _write_deeptalk_graph_files(lr_dir: Path, line_rows: list[tuple[str, str, str, str]]) -> None:
    entities = []
    relations = []
    edges = []
    for relation, sender, receiver, _ in line_rows:
        if sender not in entities:
            entities.append(sender)
        if receiver not in entities:
            entities.append(receiver)
        if relation not in relations:
            relations.append(relation)

    ent2id = {entity: index for index, entity in enumerate(entities)}
    rel2id = {relation: index for index, relation in enumerate(relations)}
    rel2id.update({f"{relation}_reverse": index + len(relations) for relation, index in rel2id.items()})
    for _, sender, receiver, _ in line_rows:
        edges.append((ent2id[sender], ent2id[receiver]))

    with (lr_dir / "ent2id.txt").open("w", encoding="utf-8") as handle:
        json.dump(ent2id, handle)
    with (lr_dir / "rel2id.txt").open("w", encoding="utf-8") as handle:
        json.dump(rel2id, handle)
    pd.DataFrame(edges).to_csv(lr_dir / "edgelist.txt", sep=" ", index=False, header=False)


def _write_pca_embedding(
    expr_cells_by_genes: pd.DataFrame,
    output_path: Path,
    *,
    max_pca_components: int,
) -> None:
    from sklearn.decomposition import PCA

    matrix = expr_cells_by_genes.to_numpy(dtype=float)
    target_components = int(max_pca_components)
    n_components = min(target_components, matrix.shape[0], matrix.shape[1])
    if n_components < 1:
        raise ValueError("Cannot create DeepTalk PCA embeddings from an empty expression matrix")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    embedding = PCA(n_components=n_components).fit_transform(centered)
    if embedding.shape[1] < target_components:
        padding = np.zeros((embedding.shape[0], target_components - embedding.shape[1]), dtype=float)
        embedding = np.concatenate([embedding, padding], axis=1)
    pd.DataFrame(embedding, index=np.arange(matrix.shape[0])).to_csv(
        output_path,
        sep=" ",
        header=False,
    )


def _safe_relation_name(ligand: str, receptor: str) -> str:
    return f"{ligand}_{receptor}".replace("-", "_").replace(" ", "_")


def _check_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} table missing columns: {sorted(missing)}")
