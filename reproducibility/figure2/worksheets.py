"""Reconstruct the four Figure 2 Source Data worksheets for scored methods."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import archived_scores


# Order and labels used to enumerate unordered method pairs in F2_overlap.
METHODS = {
    "spatialself_range_context": "ScaleComm",
    "commot": "COMMOT",
    "spaotsc": "SpaOTsc",
    "cellphonedb": "CellPhoneDB",
    "cellchat": "CellChat",
    "niches": "NICHES",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
}
RANGES = {
    "juxtacrine_contact": "contact",
    "short_range_paracrine": "short",
    "medium_range_paracrine": "medium",
    "long_range_chemokine_niche": "long_niche",
}


def historical_aupr(truth, normalized):
    """Trapezoidal PR area from the original per-event ranking, including ties."""
    ranked = truth[np.argsort(-normalized, kind="quicksort")]
    tp = np.cumsum(ranked)
    recall = tp / tp[-1]
    precision = tp / np.arange(1, len(ranked) + 1)
    return np.sum(np.diff(np.r_[0.0, recall]) * (np.r_[1.0, precision[:-1]] + precision) / 2)


def summarize_calls(truth, selected, wrong):
    tp = np.count_nonzero(truth & selected)
    fp = np.count_nonzero(~truth & selected)
    calls = int(selected.sum())
    positives = int(truth.sum())
    negatives = len(truth) - positives
    wrong_calls = np.count_nonzero(selected & wrong)
    return {
        "f1": 2 * tp / (positives + calls) if tp else np.nan,
        "precision": tp / calls if calls else np.nan,
        "recall": tp / positives if positives else np.nan,
        "specificity": (negatives - fp) / negatives if negatives else np.nan,
        "empirical_fdp": fp / calls if calls else np.nan,
        "wrong_range_rate": wrong_calls / calls if calls else np.nan,
        "tp": tp, "fp": fp, "wrong_range_fp_count": wrong_calls,
        "detected_candidate_count": calls,
    }


def reconstruct(data):
    settings_rows, modes, overlaps = [], [], []
    score_dir = Path(__file__).resolve().parent / "reference_scores"
    manifest = data / "settings.json" if data else score_dir / "manifest.json"
    for setting, entry in json.loads(manifest.read_text()).items():
        path = (data / entry / "candidate_interactions.csv" if data
                else score_dir / entry["evaluation_table"])
        table = pd.read_csv(path, usecols=["candidate_id", "lr_id", "mechanism",
                                           "ground_truth", "wrong_range_false_positive"])
        tissue, level = setting.split("_", 1)
        noise = level.removesuffix("_noise")
        dataset = tissue + "_scdesign3_200lr" + ("_" + level if noise != "clean" else "")
        labels = {"tissue": "lymph_node" if tissue == "lymph" else tissue, "noise": noise}
        ids = table.candidate_id.to_numpy(dtype=str)
        truth = table.ground_truth.to_numpy(dtype=bool)
        wrong = table.wrong_range_false_positive.to_numpy(dtype=bool)
        detected = {}
        for method, scores in archived_scores(path, setting, len(table)):
            label = METHODS[method]
            span = np.ptp(scores)
            normalized = (scores - scores.min()) / span if span else np.zeros_like(scores)
            order = np.lexsort((ids, -normalized))
            selected = np.zeros(len(table), dtype=bool)
            selected[order[:int(np.ceil((1 - 0.95) * len(table)))]] = True
            detected[method] = set(table.loc[selected, "lr_id"])
            row = {"dataset_id": dataset, **labels, "method_label": label,
                   "aupr": historical_aupr(truth, normalized),
                   "auroc": roc_auc_score(truth, normalized)}
            settings_rows.append({**row, **summarize_calls(truth, selected, wrong)})
            for mechanism, range_name in RANGES.items():
                mask = table.mechanism.eq(mechanism).to_numpy()
                metrics = summarize_calls(truth[mask], selected[mask], wrong[mask])
                modes.append({**labels, "method_label": label, "mechanism": mechanism,
                              "range_class": range_name, "f1": metrics["f1"],
                              "wrong_range_rate": metrics["wrong_range_rate"]})
        for a, b in itertools.combinations(METHODS, 2):
            left, right = detected[a], detected[b]
            overlaps.append({**labels, "overlap_level": "lr_pair",
                             "method_a": "SpatialSelf" if a == "spatialself_range_context" else METHODS[a],
                             "method_b": METHODS[b],
                             "jaccard": len(left & right) / len(left | right)})
    settings = pd.DataFrame(settings_rows)
    measures = ["aupr", "precision", "recall", "f1", "empirical_fdp", "wrong_range_rate"]
    summary = settings.groupby("method_label")[measures].mean().add_suffix("_mean").reset_index()
    return {"F2_settings": settings, "F2_summary": summary,
            "F2_modes": pd.DataFrame(modes), "F2_overlap": pd.DataFrame(overlaps)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Dataset generated in archived mode (default: packaged evaluation tables)")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the four CSVs")
    args = parser.parse_args()
    tables = reconstruct(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    for sheet, frame in tables.items():
        frame.to_csv(args.output / f"{sheet}.csv", index=False)
        print(f"{sheet}: {len(frame)} rows")
