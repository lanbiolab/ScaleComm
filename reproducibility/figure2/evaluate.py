"""Evaluate event scores against simulator truth after fitting, at the locked 5% budget."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
MODES = {
    "juxtacrine_contact": "Contact",
    "short_range_paracrine": "Short-range",
    "medium_range_paracrine": "Medium-range",
    "long_range_chemokine_niche": "Long-niche",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_scores(path, setting, count):
    records = json.loads((HERE / "reference_scores" / "manifest.json").read_text())
    ref = records[setting]
    archive = HERE / "reference_scores" / f"{setting}.npz"
    bound_hash = ref["evaluation_table_sha256"] if path.name == ref["evaluation_table"] else ref["candidate_table_sha256"]
    if count != ref["count"] or sha(path) != bound_hash:
        raise ValueError(f"{setting}: archived score rows require the matching archived candidate table")
    if sha(archive) != ref["score_archive_sha256"]:
        raise ValueError(f"{setting}: archived score checksum mismatch")
    with np.load(archive, allow_pickle=False) as values:
        for method in values.files:
            yield method, values[method]


def user_scores(folder, ids):
    files = sorted(folder.glob("*predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No *predictions.csv in {folder}")
    seen = set()
    for path in files:
        table = pd.read_csv(path, usecols=["candidate_id", "score", "method"])
        names = table.method.drop_duplicates()
        if len(names) != 1 or pd.isna(names.iloc[0]) or names.iloc[0] in seen:
            raise ValueError(f"Expected one unique method name per file: {path}")
        seen.add(names.iloc[0])
        series = table.set_index("candidate_id").score
        if not series.index.is_unique or set(ids) != set(series.index):
            raise ValueError(f"Candidate identifiers do not match exactly: {path}")
        yield names.iloc[0], series.loc[ids].to_numpy(dtype=float)


def divide(n, d):
    return n / d if d else np.nan


def historical_aupr(truth, normalized):
    """Trapezoidal PR area from the original per-event ranking, including ties."""
    ranked = truth[np.argsort(-normalized, kind="quicksort")]
    tp = np.cumsum(ranked)
    recall = tp / tp[-1]
    precision = tp / np.arange(1, len(ranked) + 1)
    return np.sum(np.diff(np.r_[0.0, recall]) * (np.r_[1.0, precision[:-1]] + precision) / 2)


def metrics(table, scores, selected):
    truth = table.ground_truth.to_numpy(dtype=bool)
    wrong = table.wrong_range_false_positive.to_numpy(dtype=bool)
    tp = (truth & selected).sum()
    fp = (~truth & selected).sum()
    weak_positive = truth & table.weak_signal.to_numpy(dtype=bool)
    decoy = table.false_positive_high_expression.to_numpy(dtype=bool)
    top100 = np.argsort(-scores)[:100]
    return dict(
        auroc=roc_auc_score(truth, scores) if truth.any() and not truth.all() else np.nan,
        precision_at_100=truth[top100].mean() if len(top100) else np.nan,
        weak_signal_recall=divide((weak_positive & selected).sum(), weak_positive.sum()),
        high_expression_fp_rate=divide((decoy & selected).sum(), decoy.sum()),
        wrong_range_rate=divide((wrong & selected).sum(), selected.sum()),
        aupr=historical_aupr(truth, scores) if truth.any() else np.nan,
        average_precision=average_precision_score(truth, scores) if truth.any() else np.nan,
        selected=int(selected.sum()), f1=divide(2*tp, truth.sum()+selected.sum()) if tp else np.nan,
        precision=divide(tp, selected.sum()), recall=divide(tp, truth.sum()),
        fdp=divide(fp, selected.sum()), specificity=divide((~truth & ~selected).sum(), (~truth).sum()),
        wrong_range_fp_rate=divide((wrong & selected).sum(), wrong.sum()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, help="Generated dataset directory; default: released simulator evaluation fields")
    p.add_argument("--predictions", type=Path, help="Directory of per-setting prediction CSVs; default: archived scores")
    p.add_argument("--setting", action="append", help="Restrict to this setting; repeatable")
    p.add_argument("--output", type=Path, required=True, help="New directory for evaluated tables")
    args = p.parse_args()
    settings = (json.loads((args.data / "settings.json").read_text()) if args.data else
                json.loads((HERE / "reference_scores" / "manifest.json").read_text()))
    overall, modes, composition = [], [], []
    for setting in args.setting or settings:
        path = (args.data / settings[setting] / "candidate_interactions.csv" if args.data else
                HERE / "reference_scores" / settings[setting]["evaluation_table"])
        table = pd.read_csv(path, usecols=["candidate_id", "mechanism", "ground_truth", "wrong_range_false_positive",
                                           "false_positive_high_expression", "weak_signal"])
        ids = table.candidate_id
        if not ids.is_unique:
            raise ValueError(f"Duplicate candidate identifiers in {path}")
        if args.predictions and args.data:
            config_file = args.predictions / setting / "fit_config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
                if config["candidate_table_sha256"] != sha(path):
                    raise ValueError(f"{setting}: predictions were trained on a different candidate table")
        inputs = (user_scores(args.predictions / setting, ids) if args.predictions
                  else archived_scores(path, setting, len(ids)))
        for method, scores in inputs:
            if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                raise ValueError(f"{setting} {method}: invalid scores")
            span = np.ptp(scores)
            normalized = (scores - scores.min()) / span if span else np.zeros_like(scores)
            order = np.lexsort((ids.to_numpy(dtype=str), -scores))
            selected = np.zeros(len(table), dtype=bool)
            selected[order[:int(np.ceil((1.0 - 0.95) * len(table)))]] = True
            labels = {"setting": setting, "method": method}
            overall.append(dict(**labels, **metrics(table, normalized, selected)))
            for mode, title in MODES.items():
                mask = table.mechanism.eq(mode).to_numpy()
                modes.append(dict(**labels, mode=title, **metrics(table[mask], normalized[mask], selected[mask])))
            for group, count in table.loc[selected, "mechanism"].value_counts().items():
                composition.append(dict(**labels, mechanism=group, selected=int(count), fraction=count/selected.sum()))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, records in [("event_metrics", overall), ("mode_metrics", modes), ("composition", composition)]:
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    print(pd.DataFrame(overall).groupby("method")[["aupr", "average_precision", "f1", "precision", "recall", "fdp"]].mean().to_string())


if __name__ == "__main__":
    main()
