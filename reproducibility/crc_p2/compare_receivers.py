"""Compare newly trained P2CRC receiver rankings with the released W12 table."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prepared", "predictions", "reference", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--check-reference", action="store_true",
                        help="Check the paper-linked CRC replay against its reference.")
    args = parser.parse_args()

    candidates = pd.read_csv(args.prepared / "candidate_interactions.csv",
                             usecols=["candidate_id", "lr_id", "receiver_id"])
    scores = pd.read_csv(args.predictions, usecols=["candidate_id", "score"])
    events = candidates.merge(scores, on="candidate_id", how="outer",
                              validate="one_to_one", indicator=True)
    if not events["_merge"].eq("both").all() or not np.isfinite(events["score"]).all():
        raise ValueError("Predictions must contain one finite score per candidate.")
    panel = pd.read_csv(args.prepared / "lr_pairs.csv")
    focal_id = panel.loc[panel["control_name"].eq("TGFB1-TGFBR2"), "lr_id"].item()
    result = (events.loc[events["lr_id"].eq(focal_id)]
              .groupby("receiver_id")["score"].mean().rename("retrained_score"))
    reference = pd.read_csv(args.reference)
    ref = reference.loc[reference["dataset"].eq("CRC Visium HD")
                        & reference["section"].eq("CRC")
                        & reference["lr_pair"].eq("TGFB1-TGFBR2"),
                        ["receiver_id", "score"]].set_index("receiver_id", verify_integrity=True)
    if ref.empty or not result.index.sort_values().equals(ref.index.sort_values()):
        raise ValueError("Retrained and reference receiver IDs differ.")
    paired = result.to_frame().join(ref.rename(columns={"score": "reference_score"}))
    if not np.isfinite(paired.to_numpy()).all() or (paired.nunique() < 2).any():
        raise ValueError("Receiver profiles must have finite, nonconstant scores.")
    rho = float(spearmanr(paired["retrained_score"], paired["reference_score"]).statistic)
    report = {"focal_lr": "TGFB1-TGFBR2", "n_candidates": len(events),
              "n_focal_events": int(events["lr_id"].eq(focal_id).sum()),
              "n_receivers": len(paired), "spearman_rank_correlation": rho,
              "max_absolute_score_difference":
                  float((paired["retrained_score"] - paired["reference_score"]).abs().max())}
    # Fix the list length and break tied scores by barcode in both rankings.
    n_top = int(np.ceil(0.05 * len(paired)))
    top_retrained = set(paired.sort_values(
        ["retrained_score", "receiver_id"], ascending=[False, True]).head(n_top).index)
    top_reference = set(paired.sort_values(
        ["reference_score", "receiver_id"], ascending=[False, True]).head(n_top).index)
    report["top_5pct_receivers_per_list"] = n_top
    report["top_5pct_shared_receivers"] = len(top_retrained & top_reference)
    report["top_5pct_receiver_jaccard"] = len(top_retrained & top_reference) / len(top_retrained | top_reference)
    args.output.mkdir(parents=True, exist_ok=True)
    paired["retrained_rank"] = paired["retrained_score"].rank(ascending=False)
    paired["reference_rank"] = paired["reference_score"].rank(ascending=False)
    paired.sort_values(["retrained_rank", "receiver_id"]).to_csv(args.output / "receiver_comparison.csv")
    summary = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "receiver_comparison.json").write_text(summary + "\n", encoding="utf8")
    print(summary)
    if args.check_reference:
        expected_counts = (91345, 12864, 4280)
        observed_counts = (report["n_candidates"], report["n_focal_events"], report["n_receivers"])
        if observed_counts != expected_counts or rho < 0.99:
            raise SystemExit("CRC replay differs from its reference. See receiver_comparison.json.")


if __name__ == "__main__":
    main()
