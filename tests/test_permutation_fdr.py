"""Regression coverage for the public permutation-FDR outputs."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.spitialself.fdr import (
    FormalFDRPanelConfig,
    PermutationFDRConfig,
    permutation_fdr,
    run_formal_fdr_panel,
)


class PermutationFDRTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame({
            "candidate_id": [0, 1, 2],
            "lr_id": [1, 1, 2],
            "ligand": ["A", "A", "C"],
            "receptor": ["B", "B", "D"],
            "mechanism": ["short", "short", "long"],
            "sender_cell_type": ["T", "T", "T"],
            "receiver_cell_type": ["B", "B", "B"],
        })
        self.predictions = pd.DataFrame({"candidate_id": [0, 1, 2], "score": [0., 1., 100.]})

    def test_singleton_lr_stays_fixed_under_stratified_null(self):
        rows = permutation_fdr(
            self.candidates, self.predictions, null_model="spatial_permutation",
            config=PermutationFDRConfig(n_permutations=100, random_seed=17),
        ).set_index("candidate_id")
        self.assertEqual(rows.at[2, "empirical_p_value"], 1.)
        self.assertEqual(rows.at[2, "null_mean"], 100.)
        self.assertLessEqual(rows.loc[[0, 1], "null_q95"].max(), 1.)

    def test_panel_significance_columns_use_configured_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_formal_fdr_panel(
                self.candidates, self.predictions, output_dir=output,
                prediction_path="scores.csv",
                config=FormalFDRPanelConfig(
                    null_models=("score_permutation",), n_permutations=10, alpha=1.,
                ),
            )
            for group in ("lr_pair", "cell_type_pair"):
                table = pd.read_csv(output / f"scores__score_permutation__{group}_direct_fdr.csv")
                self.assertGreater(len(table), 0)
                self.assertTrue(table["group_significant"].all())
