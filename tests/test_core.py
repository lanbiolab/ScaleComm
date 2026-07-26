"""Small public smoke tests for the ScaleComm core API."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from scalecomm import build_model
from src.spitialself.evaluation import _aupr


class ScaleCommCoreTests(unittest.TestCase):
    def test_official_model_forward_backward_and_scale_profiles(self) -> None:
        torch.manual_seed(3)
        data = Data(
            x=torch.randn(6, 5),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        )
        model = build_model(
            "spatialself_wavelet_scale",
            in_dim=5,
            h_dim=8,
            lr_dim=4,
            n_lr=3,
            distance_scale=2.0,
            dropout=0.0,
        )
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 4])
        lr_ids = torch.tensor([0, 1, 2, 0])
        dists = torch.ones(4, 1)
        bank = model.encode_nodes(data)
        logits = model.score_edges(bank, src, dst, lr_ids, dists)
        logits.square().mean().backward()

        self.assertEqual(tuple(bank.shape), (6, 6, 8))
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertTrue(torch.isfinite(logits).all())
        profiles = model.lr_scale_distribution(torch.arange(3))
        self.assertTrue(torch.allclose(profiles.sum(1), torch.ones(3), atol=1e-6))

    def test_aupr_is_stable_for_tied_scores(self) -> None:
        truth = np.array([1, 0, 1, 0])
        score = np.full(4, 0.5)
        self.assertAlmostEqual(_aupr(truth, score), 0.5)
        self.assertAlmostEqual(_aupr(truth[::-1], score), 0.5)


if __name__ == "__main__":
    unittest.main()
