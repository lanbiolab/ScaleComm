"""Run a minimal, data-free ScaleComm training and inference example."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalecomm import build_model


def main() -> None:
    torch.manual_seed(7)
    n_nodes, n_features, n_lr = 12, 8, 4

    x = torch.randn(n_nodes, n_features)
    source = torch.arange(n_nodes, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    edge_index = torch.stack([source, target])
    data = Data(x=x, edge_index=edge_index)

    src_idx = torch.arange(24, dtype=torch.long) % n_nodes
    dst_idx = (src_idx + 1 + torch.arange(24, dtype=torch.long) % 3) % n_nodes
    lr_ids = torch.arange(24, dtype=torch.long) % n_lr
    dists = torch.linalg.vector_norm(x[src_idx, :2] - x[dst_idx, :2], dim=1, keepdim=True)
    labels = ((src_idx + lr_ids) % 3 == 0).float()

    model = build_model(
        "spatialself_wavelet_scale",
        in_dim=n_features,
        h_dim=16,
        lr_dim=8,
        n_lr=n_lr,
        distance_scale=max(float(torch.quantile(dists, 0.95)), 1e-6),
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        scale_bank = model.encode_nodes(data)
        logits = model.score_edges(scale_bank, src_idx, dst_idx, lr_ids, dists)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        scale_bank = model.encode_nodes(data)
        logits = model.score_edges(scale_bank, src_idx, dst_idx, lr_ids, dists)
        scores = torch.sigmoid(logits)
        profiles = model.lr_scale_distribution(torch.arange(n_lr))

    with open("toy_event_scores.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sender_id", "receiver_id", "lr_id", "score"])
        writer.writerows(
            zip(
                src_idx.tolist(),
                dst_idx.tolist(),
                lr_ids.tolist(),
                scores.tolist(),
                strict=True,
            )
        )

    with open("toy_lr_scale_profiles.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lr_id", *[f"scale_{i}" for i in range(profiles.shape[1])]])
        for lr_id, profile in enumerate(profiles.tolist()):
            writer.writerow([lr_id, *profile])

    print("ScaleComm toy demo completed")
    print(f"node_scale_bank: {tuple(scale_bank.shape)}")
    print(f"event_scores: {tuple(scores.shape)}")
    print(f"lr_scale_profiles: {tuple(profiles.shape)}")
    print(f"scale_profile_row_sums: {[round(value, 6) for value in profiles.sum(1).tolist()]}")
    print(f"training_loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print("outputs: toy_event_scores.csv, toy_lr_scale_profiles.csv")


if __name__ == "__main__":
    main()
