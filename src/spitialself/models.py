"""Reusable neural network modules for LR-specific communication scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree


class SAGENodeEncoder(nn.Module):
    """Two-layer GraphSAGE encoder for tissue graph representation."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


class GATv2NodeEncoder(nn.Module):
    """Two-layer GATv2 encoder used by Visium self-supervised experiments."""

    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.norm1(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h


class GCNNodeEncoder(nn.Module):
    """Two-layer GCN encoder for architecture ablations."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.norm1(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h


class MLPNodeEncoder(nn.Module):
    """Graph-free node encoder for architecture ablations."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


def _make_undirected(edge_index):
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)


def _normalized_propagate(x, edge_index):
    """One symmetric-normalized sparse propagation step."""

    num_nodes = x.size(0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index

    deg = degree(dst, num_nodes=num_nodes, dtype=x.dtype).clamp_min(1.0)
    norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)

    out = x.new_zeros(x.shape)
    out.index_add_(0, dst, x[src] * norm.unsqueeze(-1))
    return out


class GraphWaveletNodeEncoder(nn.Module):
    """Multiscale graph-wavelet-style encoder using sparse diffusion.

    This avoids full graph eigendecomposition. It builds low-pass signals at
    several diffusion depths and uses adjacent differences as band-pass
    wavelet-like components.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        undirected=True,
    ):
        super().__init__()
        if not diffusion_steps:
            raise ValueError("diffusion_steps must contain at least one scale.")

        self.diffusion_steps = tuple(sorted(set(int(s) for s in diffusion_steps)))
        if self.diffusion_steps[0] < 1:
            raise ValueError("diffusion_steps must be positive integers.")
        self.undirected = bool(undirected)

        n_blocks = len(self.diffusion_steps) + 2  # raw + band-pass details + coarsest smooth
        self.project = nn.Sequential(
            nn.Linear(in_dim * n_blocks, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index):
        if self.undirected:
            edge_index = _make_undirected(edge_index)

        smooths = [x]
        h = x
        target_steps = set(self.diffusion_steps)
        for step in range(1, max(self.diffusion_steps) + 1):
            h = _normalized_propagate(h, edge_index)
            if step in target_steps:
                smooths.append(h)

        # Band-pass components: local detail at each scale.
        details = [smooths[i] - smooths[i + 1] for i in range(len(smooths) - 1)]
        multiscale = torch.cat([x, *details, smooths[-1]], dim=-1)
        return self.project(multiscale)


class GraphWaveletPyramidEncoder(nn.Module):
    """Multiscale graph-wavelet encoder that preserves per-scale features.

    Unlike :class:`GraphWaveletNodeEncoder`, this variant keeps each scale
    separate so LR-specific scale gating can be learned downstream.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        undirected=True,
    ):
        super().__init__()
        if not diffusion_steps:
            raise ValueError("diffusion_steps must contain at least one scale.")
        self.diffusion_steps = tuple(sorted(set(int(s) for s in diffusion_steps)))
        if self.diffusion_steps[0] < 1:
            raise ValueError("diffusion_steps must be positive integers.")
        self.undirected = bool(undirected)
        self.n_scales = len(self.diffusion_steps) + 2

        self.scale_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                    nn.LayerNorm(out_dim),
                )
                for _ in range(self.n_scales)
            ]
        )

    def _build_bank(self, x, edge_index):
        if self.undirected:
            edge_index = _make_undirected(edge_index)

        smooths = [x]
        h = x
        target_steps = set(self.diffusion_steps)
        for step in range(1, max(self.diffusion_steps) + 1):
            h = _normalized_propagate(h, edge_index)
            if step in target_steps:
                smooths.append(h)

        details = [smooths[i] - smooths[i + 1] for i in range(len(smooths) - 1)]
        return [x, *details, smooths[-1]]

    def forward(self, x, edge_index):
        bank = self._build_bank(x, edge_index)
        projected = [proj(scale) for proj, scale in zip(self.scale_projections, bank, strict=False)]
        return torch.stack(projected, dim=1)


class LRSpecificWaveletContextScorer(nn.Module):
    """LR-specific multiscale scorer with an explicit context gate."""

    def __init__(
        self,
        h_dim,
        lr_dim,
        n_scales,
        distance_scale,
        use_distance=True,
        edge_feature_dim=0,
        hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.use_distance = bool(use_distance)
        self.edge_feature_dim = int(edge_feature_dim)
        self.n_scales = int(n_scales)

        self.scale_gate = nn.Sequential(
            nn.Linear(lr_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_scales),
        )
        base_in_dim = h_dim * 2 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim
        context_in_dim = h_dim * 4 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(base_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(context_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, scale_bank, src_idx, dst_idx, dist, e_lr, edge_features=None):
        src_scales = scale_bank[src_idx]
        dst_scales = scale_bank[dst_idx]
        scale_logits = self.scale_gate(e_lr)
        scale_weights = torch.softmax(scale_logits, dim=-1)
        h_src = torch.sum(src_scales * scale_weights.unsqueeze(-1), dim=1)
        h_dst = torch.sum(dst_scales * scale_weights.unsqueeze(-1), dim=1)

        parts = [h_src, h_dst, e_lr]
        if self.use_distance:
            parts.append(dist / self.distance_scale)
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = h_src.new_zeros((h_src.size(0), self.edge_feature_dim))
            parts.append(edge_features)
        base_input = torch.cat(parts, dim=-1)

        context_parts = [
            h_src,
            h_dst,
            torch.abs(h_src - h_dst),
            h_src * h_dst,
            e_lr,
        ]
        if self.use_distance:
            context_parts.append(dist / self.distance_scale)
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = h_src.new_zeros((h_src.size(0), self.edge_feature_dim))
            context_parts.append(edge_features)
        context_input = torch.cat(context_parts, dim=-1)

        hidden = self.base_mlp(base_input)
        gate = self.context_gate(context_input)
        return self.out(hidden * gate).squeeze(-1)

    def scale_distribution(self, e_lr):
        return torch.softmax(self.scale_gate(e_lr), dim=-1)


class LRScaleMixtureWaveletScorer(nn.Module):
    """LR-specific scale-mixture scorer over graph-wavelet components.

    Each wavelet scale produces its own edge logit. The LR identity then
    mixes those per-scale logits, which makes the learned scale distribution
    directly responsible for the final communication score.
    """

    def __init__(
        self,
        h_dim,
        lr_dim,
        n_scales,
        distance_scale,
        use_distance=True,
        edge_feature_dim=0,
        hidden=96,
        dropout=0.2,
    ):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.use_distance = bool(use_distance)
        self.edge_feature_dim = int(edge_feature_dim)
        self.n_scales = int(n_scales)

        self.scale_gate = nn.Sequential(
            nn.Linear(lr_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_scales),
        )
        per_scale_in_dim = h_dim * 4 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim
        self.scale_mlp = nn.Sequential(
            nn.Linear(per_scale_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        bias_in_dim = h_dim * 2 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim
        self.bias_mlp = nn.Sequential(
            nn.Linear(bias_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, scale_bank, src_idx, dst_idx, dist, e_lr, edge_features=None):
        scale_weights = torch.softmax(self.scale_gate(e_lr), dim=-1)
        return self.forward_with_scale_weights(scale_bank, src_idx, dst_idx, dist, e_lr, scale_weights, edge_features)

    def forward_with_scale_weights(self, scale_bank, src_idx, dst_idx, dist, e_lr, scale_weights, edge_features=None):
        src_scales = scale_bank[src_idx]
        dst_scales = scale_bank[dst_idx]
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = scale_bank.new_zeros((src_idx.numel(), self.edge_feature_dim))
        else:
            edge_features = None

        per_scale_logits = []
        for scale_i in range(self.n_scales):
            h_src = src_scales[:, scale_i, :]
            h_dst = dst_scales[:, scale_i, :]
            parts = [
                h_src,
                h_dst,
                torch.abs(h_src - h_dst),
                h_src * h_dst,
                e_lr,
            ]
            if self.use_distance:
                parts.append(dist / self.distance_scale)
            if edge_features is not None:
                parts.append(edge_features)
            per_scale_logits.append(self.scale_mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        per_scale = torch.stack(per_scale_logits, dim=1)

        raw_src = src_scales[:, 0, :]
        raw_dst = dst_scales[:, 0, :]
        bias_parts = [raw_src, raw_dst, e_lr]
        if self.use_distance:
            bias_parts.append(dist / self.distance_scale)
        if edge_features is not None:
            bias_parts.append(edge_features)
        bias = 0.15 * self.bias_mlp(torch.cat(bias_parts, dim=-1)).squeeze(-1)
        return torch.sum(per_scale * scale_weights, dim=1) + bias

    def scale_distribution(self, e_lr):
        return torch.softmax(self.scale_gate(e_lr), dim=-1)


class LRRangeContextScorer(nn.Module):
    """Graph-free LR scorer with learnable distance-range and context gates.

    The scorer treats communication plausibility as two coupled pieces:

    1. a base interaction term from sender/receiver state and LR identity;
    2. a data-driven spatial compatibility term built from LR/context-specific
       weights over fixed distance basis functions.

    This keeps the mechanism clean: different LR pairs can prefer different
    spatial ranges, and the same LR can shift its preferred basis weights under
    different local cellular context.
    """

    def __init__(
        self,
        h_dim,
        lr_dim,
        distance_scale,
        use_distance=True,
        edge_feature_dim=0,
        hidden=128,
        dropout=0.2,
        n_range_basis=6,
        max_distance_multiplier=2.5,
    ):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.use_distance = bool(use_distance)
        self.edge_feature_dim = int(edge_feature_dim)
        self.n_range_basis = int(n_range_basis)

        base_in_dim = h_dim * 2 + lr_dim + self.edge_feature_dim
        context_in_dim = h_dim * 4 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim

        self.base_mlp = nn.Sequential(
            nn.Linear(base_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.base_out = nn.Linear(hidden, 1)

        self.context_gate = nn.Sequential(
            nn.Linear(context_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.range_gate = nn.Sequential(
            nn.Linear(context_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_range_basis + 1),
        )

        centers = torch.linspace(0.0, float(max_distance_multiplier), self.n_range_basis)
        if self.n_range_basis > 1:
            width = float(centers[1] - centers[0]) * 0.8
        else:
            width = float(max_distance_multiplier)
        self.register_buffer("range_basis_centers", centers)
        self.register_buffer("range_basis_widths", torch.full((self.n_range_basis,), max(width, 1e-3)))

    def forward(self, h_src, h_dst, dist, e_lr, edge_features=None):
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = h_src.new_zeros((h_src.size(0), self.edge_feature_dim))
        else:
            edge_features = None

        base_parts = [h_src, h_dst, e_lr]
        if edge_features is not None:
            base_parts.append(edge_features)
        base_input = torch.cat(base_parts, dim=-1)
        base_hidden = self.base_mlp(base_input)
        base_logit = self.base_out(base_hidden).squeeze(-1)

        context_parts = [
            h_src,
            h_dst,
            torch.abs(h_src - h_dst),
            h_src * h_dst,
            e_lr,
        ]
        if self.use_distance:
            context_parts.append(dist / self.distance_scale)
        if edge_features is not None:
            context_parts.append(edge_features)
        context_input = torch.cat(context_parts, dim=-1)

        context_gate = self.context_gate(context_input).squeeze(-1)
        if not self.use_distance:
            return base_logit * (0.5 + context_gate)

        range_params = self.range_gate(context_input)
        range_logits = range_params[:, : self.n_range_basis]
        range_offset = range_params[:, -1]
        range_weights = torch.softmax(range_logits, dim=-1)

        dist_norm = dist.squeeze(-1) / self.distance_scale
        basis = torch.exp(
            -0.5
            * torch.square(
                (dist_norm.unsqueeze(-1) - self.range_basis_centers.unsqueeze(0))
                / self.range_basis_widths.unsqueeze(0)
            )
        )
        range_compat = torch.sum(range_weights * basis, dim=-1).clamp_min(1e-6)
        range_logit = torch.log(range_compat) + range_offset
        return base_logit * (0.5 + context_gate) + range_logit

    def range_distribution(self, h_src, h_dst, e_lr, edge_features=None, dist=None):
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = h_src.new_zeros((h_src.size(0), self.edge_feature_dim))
        else:
            edge_features = None

        context_parts = [
            h_src,
            h_dst,
            torch.abs(h_src - h_dst),
            h_src * h_dst,
            e_lr,
        ]
        if self.use_distance:
            if dist is None:
                dist = h_src.new_zeros((h_src.size(0), 1))
            context_parts.append(dist / self.distance_scale)
        if edge_features is not None:
            context_parts.append(edge_features)
        context_input = torch.cat(context_parts, dim=-1)
        range_logits = self.range_gate(context_input)[:, : self.n_range_basis]
        return torch.softmax(range_logits, dim=-1)


class LREncoder(nn.Module):
    """Ligand-receptor identity encoder for ablations."""

    def __init__(self, n_lr, lr_dim, dropout=0.0, mode="learned"):
        super().__init__()
        mode = str(mode)
        if mode not in {"learned", "fixed_random", "one_hot"}:
            raise ValueError(f"Unknown LR encoding mode: {mode}")
        self.mode = mode
        self.lr_dim = int(lr_dim)
        self.n_lr = int(n_lr)
        self.dropout = nn.Dropout(dropout)
        if mode == "learned":
            self.emb = nn.Embedding(n_lr, lr_dim)
        elif mode == "fixed_random":
            basis = torch.randn(n_lr, lr_dim)
            basis = F.normalize(basis, dim=1)
            self.register_buffer("basis", basis)
        else:
            if lr_dim < n_lr:
                raise ValueError("one_hot LR encoding requires lr_dim >= n_lr.")
            basis = torch.zeros(n_lr, lr_dim)
            basis[:, :n_lr] = torch.eye(n_lr)
            self.register_buffer("basis", basis)

    def forward(self, lr_ids):
        if self.mode == "learned":
            return self.dropout(self.emb(lr_ids))
        return self.dropout(self.basis[lr_ids])


class EdgeScorer(nn.Module):
    """MLP scorer for a directed spatial edge and one LR pair."""

    def __init__(self, h_dim, lr_dim, distance_scale, use_distance=True, edge_feature_dim=0):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.use_distance = bool(use_distance)
        self.edge_feature_dim = int(edge_feature_dim)
        in_dim = h_dim * 2 + lr_dim + (1 if self.use_distance else 0) + self.edge_feature_dim
        hidden = 128
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_src, h_dst, dist, e_lr, edge_features=None):
        parts = [h_src, h_dst, e_lr]
        if self.use_distance:
            parts.append(dist / self.distance_scale)
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = h_src.new_zeros((h_src.size(0), self.edge_feature_dim))
            parts.append(edge_features)
        feat = torch.cat(parts, dim=-1)
        return self.mlp(feat).squeeze(-1)


class LRSpecificSAGEModel(nn.Module):
    """Benchmark model: GraphSAGE node encoder + LR embedding + edge MLP."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = SAGENodeEncoder(in_dim, h_dim, h_dim)
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.0, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = EdgeScorer(
            h_dim,
            lr_dim,
            distance_scale=200.0,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class LRModelVisium(nn.Module):
    """Visium model: GATv2 node encoder + LR embedding + edge MLP."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        heads=4,
        dropout=0.2,
        use_lr_embedding=True,
        use_distance=True,
        distance_scale=250.0,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = GATv2NodeEncoder(
            in_dim,
            h_dim,
            h_dim,
            heads=heads,
            dropout=dropout,
        )
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = EdgeScorer(
            h_dim,
            lr_dim,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfWaveletModel(nn.Module):
    """SpatialSelf variant with graph-wavelet multiscale node encoding."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        distance_scale=250.0,
        undirected=True,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = GraphWaveletNodeEncoder(
            in_dim=in_dim,
            hidden_dim=h_dim,
            out_dim=h_dim,
            diffusion_steps=diffusion_steps,
            dropout=dropout,
            undirected=undirected,
        )
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = EdgeScorer(
            h_dim,
            lr_dim,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfGCNModel(nn.Module):
    """SpatialSelf architecture ablation with a GCN node encoder."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        dropout=0.2,
        distance_scale=250.0,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = GCNNodeEncoder(in_dim, h_dim, h_dim, dropout=dropout)
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = EdgeScorer(
            h_dim,
            lr_dim,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfMLPModel(nn.Module):
    """SpatialSelf architecture ablation with no graph message passing."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        dropout=0.2,
        distance_scale=250.0,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = MLPNodeEncoder(in_dim, h_dim, h_dim, dropout=dropout)
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = EdgeScorer(
            h_dim,
            lr_dim,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfWaveletAdaptiveModel(nn.Module):
    """Wavelet SpatialSelf with LR-specific scale selection and context gating."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        distance_scale=250.0,
        undirected=True,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = GraphWaveletPyramidEncoder(
            in_dim=in_dim,
            hidden_dim=h_dim,
            out_dim=h_dim,
            diffusion_steps=diffusion_steps,
            dropout=dropout,
            undirected=undirected,
        )
        self.diffusion_steps = self.node_encoder.diffusion_steps
        self.n_scales = self.node_encoder.n_scales
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = LRSpecificWaveletContextScorer(
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_scales=self.node_encoder.n_scales,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
            hidden=128,
            dropout=dropout,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h, src_idx, dst_idx, dists, e_lr, edge_features)

    def lr_scale_distribution(self, lr_ids, device=None):
        if device is None:
            if self.lr_encoder is not None and hasattr(self.lr_encoder, "emb"):
                device = self.lr_encoder.emb.weight.device
            else:
                device = torch.device("cpu")
        e_lr = self._encode_lr(lr_ids, device)
        return self.edge_scorer.scale_distribution(e_lr)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfWaveletScaleModel(nn.Module):
    """Official scale-aware SpatialSelf baseline.

    The model preserves graph-wavelet features per scale and lets each LR pair
    mix per-scale edge logits. This makes LR-specific spatial scale preference
    an explicit, inspectable part of the scoring mechanism.
    """

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        diffusion_steps=(1, 2, 4, 8),
        dropout=0.2,
        distance_scale=250.0,
        undirected=True,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = GraphWaveletPyramidEncoder(
            in_dim=in_dim,
            hidden_dim=h_dim,
            out_dim=h_dim,
            diffusion_steps=diffusion_steps,
            dropout=dropout,
            undirected=undirected,
        )
        self.diffusion_steps = self.node_encoder.diffusion_steps
        self.n_scales = self.node_encoder.n_scales
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = LRScaleMixtureWaveletScorer(
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_scales=self.node_encoder.n_scales,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
            hidden=96,
            dropout=dropout,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h, src_idx, dst_idx, dists, e_lr, edge_features)

    def lr_scale_distribution(self, lr_ids, device=None):
        if device is None:
            if self.lr_encoder is not None and hasattr(self.lr_encoder, "emb"):
                device = self.lr_encoder.emb.weight.device
            else:
                device = torch.device("cpu")
        e_lr = self._encode_lr(lr_ids, device)
        return self.edge_scorer.scale_distribution(e_lr)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)


class SpatialSelfRangeContextModel(nn.Module):
    """Graph-free SpatialSelf variant with LR-specific range/context gating."""

    def __init__(
        self,
        in_dim,
        h_dim,
        lr_dim,
        n_lr,
        dropout=0.2,
        distance_scale=250.0,
        use_lr_embedding=True,
        use_distance=True,
        lr_encoding_mode="learned",
        edge_feature_dim=0,
        n_range_basis=6,
        max_distance_multiplier=2.5,
    ):
        super().__init__()
        self.lr_dim = int(lr_dim)
        self.use_lr_embedding = bool(use_lr_embedding)
        self.node_encoder = MLPNodeEncoder(in_dim, h_dim, h_dim, dropout=dropout)
        self.lr_encoder = (
            LREncoder(n_lr, lr_dim, dropout=0.1, mode=lr_encoding_mode)
            if self.use_lr_embedding
            else None
        )
        self.edge_scorer = LRRangeContextScorer(
            h_dim=h_dim,
            lr_dim=lr_dim,
            distance_scale=distance_scale,
            use_distance=use_distance,
            edge_feature_dim=edge_feature_dim,
            hidden=128,
            dropout=dropout,
            n_range_basis=n_range_basis,
            max_distance_multiplier=max_distance_multiplier,
        )

    def encode_nodes(self, data):
        return self.node_encoder(data.x, data.edge_index)

    def score_edges(self, h, src_idx, dst_idx, lr_ids, dists, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer(h_src, h_dst, dists, e_lr, edge_features)

    def range_distribution(self, h, src_idx, dst_idx, lr_ids, dists=None, edge_features=None):
        h_src = h[src_idx]
        h_dst = h[dst_idx]
        e_lr = self._encode_lr(lr_ids, h.device)
        return self.edge_scorer.range_distribution(h_src, h_dst, e_lr, edge_features=edge_features, dist=dists)

    def _encode_lr(self, lr_ids, device):
        if self.lr_encoder is None:
            return torch.zeros((lr_ids.numel(), self.lr_dim), dtype=torch.float32, device=device)
        return self.lr_encoder(lr_ids)
