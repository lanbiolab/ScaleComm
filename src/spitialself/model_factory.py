"""Factory helpers for constructing SpatialSelf model variants."""

from .models import (
    LRModelVisium,
    LRSpecificSAGEModel,
    SpatialSelfGCNModel,
    SpatialSelfMLPModel,
    SpatialSelfRangeContextModel,
    SpatialSelfWaveletScaleModel,
    SpatialSelfWaveletAdaptiveModel,
    SpatialSelfWaveletModel,
)


def build_model(
    model_name,
    in_dim,
    h_dim,
    lr_dim,
    n_lr,
    **kwargs,
):
    """Build a model variant by stable config name."""

    if model_name in {"spatialself_gat", "full_spatialself", "visium_gat"}:
        return LRModelVisium(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            heads=kwargs.get("heads", 4),
            dropout=kwargs.get("dropout", 0.2),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            distance_scale=kwargs.get("distance_scale", 250.0),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_wavelet", "wavelet"}:
        return SpatialSelfWaveletModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            diffusion_steps=kwargs.get("diffusion_steps", (1, 2, 4, 8)),
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            undirected=kwargs.get("undirected", True),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_wavelet_adaptive", "wavelet_adaptive"}:
        return SpatialSelfWaveletAdaptiveModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            diffusion_steps=kwargs.get("diffusion_steps", (1, 2, 4, 8)),
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            undirected=kwargs.get("undirected", True),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_wavelet_scale", "wavelet_scale", "spatialself_scale"}:
        return SpatialSelfWaveletScaleModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            diffusion_steps=kwargs.get("diffusion_steps", (1, 2, 4, 8)),
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            undirected=kwargs.get("undirected", True),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_gcn", "gcn"}:
        return SpatialSelfGCNModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_mlp", "mlp"}:
        return SpatialSelfMLPModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    if model_name in {"spatialself_range_context", "range_context", "range_gate"}:
        return SpatialSelfRangeContextModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            dropout=kwargs.get("dropout", 0.2),
            distance_scale=kwargs.get("distance_scale", 250.0),
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
            n_range_basis=kwargs.get("n_range_basis", 6),
            max_distance_multiplier=kwargs.get("max_distance_multiplier", 2.5),
        )

    if model_name in {"spatialself_sage", "sage"}:
        return LRSpecificSAGEModel(
            in_dim=in_dim,
            h_dim=h_dim,
            lr_dim=lr_dim,
            n_lr=n_lr,
            use_lr_embedding=kwargs.get("use_lr_embedding", True),
            use_distance=kwargs.get("use_distance", True),
            lr_encoding_mode=kwargs.get("lr_encoding_mode", "learned"),
            edge_feature_dim=kwargs.get("edge_feature_dim", 0),
        )

    raise ValueError(f"Unknown model_name: {model_name}")
