"""Public API for the DP-ViT library."""

from .batching import (
    ApproximateSignatureBatchSampler,
    ExactSignatureBatchSampler,
    SignatureBatchSampler,
    TolerantBucketBatchSampler,
)

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

from .config import (
    DPViTConfig,
    ICSCalibration,
    ICSConfig,
    PruningConfig,
)

from .ics import (
    DEFAULT_TIERS,
    compute_budget_patch_size_map_for_image,
    compute_gradient_magnitude,
    compute_ics,
    compute_local_variance,
    compute_patch_size_map_for_image,
    compute_tiered_patch_size_map_for_image,
    count_patch_tokens,
    determine_budget_patch_size_map,
    determine_patch_size_map,
    determine_tiered_patch_size_map,
    fit_detail_threshold,
    fit_ics_calibration,
    tier_token_count,
)

from .inference import predict_tensor

from .model import DPViT

from .patching import (
    AdaptivePatchEmbedding,
    adaptive_collate_fn,
    fast_adaptive_collate_fn,
    get_padded_sequence_length,
    get_patch_signature,
)

from .pruning import prune_mlp_by_magnitude


__all__ = [
    "AdaptivePatchEmbedding",
    "ApproximateSignatureBatchSampler",
    "DEFAULT_TIERS",
    "DPViT",
    "DPViTConfig",
    "ExactSignatureBatchSampler",
    "ICSCalibration",
    "ICSConfig",
    "PruningConfig",
    "SignatureBatchSampler",
    "TolerantBucketBatchSampler",
    "adaptive_collate_fn",
    "compute_budget_patch_size_map_for_image",
    "compute_gradient_magnitude",
    "compute_ics",
    "compute_local_variance",
    "compute_patch_size_map_for_image",
    "compute_tiered_patch_size_map_for_image",
    "count_patch_tokens",
    "determine_budget_patch_size_map",
    "determine_patch_size_map",
    "determine_tiered_patch_size_map",
    "fast_adaptive_collate_fn",
    "fit_detail_threshold",
    "fit_ics_calibration",
    "get_padded_sequence_length",
    "get_patch_signature",
    "load_checkpoint",
    "predict_tensor",
    "prune_mlp_by_magnitude",
    "save_checkpoint",
    "tier_token_count",
]