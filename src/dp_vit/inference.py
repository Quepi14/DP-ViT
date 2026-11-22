"""Single-image inference utilities for DP-ViT."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ICSCalibration
from .ics import (
    DEFAULT_TIERS,
    compute_patch_size_map_for_image,
    compute_tiered_patch_size_map_for_image,
)
from .model import DPViT
from .patching import adaptive_collate_fn


def _validate_class_names(
    class_names: Sequence[str] | None,
    num_classes: int,
) -> None:
    """Validate optional class names."""

    if class_names is None:
        return

    if isinstance(class_names, (str, bytes)):
        raise TypeError("class_names must be a sequence of strings.")

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match model.num_classes."
        )

    if not all(
        isinstance(name, str) and name.strip()
        for name in class_names
    ):
        raise ValueError(
            "Every class name must be a non-empty string."
        )


def _validate_image_tensor(image: Tensor, model: DPViT) -> None:
    """Validate an RGB image before ICS processing."""

    if not isinstance(image, Tensor):
        raise TypeError(
            "image_before_normalization must be a torch.Tensor."
        )

    if image.ndim != 3:
        raise ValueError(
            "The input image must have shape (channels, height, width)."
        )

    channels, height, width = image.shape

    if channels != 3:
        raise ValueError(
            "DP-ViT currently expects an RGB image with 3 channels."
        )

    expected_size = model.config.image_size

    if (height, width) != (expected_size, expected_size):
        raise ValueError(
            f"Expected an image of size {expected_size}x{expected_size}, "
            f"got {height}x{width}."
        )

    if not image.is_floating_point():
        raise TypeError(
            "The input image must use a floating-point dtype."
        )

    if not torch.isfinite(image).all():
        raise ValueError(
            "The input image must contain only finite values."
        )

    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError(
            "ICS input must be in [0, 1]. Apply channel normalization "
            "after ICS patch-map generation."
        )


def predict_tensor(
    model: DPViT,
    image_before_normalization: Tensor,
    ics_calibration: ICSCalibration,
    class_names: Sequence[str] | None = None,
    normalize_fn: Callable[[Tensor], Tensor] | None = None,
    mode: str = "threshold",
    tiers: Sequence[tuple[int, int]] = DEFAULT_TIERS,
) -> dict[str, Any]:
    """
    Predict the class of one preprocessed RGB image.

    The input must be an RGB tensor in [0, 1], before channel
    normalization. Resizing and other user-selected preprocessing
    must be applied beforehand.

    Args:
        model: Trained DP-ViT model.
        image_before_normalization: RGB tensor in [0, 1] of shape (3, H, W).
        ics_calibration: Fitted ICSCalibration instance.
        class_names: Optional sequence of class labels.
        normalize_fn: Optional channel normalization applied after ICS mapping.
        mode: Patch assignment mode, either "threshold" or "tiered".
        tiers: Tier definitions used when mode="tiered".
    """

    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT.")

    if not isinstance(ics_calibration, ICSCalibration):
        raise TypeError(
            "ics_calibration must be an ICSCalibration instance."
        )

    if mode not in ("threshold", "tiered"):
        raise ValueError("mode must be either 'threshold' or 'tiered'.")

    _validate_image_tensor(
        image_before_normalization,
        model,
    )

    _validate_class_names(
        class_names,
        model.config.num_classes,
    )

    # ICS is calculated from the original RGB image in [0, 1].
    image = image_before_normalization.detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    image_np = image.permute(1, 2, 0).contiguous().numpy()

    tier_index: int | None = None

    if mode == "threshold":
        patch_size_map = compute_patch_size_map_for_image(
            image=image_np,
            calibration=ics_calibration,
            region_size=model.config.region_size,
            patch_sizes=model.config.patch_sizes,
        )
    else:
        if ics_calibration.detail_threshold is None:
            raise ValueError(
                "ics_calibration.detail_threshold must be set when using "
                "mode='tiered'."
            )

        patch_size_map, tier_index = compute_tiered_patch_size_map_for_image(
            image=image_np,
            calibration=ics_calibration,
            detail_threshold=ics_calibration.detail_threshold,
            tiers=tiers,
            region_size=model.config.region_size,
            patch_sizes=model.config.patch_sizes,
        )

    # Apply channel normalization only after generating the patch map.
    if normalize_fn is None:
        model_image = image
    else:
        model_image = normalize_fn(image)

    if not isinstance(model_image, Tensor):
        raise TypeError(
            "normalize_fn must return a torch.Tensor."
        )

    if model_image.shape != image.shape:
        raise ValueError(
            "normalize_fn must preserve the input image shape."
        )

    if not model_image.is_floating_point():
        raise TypeError(
            "normalize_fn must return a floating-point tensor."
        )

    if not torch.isfinite(model_image).all():
        raise ValueError(
            "Normalized image must contain only finite values."
        )

    if model_image.device.type != "cpu":
        raise ValueError(
            "normalize_fn must return a CPU tensor."
        )

    batch = adaptive_collate_fn(
        [(model_image, patch_size_map, 0)],
        region_size=model.config.region_size,
        patch_sizes=model.config.patch_sizes,
    )

    device = next(model.parameters()).device

    batch = {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }

    # Preserve the training/evaluation state of every submodule.
    module_states = [
        (module, module.training)
        for module in model.modules()
    ]

    model.eval()

    try:
        with torch.inference_mode():
            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            predicted_index = int(
                probabilities.argmax().item()
            )

    finally:
        for module, training_state in module_states:
            module.training = training_state

    region_size = model.config.region_size

    region_counts = {
        int(size): int(np.count_nonzero(patch_size_map == size))
        for size in model.config.patch_sizes
    }

    token_counts = {
        int(size): region_counts[int(size)] * (region_size // int(size)) ** 2
        for size in model.config.patch_sizes
    }

    predicted_name = (
        class_names[predicted_index]
        if class_names is not None
        else None
    )

    return {
        "class_index": predicted_index,
        "class_name": predicted_name,
        "confidence": float(
            probabilities[predicted_index].cpu().item()
        ),
        "probabilities": probabilities.cpu().tolist(),
        "patch_size_map": patch_size_map.tolist(),
        "region_counts": region_counts,
        "token_counts": token_counts,
        "tier_index": tier_index,
    }