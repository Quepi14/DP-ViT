"""Single-image inference utilities for DP-ViT."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch 
from torch import Tensor

from .config import ICSCalibration
from .ics import compute_patch_size_map_for_image
from .model import DPViT
from .patching import adaptive_collate_fn

def predict_tensor(
    model: DPViT,
    image_before_normalization: Tensor,
    ics_calibration: ICSCalibration,
    class_names: Sequence[str] | None = None,
    normalize_fn : Callable[[Tensor], Tensor] | None = None,
) -> dict[str, Any]:
    """
    Predict the class of one preprocessed image tensor.

    The input must be an RGB tensor in [0, 1], before channel normalization.
    Resizing and other user-selected preprocessing must be applied beforehand.
    """
    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT.")

    if not isinstance(image_before_normalization, Tensor):
        raise TypeError("image_before_normalization must be a torch.Tensor.")

    if image_before_normalization.ndim != 3:
        raise ValueError("The input image must have shape (channels, height, width).")

    channels, height, width = image_before_normalization.shape

    if channels != 3:
        raise ValueError("DP-ViT currently expects an RGB image with 3 channels.")

    if (height, width) != (model.config.image_size, model.config.image_size):
        raise ValueError(
            f"Expected an image of size {model.config.image_size}x"
            f"{model.config.image_size}, got {height}x{width}."
        )

    if not torch.isfinite(image_before_normalization).all():
        raise ValueError("The input image must contain only finite values.")

    if (
        image_before_normalization.min() <0
        or image_before_normalization.max() > 1
    ):
        raise ValueError(
            "ICS input must be in [0, 1]. Apply channel normalization"
            "after ICS patch-map generation."
        )

    if not isinstance(ics_calibration, ICSCalibration):
        raise TypeError("ics_calibration must be an ICSCalibration instance.")

    if class_names is not None and len(class_names) != model.config.num_classes:
        raise ValueError("class_names length must match model.num_classes.")

    image = image_before_normalization.detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    image_np = image.permute(1, 2, 0).numpy()

    patch_size_map = compute_patch_size_map_for_image(
        image=image_np,
        calibration=ics_calibration,
        region_size=model.config.region_size,
        patch_sizes=model.config.patch_sizes,
    )

    if normalize_fn is None:
        model_image = image
    else: 
        model_image = normalize_fn(image)

    if not isinstance(model_image, Tensor):
        raise TypeError("normalize_fn must return a torch.Tensor.")

    if model_image.shape != image.shape:
        raise ValueError("normalize_fn must preserve the input image shape.")

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

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_index = int(probabilities.argmax())
    finally:
        model.train(was_training)

    patch_counts = {
        int(size): int(np.count_nonzero(patch_size_map == size))
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
        "confidence": float(probabilities[predicted_index].cpu()),
        "probabilities": probabilities.cpu().tolist(),
        "patch_size_map": patch_size_map.tolist(),
        "patch_counts": patch_counts,
    }