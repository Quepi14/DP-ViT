# DP-ViT

DP-ViT is a PyTorch library for image classification with ICS-guided adaptive
patch sizes. It also provides optional structured magnitude-based pruning for
the Transformer MLP layers.

The library provides:

- Image Complexity Score (ICS) calculation and calibration
- Adaptive patch-size mapping and patch embedding
- A Vision Transformer classification model
- Optional structured MLP pruning
- Checkpoint save and load utilities
- Single-image inference

Dataset loading, image cleaning, augmentation, normalization, and training
loops are handled by the user.

> **Status:** DP-ViT is under development. Its API may change before a stable
> release.

## How it works

DP-ViT divides an image into regions and assigns a patch size to each region
using its ICS value. Regions with higher complexity receive smaller patches;
regions with lower complexity receive larger patches.

The patches are projected into tokens and passed to a Transformer encoder.
After training, users can optionally apply structured magnitude-based pruning
to the MLP layers. Pruning does not automatically fine-tune the model.

## Optional ICS token budget

By default, ICS calibration uses the global 30th and 70th percentiles to
assign adaptive patch sizes. You can optionally search for global thresholds
that keep the **average number of valid patch tokens per calibration image**
at or below a target.

This is an average over the calibration images, not a strict limit for each
image. It also does not guarantee that the padded sequence processed by the
encoder will be shorter than a baseline Vision Transformer.

```python
from dp_vit import ICSConfig, fit_ics_calibration

target_tokens = (224 // 16) ** 2  # 196 + 1 class token or total 197 patch tokens for a 224x224 ViT with 16x16 patches

ics_config = ICSConfig(
    target_mean_tokens_per_image=target_tokens,
    search_step=5.0,
)

ics_calibration = fit_ics_calibration(
    calibration_images=calibration_images,
    region_size=32,
    config=ics_config,
    patch_sizes=(8, 16, 32),
)

print("Selected percentiles:",
      ics_calibration.low_percentile,
      ics_calibration.high_percentile)
print("Estimated valid tokens per image:",
      ics_calibration.estimated_tokens_per_image)
```

## Installation

Clone the repository, then run this command from its root directory:

```bash
pip install -e .
```

## Basic workflow

1. Prepare RGB images with your own data pipeline.
2. Fit ICS calibration values using representative images.
3. Calculate a patch-size map for each image before channel normalization.
4. Normalize the image for the model, if needed.
5. Use `adaptive_collate_fn` to create batches.
6. Train `DPViT` with your own training loop.
7. Optionally prune the trained model.
8. Save the checkpoint and use it for inference.

## Example

This example shows how to create the model, prepare a batch, run a forward
pass, and optionally prune and save the model. It does not include a training
loop or dataset implementation.

```python
from functools import partial

import torch
from torch.utils.data import DataLoader

from dp_vit import (
    DPViT,
    DPViTConfig,
    ICSConfig,
    PruningConfig,
    adaptive_collate_fn,
    compute_patch_size_map_for_image,
    fit_ics_calibration,
    prune_mlp_by_magnitude,
    save_checkpoint,
)

# Each calibration image must be an RGB NumPy array with values in [0, 1].
calibration_images = [
    # image_1_rgb_array,
    # image_2_rgb_array,
]

config = DPViTConfig(
    num_classes=5,
    image_size=224,
)

ics_calibration = fit_ics_calibration(
    calibration_images=calibration_images,
    region_size=config.region_size,
    config=ICSConfig(),
)

# For each dataset image, compute its patch map from the RGB image
# before channel normalization.
image_rgb_tensor = ...  # Tensor with shape (3, 224, 224), values in [0, 1]
label = 0

patch_size_map = compute_patch_size_map_for_image(
    image=image_rgb_tensor.permute(1, 2, 0).cpu().numpy(),
    calibration=ics_calibration,
    region_size=config.region_size,
    patch_sizes=config.patch_sizes,
)

# Apply your chosen normalization after computing the patch map.
image_for_model = image_rgb_tensor

dataset = [
    (image_for_model, patch_size_map, label),
    # Add more (image_tensor, patch_size_map, label) samples here.
]

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=partial(
        adaptive_collate_fn,
        region_size=config.region_size,
        patch_sizes=config.patch_sizes,
    ),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPViT(config).to(device)

# Example forward pass. Use your own loss, optimizer, and training loop.
batch = next(iter(loader))
batch = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}

logits = model(batch)
targets = batch["labels"]
```

The dataset samples passed to `adaptive_collate_fn` must have this structure:

```python
(image_tensor, patch_size_map, label)
```

`image_tensor` must have shape `(3, height, width)`. The patch map must be
computed from the RGB image before channel normalization.

To avoid padding between images, use `TierBatchSampler` with tiers based on
each image's token counts per patch size. A total token count alone may not
avoid padding when the patch-size composition differs:

```python
import numpy as np

from dp_vit import TierBatchSampler

tier_ids = {}
tiers = []
for _, patch_map, _ in dataset:
    token_counts = tuple(
        np.count_nonzero(patch_map == size)
        * (config.region_size // size) ** 2
        for size in config.patch_sizes
    )
    tiers.append(tier_ids.setdefault(token_counts, len(tier_ids)))

loader = DataLoader(
    dataset,
    batch_sampler=TierBatchSampler(tiers, batch_size=2, shuffle=True, seed=0),
    collate_fn=partial(
        adaptive_collate_fn,
        region_size=config.region_size,
        patch_sizes=config.patch_sizes,
    ),
)
```

`batch_size` is the maximum size for each tier; the final batch in a tier may
be smaller.

## Optional pruning and checkpoint saving

Apply pruning after training. Evaluate the pruned model on your validation or
test data to measure its effect.

```python
model, pruning_report = prune_mlp_by_magnitude(
    model,
    PruningConfig(target_ratio=0.2),
)

save_checkpoint(
    path="checkpoints/dp_vit.pth",
    model=model,
    ics_calibration=ics_calibration,
    class_names=["class_a", "class_b", "class_c", "class_d", "class_e"],
    pruning_report=pruning_report,
)
```

## Loading a checkpoint and inference

The inference input must be an RGB tensor with shape `(3, image_size, image_size)`
and values in `[0, 1]`. Apply the same resize and normalization conventions used
during training.

```python
from dp_vit import load_checkpoint, predict_tensor

model, ics_calibration, metadata = load_checkpoint(
    "checkpoints/dp_vit.pth",
    map_location=device,
)

result = predict_tensor(
    model=model,
    image_before_normalization=image_rgb_tensor,
    ics_calibration=ics_calibration,
    class_names=metadata["class_names"],
    normalize_fn=None,
)

print(result["class_name"])
print(result["confidence"])
print(result["patch_size_map"])
```

If training used channel normalization, pass the same normalization function
through `normalize_fn` during inference.

## Project structure

```text
DP-ViT/
├── LICENSE
├── README.md
├── pyproject.toml
└── src/
    └── dp_vit/
        ├── __init__.py
        ├── config.py
        ├── ics.py
        ├── patching.py
        ├── model.py
        ├── pruning.py
        ├── checkpoint.py
        └── inference.py
```

## Scope

DP-ViT provides model components. It does not provide:

- Dataset-specific loaders or downloads
- Image cleaning or augmentation pipelines
- A training loop, optimizer, or scheduler
- A baseline Vision Transformer implementation
- A guarantee of faster training or inference

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).