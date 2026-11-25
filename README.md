# DP-ViT

DP-ViT is a PyTorch library for image classification with ICS-guided adaptive
patch sizes. It also provides signature-aware batch samplers and optional
structured magnitude-based pruning for the Transformer MLP layers.

The library provides:

- Information Complexity Score (ICS) calculation and calibration
- Adaptive patch-size mapping (**Region-Adaptive Mode** and **Budget-Tiered Mode**)
- Adaptive patch embedding and collate utilities (`adaptive_collate_fn`, `fast_adaptive_collate_fn`)
- Signature-aware batch samplers (`TolerantBucketBatchSampler`, `ExactSignatureBatchSampler`)
- A Vision Transformer classification model (`DPViT`)
- Optional structured MLP pruning
- Checkpoint save and load utilities
- Single-image inference

Dataset loading, image cleaning, augmentation, normalization, and training
loops are handled by the user.

> **Status:** DP-ViT is under active development. Its API may evolve before a
> `1.0.0` stable release.

## How it works

DP-ViT divides an image into non-overlapping regions (default `32×32`) and
computes an Information Complexity Score (ICS) for each region. Regions with
higher complexity receive smaller patches (`8×8`), medium complexity regions
receive `16×16`, and simpler regions receive larger patches (`32×32`).

The patches are projected into tokens and passed to a Transformer encoder.
After training, users can optionally apply structured magnitude-based pruning
to the MLP layers. Pruning does not automatically fine-tune the model.

### Patch Assignment Modes

DP-ViT supports two patch assignment modes:

- **Region-Adaptive Mode (`mode="region_adaptive"`):** Evaluates each region
  independently against global ICS thresholds (`t_low` / `t_high`, default P30
  and P70). Ideal for maximizing fine-grained spatial detail. Best paired with
  `TolerantBucketBatchSampler`.
- **Budget-Tiered Mode (`mode="budget_tiered"`):** Assigns each image to a
  fixed token-budget tier based on `detail_threshold`, then ranks regions by
  ICS to fill the tier's exact patch quota. Produces predictable memory usage
  and 0% padding when paired with `ExactSignatureBatchSampler`, making it ideal
  for edge deployment.

## Optional ICS token budget

By default, ICS calibration uses the global 30th and 70th percentiles to
assign adaptive patch sizes. You can optionally search for global thresholds
that keep the **average number of valid patch tokens per calibration image**
at or below a target.

This is an average over the calibration images, not a strict limit for each
image.

```python
from dp_vit import ICSConfig, fit_ics_calibration

target_tokens = (224 // 16) ** 2  # 196 patch tokens (equivalent to a 224x224 ViT-16)

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

print(
    "Selected percentiles:",
    ics_calibration.low_percentile,
    ics_calibration.high_percentile,
)
print(
    "Estimated valid tokens per image:",
    ics_calibration.estimated_tokens_per_image,
)
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
5. Group samples using `TolerantBucketBatchSampler` (Region-Adaptive) or
   `ExactSignatureBatchSampler` (Budget-Tiered) and collate batches with
   `adaptive_collate_fn` or `fast_adaptive_collate_fn`.
6. Train `DPViT` with your own training loop.
7. Optionally prune the trained model.
8. Save the checkpoint and use it for inference.

## Example

This example shows how to create the model, prepare a batch, and run a forward
pass. It does not include a training loop or dataset implementation.

```python
from functools import partial

import torch
from torch.utils.data import DataLoader

from dp_vit import (
    DPViT,
    DPViTConfig,
    ICSConfig,
    adaptive_collate_fn,
    compute_patch_size_map_for_image,
    fit_ics_calibration,
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

The dataset samples passed to `adaptive_collate_fn` or
`fast_adaptive_collate_fn` must have this structure:

```python
(image_tensor, patch_size_map, label)
```

`image_tensor` must have shape `(3, height, width)`. The patch map must be
computed from the RGB image before channel normalization.

## Signature-Aware Batch Samplers

Because adaptive images have varying token counts per patch branch, naive random
batching introduces heavy padding. Use one of the built-in batch samplers to
minimize padding:

### 1. `TolerantBucketBatchSampler` (Recommended for Region-Adaptive Mode)

Groups images with similar token signatures within a configurable per-branch
tolerance (`max_token_delta`), keeping batch sizes large while minimizing
padding:

```python
from functools import partial
from torch.utils.data import DataLoader

from dp_vit import (
    TolerantBucketBatchSampler,
    fast_adaptive_collate_fn,
    get_patch_signature,
)

signatures = [
    get_patch_signature(
        patch_map,
        region_size=config.region_size,
        patch_sizes=config.patch_sizes,
    )
    for _, patch_map, _ in dataset
]

# Coarse bucket signatures (e.g., binned by 32 tokens per branch)
coarse_signatures = [
    tuple(count // 32 for count in sig) for sig in signatures
]

sampler = TolerantBucketBatchSampler(
    signatures=signatures,
    coarse_signatures=coarse_signatures,
    batch_size=16,
    max_token_delta=(32, 16, 8),
    shuffle=True,
    seed=0,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=partial(
        fast_adaptive_collate_fn,
        region_size=config.region_size,
        patch_sizes=config.patch_sizes,
    ),
)
```

### 2. `ExactSignatureBatchSampler` (Recommended for Budget-Tiered Mode)

Groups images with identical token signatures for **0% padding**:

```python
from dp_vit import ExactSignatureBatchSampler

sampler = ExactSignatureBatchSampler(
    signatures=signatures,
    batch_size=16,
    shuffle=True,
    seed=0,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=partial(
        fast_adaptive_collate_fn,
        region_size=config.region_size,
        patch_sizes=config.patch_sizes,
    ),
)
```

> **Backward compatibility:** `SignatureBatchSampler` and
> `ApproximateSignatureBatchSampler` remain available as aliases for
> `ExactSignatureBatchSampler` and `TolerantBucketBatchSampler`.

## Optional pruning and checkpoint saving

Apply pruning after training. Evaluate the pruned model on your validation or
test data to measure its effect.

```python
from dp_vit import PruningConfig, prune_mlp_by_magnitude, save_checkpoint

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
    mode="region_adaptive",  # or "budget_tiered"
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
        ├── batching.py
        ├── checkpoint.py
        ├── config.py
        ├── ics.py
        ├── inference.py
        ├── model.py
        ├── patching.py
        └── pruning.py
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