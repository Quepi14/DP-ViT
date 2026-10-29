"""Adaptive patch extraction, batching, and embedding for DP-ViT."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch import Tensor

from .config import DPViTConfig

def _extract_image_patches(
    image: Tensor,
    patch_size_map: np.ndarray,
    region_size: int,
    patch_sizes: tuple[int, int, int],
) -> dict[int, list[tuple[Tensor, int]]]:
    """Extract image patches and their positional indices by patch size."""
    if image.ndim != 3:
        raise ValueError("Each image must have shape (channels, height, width).")

    channels, height, width = image.shape

    if height % region_size != 0 or width % region_size != 0:
        raise ValueError("Image height and width must be divisible by region_size.")

    expected_map_shape = (height // region_size, width // region_size)
    if patch_size_map.shape != expected_map_shape:
        raise ValueError(
            f"patch_size_map must have shape {expected_map_shape},"
            f"got {patch_size_map.shape}"
        )

    if not np.isin(patch_size_map, patch_sizes).all():
        raise ValueError(f"patch_size_map values must be selected from {patch_sizes}")

    smallest_patch = min(patch_sizes)
    base_grid_width = width // smallest_patch

    patches = {size: [] for size in patch_sizes}

    for region_row in range(patch_size_map.shape[0]):
        for region_col in range(patch_size_map.shape[1]):
            patch_size = int(patch_size_map[region_row, region_col])

            region_top = region_row * region_size
            region_left = region_col * region_size
            patches_per_side = region_size // patch_size

            for patch_row in range(patches_per_side):
                for patch_col in range(patches_per_side):
                    top = region_top + patch_row * patch_size
                    left = region_left + patch_col * patch_size

                    patch = image[
                        :,
                        top : top + patch_size,
                        left : left + patch_size,
                    ]
                    flat_patch = patch.reshape(channels * patch_size * patch_size)

                    base_row = top // smallest_patch
                    base_col = left // smallest_patch
                    position_id = base_row * base_grid_width + base_col

                    patches[patch_size].append((flat_patch, position_id))

    return patches

def adaptive_collate_fn (
    batch: Sequence[tuple[Tensor, np.ndarray, int]],
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> dict[str, Tensor]:
    """
    Convert image samples with patch maps into a padded adaptive-token batch.

    Each sample is expected to be:
        (image_tensor, patch_size_map, label)

    The image tensor may already be normalized for the model. the patch map,
    however, must have been computed from the image before channel normalization.
    """
    if not batch:
        raise ValueError("batch must contain at least one sample.")

    if len(patch_sizes) != 3 or not (
        patch_sizes[0] < patch_sizes[1] < patch_sizes[2]
    ):
        raise ValueError(
            "patch_sizes must contain three values ordered from smallest to largest."
        )

    if isinstance(region_size, bool) or not isinstance(region_size, int):
        raise TypeError("region_size must be an integer.")

    if region_size <= 0:
        raise ValueError("region_size must be greater than zero.")

    if any(
        isinstance(size, bool) or not isinstance(size, int)
        for size in patch_sizes
    ):
        raise TypeError("Each patch size must be an integer.")

    if any(size <= 0 for size in patch_sizes):
        raise ValueError("Each patch size must be greater than zero.")

    if any(region_size % size != 0 for size in patch_sizes):
        raise ValueError("region_size must be divisible by every patch size.")

    expected_image_shape = None
    extracted_batch = []
    labels = []

    for image_tensor, patch_size_map, label in batch:
        if not isinstance(image_tensor, Tensor):
            raise TypeError("Each image tensor must be a torch.Tensor.")

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError("Each image tensor must have shape (3, height, width) for RGB input.")

        if isinstance(patch_size_map, Tensor):
            patch_size_map = patch_size_map.detach().cpu().numpy()
        else:
            patch_size_map = np.asarray(patch_size_map)

        if patch_size_map.ndim != 2:
            raise ValueError("Each patch_size_map must be a 2D array.")

        image_shape = tuple(image_tensor.shape)

        if expected_image_shape is None:
            expected_image_shape = image_shape
        elif image_shape != expected_image_shape:
            raise ValueError("All images in a batch must have the same shape.")

        extracted = _extract_image_patches(
            image=image_tensor,
            patch_size_map=patch_size_map,
            region_size=region_size,
            patch_sizes=patch_sizes
        )
        extracted_batch.append(extracted)

        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise TypeError("Each label must be an integer") 

        labels.append(int(label))

    output: dict[str, Tensor] = {
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    assert expected_image_shape is not None
    channels = expected_image_shape[0]

    for patch_size in patch_sizes:
        patch_dim = channels * patch_size * patch_size
        max_patches = max(
            len(image_patches[patch_size])
            for image_patches in extracted_batch
        )
        max_patches = max(max_patches, 1)

        batch_size = len(batch)
        patch_tensor = torch.zeros(
            batch_size,
            max_patches,
            patch_dim,
            dtype=batch[0][0].dtype,
        )
        position_tensor = torch.zeros(
            batch_size,
            max_patches,
            dtype=torch.long,
        )
        valid_mask = torch.zeros(
            batch_size,
            max_patches,
            dtype=torch.bool
        )

        for image_index, image_patches in enumerate(extracted_batch):
            items = image_patches[patch_size]

            for patch_index, (flat_patch, position_id) in enumerate(items):
                patch_tensor[image_index, patch_index] = flat_patch
                position_tensor[image_index, patch_index] = position_id
                valid_mask[image_index, patch_index] = True

        output[f"patches_{patch_size}"] = patch_tensor
        output[f"position_{patch_size}"] = position_tensor
        output[f"mask_{patch_size}"] = valid_mask

    return output

def get_padded_sequence_length(
    batch: dict[str, Tensor],
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
    include_class_token: bool = True,
) -> int:
    """Return encoder sequence length, including group padding and class token."""
    sequence_length = 0

    for patch_size in patch_sizes:
        key = f"patches_{patch_size}"

        if key not in batch:
            raise KeyError(f"Batch is missing {key}.")

        patches = batch[key]

        if patches.ndim != 3:
            raise ValueError(
                f"{key} must have shape (batch, tokens, patch_dim)."
            )

        sequence_length += patches.shape[1]

    if include_class_token:
        sequence_length += 1

    return sequence_length

class AdaptivePatchEmbedding(nn.Module):
    """Project each supported patch size into a shared token space."""

    def __init__(self, config: DPViTConfig) -> None:
        super().__init__()

        self.patch_size = config.patch_sizes
        self.embed_dim = config.embed_dim

        self.projections = nn.ModuleDict(
            {
                str(size): nn.Linear(
                    3 * size * size,
                    config.embed_dim
                )

                for size in self.patch_size
            }
        )

        image_height = image_width = config.image_size
        smallest_patch = min(self.patch_size)
        max_position_cells = (
            image_height // smallest_patch
        ) * (image_width // smallest_patch)

        self.position_embedding = nn.Embedding(
            max_position_cells,
            config.embed_dim,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Convert padded patches into one token sequence per image.

        Returns:
            tokens: Tensor with shape (batch, sequence_length, embed_dim)
            valid_mask: Boolean tensor where True marks a real token
        """
        token_groups = []
        mask_groups = []

        for patch_size in self.patch_size:
            patch_key = f"patches_{patch_size}"
            position_key = f"position_{patch_size}"
            mask_key = f"mask_{patch_size}"

            if patch_key not in batch or position_key not in batch or mask_key not in batch:
                raise KeyError(
                    f"Batch is missing patchig data for size {patch_size}"
                )

            patches = batch[patch_key]
            positions = batch[position_key]
            valid_mask = batch[mask_key]

            if patches.ndim != 3:
                raise ValueError(
                    f"{patch_key} must have shape (batch, tokens, patch_dim)."
                )

            if positions.shape != patches.shape[:2]:
                raise ValueError(
                    f"{position_key} shape must match the first two"
                    f"dimensions of {patch_key}."
                )

            if valid_mask.shape != patches.shape[:2]:
                raise ValueError(
                    f"{mask_key} shape must match the first two"
                    f"dimensions of {patch_key}"
                )

            tokens = self.projections[str(patch_size)](patches)
            tokens = tokens + self.position_embedding(positions)

            # Remove positional values from padded tokens.
            tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)

            token_groups.append(tokens)
            mask_groups.append(valid_mask)

        all_tokens = torch.cat(token_groups, dim=1)
        all_valid_mask = torch.cat(mask_groups, dim=1)

        return all_tokens, all_valid_mask


def _vectorized_group_arrays(
    images: np.ndarray,
    patch_maps: np.ndarray,
    region_size: int,
    patch_sizes: tuple[int, int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Vectorized patch extraction for a whole batch (no Python loop per patch).

    images:     (B, C, H, W)
    patch_maps: (B, H // region_size, W // region_size)
    Returns {patch_size: (patches (B, N, C*s*s), positions (B, N), valid (B, N))}
    with the same padding rule as adaptive_collate_fn (pad per patch size).
    Only the selected patches are copied.
    """
    batch, channels, height, width = images.shape
    smallest = min(patch_sizes)
    base_grid_width = width // smallest
    result = {}

    for size in patch_sizes:
        rows, cols = height // size, width // size
        view = images.reshape(batch, channels, rows, size, cols, size)

        token_rows = np.repeat(np.arange(rows), cols)
        token_cols = np.tile(np.arange(cols), rows)
        region_r = token_rows * size // region_size
        region_c = token_cols * size // region_size
        selected = patch_maps[:, region_r, region_c] == size          # (B, rows*cols)

        counts = selected.sum(axis=1)
        max_tokens = max(int(counts.max()), 1)
        starts = np.cumsum(counts) - counts

        batch_idx, token_idx = np.nonzero(selected)                   # raster order per image
        slot = np.arange(batch_idx.size) - starts[batch_idx]
        row_idx, col_idx = token_rows[token_idx], token_cols[token_idx]

        chosen = view[batch_idx, :, row_idx, :, col_idx, :]           # (N, C, s, s)

        patches = np.zeros((batch, max_tokens, channels * size * size), dtype=images.dtype)
        positions = np.zeros((batch, max_tokens), dtype=np.int64)
        patches[batch_idx, slot] = chosen.reshape(batch_idx.size, -1)
        positions[batch_idx, slot] = (row_idx * size // smallest) * base_grid_width + (col_idx * size // smallest)
        valid = np.arange(max_tokens)[None, :] < counts[:, None]

        result[size] = (patches, positions, valid)

    return result


def fast_adaptive_collate_fn(
    batch: Sequence[tuple[Tensor, np.ndarray, int]],
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> dict[str, Tensor]:
    """
    Drop-in, much faster replacement for adaptive_collate_fn.

    Same input samples and same output keys, but patches are extracted with
    reshape/gather instead of a Python loop over every patch. Token ORDER inside
    each patch-size group differs from adaptive_collate_fn; this does not change
    model output because attention is permutation-equivariant and every token
    carries its own position embedding.
    """
    if not batch:
        raise ValueError("batch must contain at least one sample.")

    images = torch.stack([sample[0] for sample in batch])
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Each image tensor must have shape (3, height, width) for RGB input.")

    _, _, height, width = images.shape
    if height % region_size != 0 or width % region_size != 0:
        raise ValueError("Image height and width must be divisible by region_size.")

    patch_maps = np.stack([np.asarray(sample[1]) for sample in batch])
    if patch_maps.shape[1:] != (height // region_size, width // region_size):
        raise ValueError("patch_size_map shape does not match image / region_size.")
    if not np.isin(patch_maps, patch_sizes).all():
        raise ValueError(f"patch_size_map values must be selected from {patch_sizes}")

    groups = _vectorized_group_arrays(images.numpy(), patch_maps, region_size, patch_sizes)

    output = {"labels": torch.tensor([int(sample[2]) for sample in batch], dtype=torch.long)}
    for size, (patches, positions, valid) in groups.items():
        output[f"patches_{size}"] = torch.from_numpy(np.ascontiguousarray(patches)).to(images.dtype)
        output[f"position_{size}"] = torch.from_numpy(positions).long()
        output[f"mask_{size}"] = torch.from_numpy(valid)

    return output