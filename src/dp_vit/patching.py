"""Adaptive patch extraction, batching, and embedding for DP-ViT."""

from collections.abc import Sequence
from math import lcm
from numbers import Integral

import numpy as np
import torch
from torch import Tensor, nn

from .config import DPViTConfig


Sample = tuple[Tensor, np.ndarray | Tensor, int]


def _validate_adaptive_config(
    region_size: int,
    patch_sizes: Sequence[int],
) -> tuple[int, tuple[int, int, int]]:
    """Validate and normalize the adaptive patch configuration."""
    if isinstance(region_size, bool) or not isinstance(region_size, Integral):
        raise TypeError("region_size must be an integer.")

    region_size = int(region_size)

    if region_size <= 0:
        raise ValueError("region_size must be greater than zero.")

    if not isinstance(patch_sizes, (tuple, list)):
        raise TypeError("patch_sizes must be a tuple or list.")

    if len(patch_sizes) != 3:
        raise ValueError("patch_sizes must contain exactly three values.")

    if any(
        isinstance(size, bool) or not isinstance(size, Integral)
        for size in patch_sizes
    ):
        raise TypeError("Each patch size must be an integer.")

    normalized = (
        int(patch_sizes[0]),
        int(patch_sizes[1]),
        int(patch_sizes[2]),
    )

    if not (0 < normalized[0] < normalized[1] < normalized[2]):
        raise ValueError(
            "patch_sizes must be positive and strictly ascending."
        )

    if any(region_size % size != 0 for size in normalized):
        raise ValueError(
            "region_size must be divisible by every patch size."
        )

    return region_size, normalized


def _to_numpy_patch_map(
    patch_size_map: np.ndarray | Tensor,
) -> np.ndarray:
    """Convert a patch map to a CPU NumPy array without retaining gradients."""
    if isinstance(patch_size_map, Tensor):
        patch_size_map = patch_size_map.detach().cpu().numpy()
    else:
        patch_size_map = np.asarray(patch_size_map)

    if patch_size_map.ndim != 2:
        raise ValueError("patch_size_map must be a 2D array.")

    if patch_size_map.size == 0:
        raise ValueError("patch_size_map must not be empty.")

    return patch_size_map


def _validate_patch_map(
    patch_size_map: np.ndarray | Tensor,
    expected_shape: tuple[int, int],
    patch_sizes: tuple[int, int, int],
) -> np.ndarray:
    patch_map = _to_numpy_patch_map(patch_size_map)

    if patch_map.shape != expected_shape:
        raise ValueError(
            "patch_size_map shape must match the image region grid. "
            f"Expected {expected_shape}, got {patch_map.shape}."
        )

    if not np.isin(patch_map, patch_sizes).all():
        raise ValueError(
            f"patch_size_map values must be selected from {patch_sizes}."
        )

    return patch_map


def _validate_label(label: object) -> int:
    if isinstance(label, (bool, np.bool_)) or not isinstance(
        label, Integral
    ):
        raise TypeError(
            "Each label must be an integer "
            "(boolean values are not accepted)."
        )

    return int(label)


def _validate_image_tensor(image: Tensor) -> None:
    if not isinstance(image, Tensor):
        raise TypeError("Each image tensor must be a torch.Tensor.")

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            "Each image tensor must have shape "
            "(3, height, width) for RGB input."
        )

    if not image.is_floating_point():
        raise TypeError("Image tensors must use a floating-point dtype.")


def _extract_image_patches(
    image: Tensor,
    patch_size_map: np.ndarray | Tensor,
    region_size: int,
    patch_sizes: tuple[int, int, int],
) -> dict[int, list[tuple[Tensor, int]]]:
    """Extract patches in region-major order, grouped by patch size."""
    region_size, patch_sizes = _validate_adaptive_config(
        region_size,
        patch_sizes,
    )

    _validate_image_tensor(image)

    _, height, width = image.shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image height and width must be greater than zero."
        )

    if height % region_size or width % region_size:
        raise ValueError(
            "Image height and width must be divisible by region_size."
        )

    patch_map = _validate_patch_map(
        patch_size_map,
        (height // region_size, width // region_size),
        patch_sizes,
    )

    channels = image.shape[0]
    smallest_patch = patch_sizes[0]
    base_grid_width = width // smallest_patch

    grouped: dict[int, list[tuple[Tensor, int]]] = {
        size: [] for size in patch_sizes
    }

    for region_row in range(patch_map.shape[0]):
        for region_col in range(patch_map.shape[1]):
            size = int(patch_map[region_row, region_col])

            top0 = region_row * region_size
            left0 = region_col * region_size

            patches_per_side = region_size // size

            for patch_row in range(patches_per_side):
                for patch_col in range(patches_per_side):
                    top = top0 + patch_row * size
                    left = left0 + patch_col * size

                    patch = image[
                        :,
                        top:top + size,
                        left:left + size,
                    ]

                    flat = patch.reshape(channels * size * size)

                    position = (
                        (top // smallest_patch) * base_grid_width
                        + (left // smallest_patch)
                    )

                    grouped[size].append((flat, position))

    return grouped


def get_patch_signature(
    patch_size_map: np.ndarray | Tensor,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> tuple[int, int, int]:
    """Return the number of extracted patches for each supported patch size."""
    region_size, patch_sizes = _validate_adaptive_config(
        region_size,
        patch_sizes,
    )

    patch_map = _to_numpy_patch_map(patch_size_map)

    if not np.isin(patch_map, patch_sizes).all():
        raise ValueError(
            f"patch_size_map values must be selected from {patch_sizes}."
        )

    return tuple(
        int(np.count_nonzero(patch_map == size))
        * (region_size // size) ** 2
        for size in patch_sizes
    )  # type: ignore[return-value]


def _validate_batch_samples(
    batch: Sequence[Sample],
    region_size: int,
    patch_sizes: Sequence[int],
) -> tuple[
    int,
    tuple[int, int, int],
    tuple[int, int, int],
    torch.dtype,
    torch.device,
    list[np.ndarray],
    list[int],
]:
    if not isinstance(batch, Sequence) or len(batch) == 0:
        raise ValueError("batch must contain at least one sample.")

    region_size, patch_sizes = _validate_adaptive_config(
        region_size,
        patch_sizes,
    )

    first_sample = batch[0]

    if not isinstance(first_sample, (tuple, list)) or len(first_sample) != 3:
        raise ValueError(
            "Sample 0 must be a 3-item "
            "(image, patch_map, label) sequence."
        )

    first_image = first_sample[0]
    _validate_image_tensor(first_image)

    expected_shape: tuple[int, int, int] = (
        int(first_image.shape[0]),
        int(first_image.shape[1]),
        int(first_image.shape[2]),
    )
    expected_dtype = first_image.dtype
    expected_device = first_image.device

    height, width = expected_shape[-2:]

    if (
        height <= 0
        or width <= 0
        or height % region_size
        or width % region_size
    ):
        raise ValueError(
            "Image height and width must be positive "
            "and divisible by region_size."
        )

    maps: list[np.ndarray] = []
    labels: list[int] = []

    expected_map_shape = (
        height // region_size,
        width // region_size,
    )

    for index, sample in enumerate(batch):
        if not isinstance(sample, (tuple, list)) or len(sample) != 3:
            raise ValueError(
                f"Sample {index} must be a 3-item "
                "(image, patch_map, label) sequence."
            )

        image, patch_map, label = sample

        _validate_image_tensor(image)

        if tuple(image.shape) != expected_shape:
            raise ValueError(
                "All images in a batch must have the same shape."
            )

        if image.dtype != expected_dtype:
            raise ValueError(
                "All image tensors in a batch must have the same dtype."
            )

        if image.device != expected_device:
            raise ValueError(
                "All image tensors in a batch must be on the same device."
            )

        maps.append(
            _validate_patch_map(
                patch_map,
                expected_map_shape,
                patch_sizes,
            )
        )

        labels.append(_validate_label(label))

    return (
        region_size,
        patch_sizes,
        expected_shape,
        expected_dtype,
        expected_device,
        maps,
        labels,
    )


def adaptive_collate_fn(
    batch: Sequence[Sample],
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> dict[str, Tensor]:
    """Reference collate function using straightforward per-image patch extraction."""
    (
        region_size,
        patch_sizes,
        image_shape,
        dtype,
        device,
        maps,
        labels,
    ) = _validate_batch_samples(
        batch,
        region_size,
        patch_sizes,
    )

    extracted = [
        _extract_image_patches(
            sample[0],
            patch_map,
            region_size,
            patch_sizes,
        )
        for sample, patch_map in zip(batch, maps)
    ]

    batch_size = len(batch)
    channels = image_shape[0]

    output: dict[str, Tensor] = {
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )
    }

    for size in patch_sizes:
        max_count = max(len(item[size]) for item in extracted)

        patches = torch.zeros(
            (batch_size, max_count, channels * size * size),
            dtype=dtype,
            device=device,
        )

        positions = torch.zeros(
            (batch_size, max_count),
            dtype=torch.long,
            device=device,
        )

        mask = torch.zeros(
            (batch_size, max_count),
            dtype=torch.bool,
            device=device,
        )

        if max_count > 0:
            for batch_index, grouped in enumerate(extracted):
                for token_index, (flat_patch, position_id) in enumerate(
                    grouped[size]
                ):
                    patches[batch_index, token_index] = flat_patch
                    positions[batch_index, token_index] = position_id
                    mask[batch_index, token_index] = True

        output[f"patches_{size}"] = patches
        output[f"position_{size}"] = positions
        output[f"mask_{size}"] = mask

    return output


def get_padded_sequence_length(
    batch: dict[str, Tensor],
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
    include_class_token: bool = True,
) -> int:
    """Return padded encoder sequence length, optionally including the class token."""
    patch_values = (
        tuple(patch_sizes)
        if isinstance(patch_sizes, (tuple, list))
        else patch_sizes
    )

    if not isinstance(patch_values, tuple) or len(patch_values) != 3:
        raise ValueError(
            "patch_sizes must contain exactly three values."
        )

    if any(
        isinstance(size, bool) or not isinstance(size, Integral)
        for size in patch_values
    ):
        raise TypeError("Each patch size must be an integer.")

    _, patch_sizes = _validate_adaptive_config(
        lcm(*(int(size) for size in patch_values)),
        patch_values,
    )

    if not isinstance(include_class_token, bool):
        raise TypeError("include_class_token must be a boolean.")

    if not isinstance(batch, dict):
        raise TypeError("batch must be a dictionary of tensors.")

    sequence_length = 0
    expected_batch_size: int | None = None

    for size in patch_sizes:
        key = f"patches_{size}"

        if key not in batch:
            raise KeyError(f"Batch is missing {key}.")

        patches = batch[key]

        if not isinstance(patches, Tensor):
            raise TypeError(f"{key} must be a torch.Tensor.")

        if patches.ndim != 3 or patches.shape[0] == 0:
            raise ValueError(
                f"{key} must have shape (batch, tokens, patch_dim) "
                "with non-empty batch dimension."
            )

        if expected_batch_size is None:
            expected_batch_size = patches.shape[0]
        elif patches.shape[0] != expected_batch_size:
            raise ValueError(
                "All patch groups must have the same batch dimension."
            )

        sequence_length += int(patches.shape[1])

    return sequence_length + int(include_class_token)


class AdaptivePatchEmbedding(nn.Module):
    """Project adaptive-size patches into one shared token embedding space."""

    def __init__(self, config: DPViTConfig) -> None:
        super().__init__()

        if not isinstance(config, DPViTConfig):
            raise TypeError("config must be a DPViTConfig instance.")

        if (
            isinstance(config.embed_dim, bool)
            or not isinstance(config.embed_dim, Integral)
        ):
            raise TypeError(
                "config.embed_dim must be a positive integer."
            )

        if config.embed_dim <= 0:
            raise ValueError(
                "config.embed_dim must be greater than zero."
            )

        if (
            isinstance(config.image_size, bool)
            or not isinstance(config.image_size, Integral)
        ):
            raise TypeError(
                "config.image_size must be a positive integer."
            )

        if config.image_size <= 0:
            raise ValueError(
                "config.image_size must be greater than zero."
            )

        self.patch_sizes = tuple(config.patch_sizes)
        self.embed_dim = int(config.embed_dim)
        self.image_size = int(config.image_size)

        _, self.patch_sizes = _validate_adaptive_config(
            lcm(*self.patch_sizes),
            self.patch_sizes,
        )

        if any(
            self.image_size % size
            for size in self.patch_sizes
        ):
            raise ValueError(
                "config.image_size must be divisible by every patch size."
            )

        self.projections = nn.ModuleDict(
            {
                str(size): nn.Linear(
                    3 * size * size,
                    self.embed_dim,
                )
                for size in self.patch_sizes
            }
        )

        smallest = self.patch_sizes[0]
        position_count = (self.image_size // smallest) ** 2

        self.position_embedding = nn.Embedding(
            position_count,
            self.embed_dim,
        )

        self.scale_embedding = nn.Embedding(
            len(self.patch_sizes),
            self.embed_dim,
        )

        nn.init.trunc_normal_(
            self.position_embedding.weight,
            std=0.02,
        )
        nn.init.trunc_normal_(
            self.scale_embedding.weight,
            std=0.02,
        )

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Return concatenated tokens and a boolean mask (True means a real token)."""
        if not isinstance(batch, dict):
            raise TypeError("batch must be a dictionary of tensors.")

        token_groups: list[Tensor] = []
        mask_groups: list[Tensor] = []

        expected_batch_size: int | None = None
        expected_device: torch.device | None = None

        for scale_index, size in enumerate(self.patch_sizes):
            patch_key = f"patches_{size}"
            position_key = f"position_{size}"
            mask_key = f"mask_{size}"

            missing = [
                key
                for key in (patch_key, position_key, mask_key)
                if key not in batch
            ]

            if missing:
                raise KeyError(
                    "Batch is missing required keys: "
                    f"{', '.join(missing)}."
                )

            patches = batch[patch_key]
            positions = batch[position_key]
            valid_mask = batch[mask_key]

            if not all(
                isinstance(value, Tensor)
                for value in (patches, positions, valid_mask)
            ):
                raise TypeError(
                    f"{patch_key}, {position_key}, and {mask_key} "
                    "must be tensors."
                )

            if patches.ndim != 3:
                raise ValueError(
                    f"{patch_key} must have shape "
                    "(batch, tokens, patch_dim)."
                )

            if patches.shape[0] == 0:
                raise ValueError(
                    f"{patch_key} must have a non-empty batch dimension."
                )

            if patches.shape[2] != 3 * size * size:
                raise ValueError(
                    f"{patch_key} last dimension must be "
                    f"{3 * size * size}."
                )

            if positions.shape != patches.shape[:2]:
                raise ValueError(
                    f"{position_key} shape must match the first "
                    f"two dimensions of {patch_key}."
                )

            if positions.dtype != torch.long:
                raise TypeError(
                    f"{position_key} must use torch.long dtype."
                )

            if valid_mask.shape != patches.shape[:2]:
                raise ValueError(
                    f"{mask_key} shape must match the first "
                    f"two dimensions of {patch_key}."
                )

            if valid_mask.dtype != torch.bool:
                raise TypeError(
                    f"{mask_key} must use torch.bool dtype."
                )

            if not patches.is_floating_point():
                raise TypeError(
                    f"{patch_key} must use a floating-point dtype."
                )

            if positions.numel() and (
                torch.any(positions < 0)
                or torch.any(
                    positions >= self.position_embedding.num_embeddings
                )
            ):
                raise ValueError(
                    f"{position_key} contains an out-of-range "
                    "positional index."
                )

            if (
                patches.device != positions.device
                or patches.device != valid_mask.device
            ):
                raise ValueError(
                    f"All tensors in patch group {size} "
                    "must be on the same device."
                )

            projection = self.projections[str(size)]

            if patches.device != projection.weight.device:
                raise ValueError(
                    "Batch tensors and AdaptivePatchEmbedding "
                    "must be on the same device."
                )

            if patches.dtype != projection.weight.dtype:
                raise TypeError(
                    "Patch tensor dtype must match the "
                    "embedding projection dtype."
                )

            if (
                positions.device
                != self.position_embedding.weight.device
            ):
                raise ValueError(
                    "Position tensors and positional embedding "
                    "must be on the same device."
                )

            if expected_batch_size is None:
                expected_batch_size = patches.shape[0]
                expected_device = patches.device

            elif patches.shape[0] != expected_batch_size:
                raise ValueError(
                    "All patch groups must have the same batch dimension."
                )

            elif patches.device != expected_device:
                raise ValueError(
                    "All patch groups must be on the same device."
                )

            if patches.shape[1] == 0:
                continue

            tokens = projection(patches)
            tokens = tokens + self.position_embedding(positions)
            tokens = tokens + self.scale_embedding.weight[scale_index].to(
                dtype=tokens.dtype
            )

            tokens = tokens * valid_mask.unsqueeze(-1).to(
                dtype=tokens.dtype
            )

            token_groups.append(tokens)
            mask_groups.append(valid_mask)

        if not token_groups:
            raise ValueError(
                "Batch must contain at least one patch token across all sizes."
            )

        return (
            torch.cat(token_groups, dim=1),
            torch.cat(mask_groups, dim=1),
        )


def _vectorized_group_arrays(
    images: np.ndarray,
    patch_maps: np.ndarray,
    region_size: int,
    patch_sizes: tuple[int, int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Vectorized CPU patch extraction; tokens are ordered in raster order per group."""
    region_size, patch_sizes = _validate_adaptive_config(
        region_size,
        patch_sizes,
    )

    images = np.asarray(images)
    patch_maps = np.asarray(patch_maps)

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "images must have shape (batch, 3, height, width)."
        )

    batch_size, channels, height, width = images.shape

    if batch_size == 0 or height == 0 or width == 0:
        raise ValueError(
            "images must have non-empty batch, height, "
            "and width dimensions."
        )

    if not np.issubdtype(images.dtype, np.floating):
        raise TypeError(
            "images must use a floating-point dtype."
        )

    if height % region_size or width % region_size:
        raise ValueError(
            "Image height and width must be divisible by region_size."
        )

    expected_map_shape = (
        batch_size,
        height // region_size,
        width // region_size,
    )

    if patch_maps.shape != expected_map_shape:
        raise ValueError(
            f"patch_maps must have shape {expected_map_shape}, "
            f"got {patch_maps.shape}."
        )

    if not np.isin(patch_maps, patch_sizes).all():
        raise ValueError(
            f"patch_maps values must be selected from {patch_sizes}."
        )

    smallest = patch_sizes[0]
    base_grid_width = width // smallest

    result: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    for size in patch_sizes:
        rows = height // size
        cols = width // size

        view = images.reshape(
            batch_size,
            channels,
            rows,
            size,
            cols,
            size,
        )

        token_rows = np.repeat(np.arange(rows), cols)
        token_cols = np.tile(np.arange(cols), rows)

        region_rows = (token_rows * size) // region_size
        region_cols = (token_cols * size) // region_size

        selected = (
            patch_maps[:, region_rows, region_cols] == size
        )

        counts = selected.sum(axis=1, dtype=np.int64)
        max_tokens = int(counts.max())

        patches = np.zeros(
            (
                batch_size,
                max_tokens,
                channels * size * size,
            ),
            dtype=images.dtype,
        )

        positions = np.zeros(
            (batch_size, max_tokens),
            dtype=np.int64,
        )

        if max_tokens > 0:
            starts = np.cumsum(counts) - counts
            batch_indices, token_indices = np.nonzero(selected)

            slots = (
                np.arange(batch_indices.size)
                - starts[batch_indices]
            )

            row_indices = token_rows[token_indices]
            col_indices = token_cols[token_indices]

            chosen = view[
                batch_indices,
                :,
                row_indices,
                :,
                col_indices,
                :,
            ]

            patches[batch_indices, slots] = chosen.reshape(
                batch_indices.size,
                -1,
            )

            positions[batch_indices, slots] = (
                (
                    row_indices * size // smallest
                ) * base_grid_width
                + (col_indices * size // smallest)
            )

        valid = (
            np.arange(max_tokens)[None, :]
            < counts[:, None]
        )

        result[size] = (
            patches,
            positions,
            valid,
        )

    return result


def fast_adaptive_collate_fn(
    batch: Sequence[Sample],
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> dict[str, Tensor]:
    """Optimized CPU collate function with the same schema as adaptive_collate_fn.

    This implementation orders tokens in raster order within each patch-size
    group, whereas the reference function uses region-major order. Positions
    and masks are permuted with their tokens, so evaluation-time model results
    should be equivalent up to floating-point tolerance. Dropout can produce
    different stochastic training results because random masks follow sequence
    indices. Use the reference function as the correctness baseline.
    """
    (
        region_size,
        patch_sizes,
        image_shape,
        dtype,
        device,
        maps,
        labels,
    ) = _validate_batch_samples(
        batch,
        region_size,
        patch_sizes,
    )

    if device.type != "cpu":
        raise ValueError(
            "fast_adaptive_collate_fn requires CPU image tensors; "
            "use adaptive_collate_fn for other devices."
        )

    if any(sample[0].requires_grad for sample in batch):
        raise ValueError(
            "fast_adaptive_collate_fn does not support "
            "image tensors requiring gradients."
        )

    stacked = torch.stack([sample[0] for sample in batch]).contiguous()
    if dtype == torch.bfloat16:
        images = stacked.to(dtype=torch.float32).numpy()
    else:
        images = stacked.numpy()

    patch_maps = np.stack(maps)

    groups = _vectorized_group_arrays(
        images,
        patch_maps,
        region_size,
        patch_sizes,
    )

    output: dict[str, Tensor] = {
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        )
    }

    for size, (patches, positions, valid) in groups.items():
        tensor_patches = torch.from_numpy(np.ascontiguousarray(patches))
        if tensor_patches.dtype != dtype:
            tensor_patches = tensor_patches.to(dtype=dtype)

        output[f"patches_{size}"] = tensor_patches

        output[f"position_{size}"] = torch.from_numpy(
            positions
        )

        output[f"mask_{size}"] = torch.from_numpy(
            valid
        )

    return output