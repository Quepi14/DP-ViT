"""Information Complexity Score (ICS) calculations for DP-ViT."""

import math
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import sobel

from .config import ICSCalibration, ICSConfig


DEFAULT_PATCH_SIZES = (8, 16, 32)

# (number of 8x8 regions, number of 16x16 regions)
DEFAULT_TIERS = (
    (0, 10),
    (2, 12),
    (4, 14),
    (6, 12),
)


def _validate_region_size(region_size: int) -> int:
    """Validate region size."""
    if isinstance(region_size, bool) or not isinstance(region_size, Integral):
        raise TypeError("region_size must be an integer.")

    if region_size <= 0:
        raise ValueError("region_size must be greater than 0.")

    return int(region_size)


def _validate_patch_sizes(
    patch_sizes: tuple[int, int, int],
    region_size: int | None = None,
) -> tuple[int, int, int]:
    """Validate and normalize the three patch sizes."""
    if not isinstance(patch_sizes, (tuple, list)) or len(patch_sizes) != 3:
        raise ValueError("patch_sizes must contain exactly three values.")

    if any(
        isinstance(size, bool) or not isinstance(size, Integral)
        for size in patch_sizes
    ):
        raise TypeError("All patch sizes must be integers.")

    sizes = (
        int(patch_sizes[0]),
        int(patch_sizes[1]),
        int(patch_sizes[2]),
    )

    if any(size <= 0 for size in sizes):
        raise ValueError("All patch sizes must be greater than 0.")

    if not sizes[0] < sizes[1] < sizes[2]:
        raise ValueError(
            "patch_sizes must be ordered from smallest to largest."
        )

    if region_size is not None:
        region_size = _validate_region_size(region_size)

        if any(region_size % size != 0 for size in sizes):
            raise ValueError(
                "region_size must be divisible by every patch size."
            )

    return sizes


def _validate_ics_map(ics_map: np.ndarray) -> np.ndarray:
    """Validate a two-dimensional ICS map."""
    try:
        ics_map = np.asarray(ics_map, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("ics_map must contain numeric values.") from exc

    if ics_map.ndim != 2:
        raise ValueError("ics_map must be a 2D array.")

    if ics_map.size == 0:
        raise ValueError("ics_map must not be empty.")

    if not np.isfinite(ics_map).all():
        raise ValueError("ics_map must contain only finite values.")

    return ics_map


def _validate_calibration(
    calibration: ICSCalibration,
) -> ICSCalibration:
    """Validate an ICS calibration object."""
    if not isinstance(calibration, ICSCalibration):
        raise TypeError(
            "calibration must be an ICSCalibration instance."
        )

    for name, value in (
        ("alpha", calibration.alpha),
        ("beta", calibration.beta),
        ("t_low", calibration.t_low),
        ("t_high", calibration.t_high),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"calibration.{name} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(
                f"calibration.{name} must be finite."
            )

    if calibration.alpha <= 0 or calibration.beta <= 0:
        raise ValueError(
            "calibration alpha and beta must be positive."
        )

    if calibration.t_low > calibration.t_high:
        raise ValueError(
            "calibration.t_low must not exceed calibration.t_high."
        )

    return calibration


def _validate_calibration_images(calibration_images) -> tuple:
    """Convert calibration images to a reusable non-empty tuple."""
    if calibration_images is None:
        raise ValueError("calibration_images must not be None.")

    try:
        images = tuple(calibration_images)
    except TypeError as exc:
        raise TypeError(
            "calibration_images must be an iterable of images."
        ) from exc

    if not images:
        raise ValueError("calibration_images must not be empty.")

    return images


def _prepare_image_channels(image: np.ndarray) -> np.ndarray:
    """Validate and normalize a grayscale or RGB image to shape (H, W, C)."""
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("image must contain numeric values.") from exc

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(
            "image must have shape (height, width) "
            "or (height, width, 3)."
        )

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image dimensions must not be empty.")

    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values.")

    return arr


def _make_region_blocks(
    image_hwc: np.ndarray,
    region_size: int,
) -> np.ndarray:
    """Split a (H, W, C) array into non-overlapping square regions."""
    region_size = _validate_region_size(region_size)

    if image_hwc.ndim != 3:
        raise ValueError("image_hwc must be a 3D array (H, W, C).")

    height, width, channels = image_hwc.shape

    if height == 0 or width == 0:
        raise ValueError("image dimensions must not be empty.")

    if height % region_size != 0 or width % region_size != 0:
        raise ValueError(
            "Image height and width must be divisible by region_size."
        )

    rows = height // region_size
    cols = width // region_size

    blocks = image_hwc.reshape(
        rows,
        region_size,
        cols,
        region_size,
        channels,
    )

    blocks = blocks.transpose(0, 2, 1, 3, 4)

    return blocks.reshape(
        rows,
        cols,
        region_size * region_size,
        channels,
    )


def _compute_variance_from_hwc(
    image_hwc: np.ndarray,
    region_size: int,
) -> np.ndarray:
    """Compute mean per-channel local variance in each region."""
    blocks = _make_region_blocks(image_hwc, region_size)
    region_means = blocks.mean(axis=2, keepdims=True)
    per_channel_var = np.mean((blocks - region_means) ** 2, axis=2)
    return per_channel_var.mean(axis=-1)


def _compute_gradient_from_hwc(
    image_hwc: np.ndarray,
    region_size: int,
) -> np.ndarray:
    """Compute mean per-channel Sobel gradient magnitude in each region."""
    gradient_x = sobel(image_hwc, axis=1)
    gradient_y = sobel(image_hwc, axis=0)
    magnitude = np.hypot(gradient_x, gradient_y)
    blocks = _make_region_blocks(magnitude, region_size)
    return blocks.mean(axis=(2, 3))


def _compute_variance_and_gradient(
    image: np.ndarray,
    region_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both local variance and gradient magnitude in a single pass."""
    image_hwc = _prepare_image_channels(image)
    variance_map = _compute_variance_from_hwc(image_hwc, region_size)
    gradient_map = _compute_gradient_from_hwc(image_hwc, region_size)
    return variance_map, gradient_map


def compute_local_variance(
    image: np.ndarray,
    region_size: int = 32,
) -> np.ndarray:
    """
    Compute mean per-channel population variance in each region.

    Var(R) = mean_c(mean_p((pixel_{p,c} - region_mean_c) ** 2))
    """
    image_hwc = _prepare_image_channels(image)
    return _compute_variance_from_hwc(image_hwc, region_size)


def compute_gradient_magnitude(
    image: np.ndarray,
    region_size: int = 32,
) -> np.ndarray:
    """
    Compute mean per-channel Sobel gradient magnitude in each region.

    Sobel gradients are calculated on the full image before region averaging.
    """
    image_hwc = _prepare_image_channels(image)
    return _compute_gradient_from_hwc(image_hwc, region_size)


def compute_ics(
    variance_map: np.ndarray,
    gradient_map: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Combine local variance and gradient magnitude into an ICS map."""
    variance_map = _validate_ics_map(variance_map)
    gradient_map = _validate_ics_map(gradient_map)

    if variance_map.shape != gradient_map.shape:
        raise ValueError(
            "variance_map and gradient_map must have the same shape."
        )

    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a numeric value.")

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be a finite positive number."
            )

    ics_map = alpha * variance_map + beta * gradient_map

    if not np.isfinite(ics_map).all():
        raise ValueError("Computed ICS values must be finite.")

    return ics_map


def count_patch_tokens(
    patch_size_map: np.ndarray,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> int:
    """Count actual patch tokens represented by a region-level patch map."""
    region_size = _validate_region_size(region_size)
    patch_sizes = _validate_patch_sizes(patch_sizes, region_size)

    patch_size_map = np.asarray(patch_size_map)

    if patch_size_map.ndim != 2:
        raise ValueError("patch_size_map must be a 2D array.")

    if patch_size_map.size == 0:
        raise ValueError("patch_size_map must not be empty.")

    if not np.isin(patch_size_map, patch_sizes).all():
        raise ValueError(
            f"patch_size_map values must be selected from {patch_sizes}."
        )

    return int(
        sum(
            np.count_nonzero(patch_size_map == size)
            * (region_size // size) ** 2
            for size in patch_sizes
        )
    )


def _mean_tokens_from_sorted_ics(
    sorted_ics: np.ndarray,
    num_images: int,
    low_threshold: float,
    high_threshold: float,
    tokens_per_size: tuple[int, int, int],
) -> float:
    """Compute exact mean tokens per image in O(log N) using sorted ICS."""
    total_regions = sorted_ics.size
    tok_small, tok_medium, tok_large = tokens_per_size

    idx_low = int(np.searchsorted(sorted_ics, low_threshold, side="left"))
    idx_high = int(np.searchsorted(sorted_ics, high_threshold, side="left"))

    n_large = idx_low
    n_medium = max(0, idx_high - idx_low)
    n_small = total_regions - idx_high

    total_tokens = (
        n_small * tok_small
        + n_medium * tok_medium
        + n_large * tok_large
    )
    return total_tokens / num_images


def fit_ics_calibration(
    calibration_images,
    region_size: int = 32,
    config: ICSConfig | None = None,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> ICSCalibration:
    """Fit global ICS thresholds, optionally targeting a mean token budget."""
    calibration_images = _validate_calibration_images(calibration_images)

    region_size = _validate_region_size(region_size)
    patch_sizes = _validate_patch_sizes(patch_sizes, region_size)

    if config is None:
        config = ICSConfig()

    if not isinstance(config, ICSConfig):
        raise TypeError("config must be an ICSConfig instance.")

    variance_maps = []
    gradient_maps = []

    for image in calibration_images:
        var_map, grad_map = _compute_variance_and_gradient(
            image,
            region_size,
        )
        variance_maps.append(var_map)
        gradient_maps.append(grad_map)

    all_variances = np.concatenate(
        [item.ravel() for item in variance_maps]
    )

    all_gradients = np.concatenate(
        [item.ravel() for item in gradient_maps]
    )

    epsilon = 1e-8

    alpha = 1.0 / (all_variances.std() + epsilon)
    beta = 1.0 / (all_gradients.std() + epsilon)

    all_ics = alpha * all_variances + beta * all_gradients

    if not np.isfinite(all_ics).all():
        raise ValueError("Calibration produced non-finite ICS values.")

    num_images = len(variance_maps)
    sorted_ics = np.sort(all_ics)

    tokens_per_size = (
        (region_size // patch_sizes[0]) ** 2,
        (region_size // patch_sizes[1]) ** 2,
        (region_size // patch_sizes[2]) ** 2,
    )

    selected_low = config.low_percentile
    selected_high = config.high_percentile

    if config.target_mean_tokens_per_image is None:
        t_low = float(np.percentile(sorted_ics, selected_low))
        t_high = float(np.percentile(sorted_ics, selected_high))

        estimated_tokens = _mean_tokens_from_sorted_ics(
            sorted_ics,
            num_images,
            t_low,
            t_high,
            tokens_per_size,
        )

    else:
        region_counts = {item.size for item in variance_maps}

        if len(region_counts) != 1:
            raise ValueError(
                "Calibration images must contain the same number "
                "of regions when selecting a token budget."
            )

        minimum_budget = next(iter(region_counts)) * tokens_per_size[2]

        if config.target_mean_tokens_per_image < minimum_budget:
            raise ValueError(
                f"Target token budget must be at least {minimum_budget}, "
                "the minimum possible token count."
            )

        percentiles = np.unique(
            np.append(
                np.arange(
                    0.0,
                    100.0,
                    config.search_step,
                ),
                100.0,
            )
        )

        thresholds = [
            float(np.percentile(sorted_ics, pct))
            for pct in percentiles
        ]

        feasible = []

        for i, low_pct in enumerate(percentiles):
            low_threshold = thresholds[i]

            for j in range(i + 1, len(percentiles)):
                high_pct = percentiles[j]
                high_threshold = thresholds[j]

                mean_tokens = _mean_tokens_from_sorted_ics(
                    sorted_ics,
                    num_images,
                    low_threshold,
                    high_threshold,
                    tokens_per_size,
                )

                if mean_tokens <= config.target_mean_tokens_per_image:
                    feasible.append(
                        (
                            mean_tokens,
                            float(low_pct),
                            float(high_pct),
                            low_threshold,
                            high_threshold,
                        )
                    )

        if not feasible:
            raise ValueError(
                "No global percentile pair meets the target token budget. "
                "Raise the target or use a smaller search_step."
            )

        def _selection_key(item: tuple[float, float, float, float, float]):
            mean_tok, low_p, high_p, _, _ = item
            dist = (
                abs(low_p - config.low_percentile)
                + abs(high_p - config.high_percentile)
            )
            return (round(mean_tok, 6), -dist)

        (
            estimated_tokens,
            selected_low,
            selected_high,
            t_low,
            t_high,
        ) = max(feasible, key=_selection_key)

    return ICSCalibration(
        alpha=float(alpha),
        beta=float(beta),
        t_low=float(t_low),
        t_high=float(t_high),
        low_percentile=float(selected_low),
        high_percentile=float(selected_high),
        estimated_tokens_per_image=float(estimated_tokens),
    )


def determine_patch_size_map(
    ics_map: np.ndarray,
    calibration: ICSCalibration,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> np.ndarray:
    """
    Map each ICS region to an adaptive patch size.

    The patch_sizes tuple is ordered as (small, medium, large).
    """
    ics_map = _validate_ics_map(ics_map)
    calibration = _validate_calibration(calibration)
    patch_sizes = _validate_patch_sizes(patch_sizes)

    small_patch, medium_patch, large_patch = patch_sizes

    size_map = np.full(
        ics_map.shape,
        large_patch,
        dtype=np.int32,
    )

    medium_mask = (
        (ics_map >= calibration.t_low)
        & (ics_map < calibration.t_high)
    )

    small_mask = ics_map >= calibration.t_high

    size_map[medium_mask] = medium_patch
    size_map[small_mask] = small_patch

    return size_map


def compute_patch_size_map_for_image(
    image: np.ndarray,
    calibration: ICSCalibration,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> np.ndarray:
    """Compute an image's ICS map and convert it to an adaptive patch map."""
    calibration = _validate_calibration(calibration)

    variance_map, gradient_map = _compute_variance_and_gradient(
        image,
        region_size,
    )

    ics_map = compute_ics(
        variance_map,
        gradient_map,
        alpha=calibration.alpha,
        beta=calibration.beta,
    )

    return determine_patch_size_map(
        ics_map,
        calibration=calibration,
        patch_sizes=patch_sizes,
    )


def determine_budget_patch_size_map(
    ics_map: np.ndarray,
    num_small: int,
    num_medium: int,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> np.ndarray:
    """
    Assign patch sizes using a fixed number of regions per size.

    Regions with the highest ICS receive the smallest patches first.
    """
    ics_map = _validate_ics_map(ics_map)
    patch_sizes = _validate_patch_sizes(patch_sizes)

    for name, value in (
        ("num_small", num_small),
        ("num_medium", num_medium),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must not be negative.")

    num_small = int(num_small)
    num_medium = int(num_medium)

    total = ics_map.size

    if num_small + num_medium > total:
        raise ValueError(
            "num_small + num_medium must not exceed the number of regions."
        )

    small_patch, medium_patch, large_patch = patch_sizes

    order = np.argsort(
        -ics_map.ravel(),
        kind="stable",
    )

    flat = np.full(
        total,
        large_patch,
        dtype=np.int32,
    )

    flat[order[:num_small]] = small_patch

    flat[
        order[num_small:num_small + num_medium]
    ] = medium_patch

    return flat.reshape(ics_map.shape)


def compute_budget_patch_size_map_for_image(
    image: np.ndarray,
    calibration: ICSCalibration,
    num_small: int,
    num_medium: int,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> np.ndarray:
    """Compute an image's ICS map and convert it to a fixed-budget patch map."""
    calibration = _validate_calibration(calibration)

    variance_map, gradient_map = _compute_variance_and_gradient(
        image,
        region_size,
    )

    ics_map = compute_ics(
        variance_map,
        gradient_map,
        alpha=calibration.alpha,
        beta=calibration.beta,
    )

    return determine_budget_patch_size_map(
        ics_map,
        num_small,
        num_medium,
        patch_sizes,
    )


def tier_token_count(
    num_small: int,
    num_medium: int,
    num_regions: int = 49,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> int:
    """Calculate token count produced by one tier."""
    region_size = _validate_region_size(region_size)
    patch_sizes = _validate_patch_sizes(patch_sizes, region_size)

    for name, value in (
        ("num_small", num_small),
        ("num_medium", num_medium),
        ("num_regions", num_regions),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must not be negative.")

    num_small = int(num_small)
    num_medium = int(num_medium)
    num_regions = int(num_regions)

    if num_regions == 0:
        raise ValueError("num_regions must be greater than 0.")

    if num_small + num_medium > num_regions:
        raise ValueError(
            "num_small + num_medium must not exceed num_regions."
        )

    small, medium, large = patch_sizes
    num_large = num_regions - num_small - num_medium

    return int(
        num_small * (region_size // small) ** 2
        + num_medium * (region_size // medium) ** 2
        + num_large * (region_size // large) ** 2
    )


def fit_detail_threshold(
    calibration_images,
    calibration: ICSCalibration,
    region_size: int = 32,
    percentile: float = 90.0,
) -> float:
    """
    Calculate the ICS threshold for identifying regions needing
    the smallest patch size.
    """
    calibration_images = _validate_calibration_images(calibration_images)
    calibration = _validate_calibration(calibration)
    region_size = _validate_region_size(region_size)

    if isinstance(percentile, bool) or not isinstance(percentile, Real):
        raise TypeError("percentile must be a numeric value.")

    if not math.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("percentile must be in the range (0, 100).")

    values = []

    for image in calibration_images:
        variance_map, gradient_map = _compute_variance_and_gradient(
            image,
            region_size,
        )
        ics_map = compute_ics(
            variance_map,
            gradient_map,
            alpha=calibration.alpha,
            beta=calibration.beta,
        )

        values.append(ics_map.ravel())

    return float(
        np.percentile(
            np.concatenate(values),
            percentile,
        )
    )


def _validate_tiers(
    tiers: Sequence[tuple[int, int]],
    num_regions: int,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> tuple[tuple[int, int], ...]:
    """Validate tier definitions and their token-budget ordering."""
    try:
        tiers = tuple(tiers)
    except TypeError as exc:
        raise TypeError("tiers must be a sequence of pairs.") from exc

    if not tiers:
        raise ValueError("tiers must not be empty.")

    region_size = _validate_region_size(region_size)
    patch_sizes = _validate_patch_sizes(patch_sizes, region_size)

    if isinstance(num_regions, bool) or not isinstance(
        num_regions, Integral
    ):
        raise TypeError("num_regions must be an integer.")

    if num_regions <= 0:
        raise ValueError("num_regions must be greater than 0.")

    num_regions = int(num_regions)
    validated = []

    for tier in tiers:
        if not isinstance(tier, (tuple, list)) or len(tier) != 2:
            raise ValueError(
                "Each tier must contain exactly two values."
            )

        num_small, num_medium = tier

        for name, value in (
            ("num_small", num_small),
            ("num_medium", num_medium),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"Tier {name} must be an integer.")

            if value < 0:
                raise ValueError(f"Tier {name} must not be negative.")

        num_small = int(num_small)
        num_medium = int(num_medium)

        if num_small + num_medium > num_regions:
            raise ValueError(
                "Each tier must satisfy num_small + num_medium "
                "<= number of regions."
            )

        validated.append((num_small, num_medium))

    if any(
        validated[index][0] >= validated[index + 1][0]
        for index in range(len(validated) - 1)
    ):
        raise ValueError(
            "tiers must be sorted by strictly increasing num_small."
        )

    token_counts = [
        tier_token_count(
            num_small,
            num_medium,
            num_regions=num_regions,
            region_size=region_size,
            patch_sizes=patch_sizes,
        )
        for num_small, num_medium in validated
    ]

    if any(
        token_counts[index] >= token_counts[index + 1]
        for index in range(len(token_counts) - 1)
    ):
        raise ValueError(
            "Each subsequent tier must have a strictly larger "
            "token budget than the previous tier."
        )

    return tuple(validated)


def determine_tiered_patch_size_map(
    ics_map: np.ndarray,
    detail_threshold: float,
    tiers: Sequence[tuple[int, int]] = DEFAULT_TIERS,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
    region_size: int = 32,
) -> tuple[np.ndarray, int]:
    """
    Assign an image to a fixed token-budget tier.

    The number of regions at or above detail_threshold determines
    the selected tier. Within that tier, regions are ranked by ICS.

    Returns:
        A tuple containing the patch size map and selected tier index.
    """
    ics_map = _validate_ics_map(ics_map)
    patch_sizes = _validate_patch_sizes(patch_sizes, region_size)

    if isinstance(detail_threshold, bool) or not isinstance(
        detail_threshold, Real
    ):
        raise TypeError("detail_threshold must be a numeric value.")

    if not math.isfinite(detail_threshold):
        raise ValueError("detail_threshold must be finite.")

    tiers = _validate_tiers(
        tiers,
        ics_map.size,
        region_size=region_size,
        patch_sizes=patch_sizes,
    )

    wanted = int(
        np.count_nonzero(ics_map >= detail_threshold)
    )

    tier_index = 0

    for index, (num_small, _) in enumerate(tiers):
        if num_small <= wanted:
            tier_index = index

    num_small, num_medium = tiers[tier_index]

    patch_map = determine_budget_patch_size_map(
        ics_map,
        num_small,
        num_medium,
        patch_sizes,
    )

    return patch_map, tier_index


def compute_tiered_patch_size_map_for_image(
    image: np.ndarray,
    calibration: ICSCalibration,
    detail_threshold: float,
    tiers: Sequence[tuple[int, int]] = DEFAULT_TIERS,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES,
) -> tuple[np.ndarray, int]:
    """Compute an image's ICS map and convert it to a tiered patch map."""
    calibration = _validate_calibration(calibration)

    variance_map, gradient_map = _compute_variance_and_gradient(
        image,
        region_size,
    )

    ics_map = compute_ics(
        variance_map,
        gradient_map,
        alpha=calibration.alpha,
        beta=calibration.beta,
    )

    return determine_tiered_patch_size_map(
        ics_map,
        detail_threshold,
        tiers,
        patch_sizes,
        region_size,
    )