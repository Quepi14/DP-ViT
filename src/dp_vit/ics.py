"""INformation Complexity Score (ICS) calculations for DP-ViT."""

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import sobel

from .config import ICSCalibration, ICSConfig

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB image to a 2D float64 intensity array."""
    image = np.asarray(image)

    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = image.astype(np.float64)
        gray = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        )
    else:
        raise ValueError("image must have shape (height, width) or (height, width, 3).")

    if not np.isfinite(gray).all():
        raise ValueError("image must contain only finite values.")

    return gray

def _make_region_blocks(
    image: np.ndarray,
    region_size: int,
) -> np.ndarray:
    """Split a 2D image into non-overlapping square regions."""
    if isinstance(region_size, bool) or not isinstance(region_size, int):
        raise TypeError("region_size must be an integer.")

    if region_size <= 0:
        raise ValueError("region_size must be greater than zero.")

    height, width = image.shape

    if height % region_size != 0 or width % region_size != 0:
        raise ValueError("Image height and width must be divisible by region_size.")

    rows = height // region_size
    cols = width // region_size

    blocks = image.reshape(
        rows, region_size, cols, region_size
    )
    blocks = blocks.transpose(0, 2, 1, 3)

    return blocks.reshape(rows, cols, region_size * region_size)

def compute_local_variance(
    image: np.ndarray,
    region_size: int = 32,
) -> np.ndarray:
    """
    Compute the population variance of grayscale intensity in each region.
    
    The variance for each region is
        Var(R) = mean((pixel - region_mean) ** 2)
    """
    gray = _to_grayscale(image)
    blocks = _make_region_blocks(gray, region_size)

    region_means = blocks.mean(axis=-1, keepdims=True)
    variance_map = np.mean((blocks - region_means) ** 2, axis=-1)

    return variance_map

def compute_gradient_magnitude(
    image: np.ndarray,
    region_size: int = 32,
) -> np.ndarray:
    """ 
    Compute the mean Sobel gradient magnitude in each region.

    Sobel gradients are calculated on full image before region averaging.
    """
    gray = _to_grayscale(image)

    gradient_x = sobel(gray, axis=1)
    gradient_y = sobel(gray, axis=0)
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    blocks = _make_region_blocks(magnitude, region_size)
    return blocks.mean(axis=-1)

def compute_ics(
    variance_map: np.ndarray,
    gradient_map: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Combine local variance and gradient magnitude into ICS map."""
    variance_map = np.asarray(variance_map, dtype=np.float64)
    gradient_map = np.asarray(gradient_map, dtype=np.float64)

    if variance_map.shape != gradient_map.shape:
        raise ValueError("variance_map and gradient_map must have the same shape.")

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number.")

    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number.")

    return alpha * variance_map + beta * gradient_map

def count_patch_tokens(
    patch_size_map: np.ndarray,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> int:
    "Count the actual patch tokens represented by a region-level patch map."
    patch_size_map = np.asarray(patch_size_map)

    if patch_size_map.ndim != 2:
        raise ValueError("patch_size_Map must be a 2D array.")

    if len(patch_sizes) != 3:
        raise ValueError("patch_sizes must contain exactly three values.")

    if any(region_size % size != 0 for size in patch_sizes):
        raise ValueError("region_size must be divisible by every patch size.")

    if not np.isin(patch_size_map, patch_sizes).all():
        raise ValueError(f"patch_size_map values must be selected from {patch_sizes}.")

    return int(sum(
        np.count_nonzero(patch_size_map == size)
        * (region_size // size) ** 2
        for size in patch_sizes
    ))

def _mean_tokens_for_thresholds(
    ics_maps,
    low_threshold: float,
    high_threshold: float,
    region_size: int,
    patch_sizes: tuple[int, int, int],
) -> float:
    """Measure mean valid tokens per calibration image for a threshold pair."""
    calibration = ICSCalibration(
        alpha=1.0,
        beta=1.0,
        p30=low_threshold,
        p70=high_threshold,
    )

    total_tokens = 0

    for ics_map in ics_maps:
        patch_map = determine_patch_size_map(
            ics_map,
            calibration,
            patch_sizes=patch_sizes,
        )
        total_tokens += count_patch_tokens(
            patch_map,
            region_size=region_size,
            patch_sizes=patch_sizes,
        )

    return total_tokens / len(ics_maps)

def fit_ics_calibration(
    calibration_images,
    region_size: int = 32,
    config: ICSConfig | None = None,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> ICSCalibration:
    """Fit global ICS thresholds, optionally targeting a mean token budget."""
    if not calibration_images:
        raise ValueError("calibration_images must not be empty.")

    if config is None:
        config = ICSConfig()

    all_variances = []
    all_gradients = []

    for image in calibration_images:
        all_variances.append(
            compute_local_variance(image, region_size).ravel()
        )
        all_gradients.append(
            compute_gradient_magnitude(image, region_size).ravel()
        )

    all_variances = np.concatenate(all_variances)
    all_gradients = np.concatenate(all_gradients)

    epsilon = 1e-8
    alpha = 1.0 / (all_variances.std() + epsilon)
    beta = 1.0 / (all_gradients.std() + epsilon)

    all_ics = alpha * all_variances + beta * all_gradients

    # Retain one ICS map for each calibration image.
    ics_maps = []
    offset = 0

    for image in calibration_images:
        shape = compute_local_variance(image, region_size).shape
        count = int(np.prod(shape))

        ics_maps.append(
            all_ics[offset : offset + count].reshape(shape)
        )
        offset += count

    selected_low = config.low_percentile
    selected_high = config.high_percentile

    # Without a target, keep the original percentile behavior.
    if config.target_mean_tokens_per_image is None:
        p30 = float(np.percentile(all_ics, selected_low))
        p70 = float(np.percentile(all_ics, selected_high))
        estimated_tokens = _mean_tokens_for_thresholds(
            ics_maps,
            p30,
            p70,
            region_size,
            patch_sizes,
        )

    else:
        region_counts = {ics_map.size for ics_map in ics_maps}

        if len(region_counts) != 1:
            raise ValueError(
                "Calibration images must contain the same number of regions "
                "when selecting a token budget."
            )

        minimum_budget = next(iter(region_counts))

        if config.target_mean_tokens_per_image < minimum_budget:
            raise ValueError(
                f"Target token budget must be at least {minimum_budget}, "
                "the one-token-per-region minimum."
            )

        percentiles = np.arange(
            0.0,
            100.0 + config.search_step * 0.5,
            config.search_step,
        )
        percentiles[-1] = 100.0
        sorted_ics = np.sort(all_ics)
        feasible = []

        for low_pct in percentiles:
            low_threshold = float(np.percentile(sorted_ics, low_pct))

            for high_pct in percentiles:
                if high_pct <= low_pct:
                    continue

                high_threshold = float(np.percentile(sorted_ics, high_pct))

                mean_tokens = _mean_tokens_for_thresholds(
                    ics_maps,
                    low_threshold,
                    high_threshold,
                    region_size,
                    patch_sizes,
                )

                if mean_tokens <= config.target_mean_tokens_per_image:
                    feasible.append(
                        (
                            mean_tokens,
                            low_pct,
                            high_pct,
                            low_threshold,
                            high_threshold,
                        )
                    )

        if not feasible:
            raise ValueError(
                "No global percentile pair meets the target token budget. "
                "Raise the target or use a smaller search_step."
            )

        (
            estimated_tokens,
            selected_low,
            selected_high,
            p30,
            p70,
        ) = max(feasible, key=lambda item: item[0])

    return ICSCalibration(
        alpha=float(alpha),
        beta=float(beta),
        p30=float(p30),
        p70=float(p70),
        low_percentile=float(selected_low),
        high_percentile=float(selected_high),
        estimated_tokens_per_image=float(estimated_tokens),
    )

def determine_patch_size_map(
    ics_map: np.ndarray,
    calibration: ICSCalibration,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> np.ndarray:
    """
    Map each ICS region to a patch size.

    The patch_sizes tuple is ordewred as (small, medium, large).
    """
    if len(patch_sizes) != 3:
        raise ValueError("patch_sizes must contain exactly three values.")

    small_patch, medium_patch, large_patch = patch_sizes

    if not small_patch < medium_patch < large_patch:
        raise ValueError("patch_sizes must be ordered from smallest to largest.")

    size_map = np.full(ics_map.shape, large_patch, dtype=np.int32)

    medium_mask = (
        (ics_map >= calibration.p30)
        & (ics_map < calibration.p70)
    )

    small_mask = ics_map >= calibration.p70

    size_map[medium_mask] = medium_patch
    size_map[small_mask] = small_patch

    return size_map

def compute_patch_size_map_for_image(
    image: np.ndarray,
    calibration: ICSCalibration,
    region_size: int = 32,
    patch_sizes: tuple[int, int, int] = (8, 16, 32),
) -> np.ndarray:
    """
    Compute an iamge's ICS map and convert it to an adaptive patch map.
    """
    variance_map = compute_local_variance(image, region_size)
    gradient_map = compute_gradient_magnitude(image, region_size)

    ics_map = compute_ics(
        variance_map,
        gradient_map,
        alpha=calibration.alpha,
        beta=calibration.beta,
    )

    return determine_patch_size_map(
        ics_map,
        calibration=calibration,
        patch_sizes=patch_sizes
    )