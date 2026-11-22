"""Configuration objects for the DP-ViT Library."""

from dataclasses import dataclass
import math
from numbers import Integral, Real


DEFAULT_PATCH_SIZES = (8, 16, 32)


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a positive integer, excluding boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_finite_number(name: str, value: float) -> None:
    """Validate a finite numeric value, excluding boolean values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value.")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_patch_sizes(
    patch_sizes: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Validate and normalize the supported patch sizes."""
    if not isinstance(patch_sizes, (tuple, list)):
        raise TypeError("patch_sizes must be a tuple or list.")

    if len(patch_sizes) != 3:
        raise ValueError(
            "patch_sizes must contain exactly three values."
        )

    if any(
        isinstance(size, bool) or not isinstance(size, Integral)
        for size in patch_sizes
    ):
        raise TypeError("All patch sizes must be integers.")

    normalized = (
        int(patch_sizes[0]),
        int(patch_sizes[1]),
        int(patch_sizes[2]),
    )

    if normalized != DEFAULT_PATCH_SIZES:
        raise ValueError(
            f"patch_sizes must be exactly {DEFAULT_PATCH_SIZES}."
        )

    return normalized


@dataclass(frozen=True)
class DPViTConfig:
    """Configuration for the Dynamic Patch Vision Transformer."""

    num_classes: int

    image_size: int = 224
    region_size: int = 32
    patch_sizes: tuple[int, int, int] = DEFAULT_PATCH_SIZES

    embed_dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive_int("num_classes", self.num_classes)
        _validate_positive_int("image_size", self.image_size)
        _validate_positive_int("region_size", self.region_size)
        _validate_positive_int("embed_dim", self.embed_dim)
        _validate_positive_int("num_layers", self.num_layers)
        _validate_positive_int("num_heads", self.num_heads)

        patch_sizes = _validate_patch_sizes(self.patch_sizes)

        object.__setattr__(
            self,
            "patch_sizes",
            patch_sizes,
        )

        if self.image_size % self.region_size != 0:
            raise ValueError(
                "image_size must be divisible by region_size."
            )

        if any(
            self.region_size % size != 0
            for size in patch_sizes
        ):
            raise ValueError(
                "region_size must be divisible by all patch sizes."
            )

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        _validate_finite_number("mlp_ratio", self.mlp_ratio)
        _validate_finite_number("dropout", self.dropout)

        if self.mlp_ratio <= 0:
            raise ValueError(
                "mlp_ratio must be greater than 0."
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in range [0, 1)."
            )

    @property
    def max_position_cell(self) -> int:
        """Maximum number of positional cells."""
        smallest_patch = min(self.patch_sizes)

        return (
            self.image_size // smallest_patch
        ) ** 2


@dataclass(frozen=True)
class ICSConfig:
    """Configuration for ICS thresholds and optional token budget."""

    low_percentile: float = 30.0
    high_percentile: float = 70.0

    target_mean_tokens_per_image: float | None = None
    search_step: float = 5.0

    def __post_init__(self) -> None:
        _validate_finite_number(
            "low_percentile",
            self.low_percentile,
        )

        _validate_finite_number(
            "high_percentile",
            self.high_percentile,
        )

        _validate_finite_number(
            "search_step",
            self.search_step,
        )

        if not 0 <= self.low_percentile <= 100:
            raise ValueError(
                "low_percentile must be in range [0, 100]."
            )

        if not 0 <= self.high_percentile <= 100:
            raise ValueError(
                "high_percentile must be in range [0, 100]."
            )

        if self.low_percentile >= self.high_percentile:
            raise ValueError(
                "low_percentile must be smaller than high_percentile."
            )

        if self.target_mean_tokens_per_image is not None:
            _validate_finite_number(
                "target_mean_tokens_per_image",
                self.target_mean_tokens_per_image,
            )

            if self.target_mean_tokens_per_image <= 0:
                raise ValueError(
                    "target_mean_tokens_per_image must be greater than 0."
                )

        if not 0 < self.search_step <= 50:
            raise ValueError(
                "search_step must be in range (0, 50]."
            )


@dataclass(frozen=True)
class ICSCalibration:
    """Calibration values used for adaptive patch mapping."""

    alpha: float
    beta: float
    t_low: float
    t_high: float

    low_percentile: float = 30.0
    high_percentile: float = 70.0

    estimated_tokens_per_image: float | None = None
    detail_threshold: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("t_low", self.t_low),
            ("t_high", self.t_high),
            ("low_percentile", self.low_percentile),
            ("high_percentile", self.high_percentile),
        ):
            _validate_finite_number(name, value)

        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                "alpha and beta must be greater than 0."
            )

        if self.t_low > self.t_high:
            raise ValueError(
                "t_low must be less than or equal to t_high."
            )

        if not 0 <= self.low_percentile <= 100:
            raise ValueError(
                "low_percentile must be in range [0, 100]."
            )

        if not 0 <= self.high_percentile <= 100:
            raise ValueError(
                "high_percentile must be in range [0, 100]."
            )

        if self.low_percentile >= self.high_percentile:
            raise ValueError(
                "low_percentile must be smaller than high_percentile."
            )

        if self.estimated_tokens_per_image is not None:
            _validate_finite_number(
                "estimated_tokens_per_image",
                self.estimated_tokens_per_image,
            )

            if self.estimated_tokens_per_image <= 0:
                raise ValueError(
                    "estimated_tokens_per_image must be greater than 0."
                )

        if self.detail_threshold is not None:
            _validate_finite_number(
                "detail_threshold",
                self.detail_threshold,
            )


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for structured magnitude-based pruning."""

    target_ratio: float = 0.2

    def __post_init__(self) -> None:
        _validate_finite_number(
            "target_ratio",
            self.target_ratio,
        )

        if not 0.0 <= self.target_ratio < 1.0:
            raise ValueError(
                "target_ratio must be in range [0, 1)."
            )