"""Configuration objects for the DP-ViT Library."""

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class DPViTConfig:
    """Architecture Dynamic Patch Vision Transformer"""

    num_classes: int

    image_size: int = 224
    region_size: int = 32
    patch_sizes: tuple[int, int, int] = (8, 16, 32)

    embed_dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be bigger than 0")

        if self.image_size <= 0:
            raise ValueError("image_size must be bigger than 0")

        if self.region_size <= 0:
            raise ValueError("region_size must be bigger than 0")

        if self.image_size % self.region_size !=0:
            raise ValueError("image_size must be divisible by region_size")

        if self.patch_sizes != (8, 16, 32):
            raise ValueError("patch_size must be (8, 16, 32)")

        if any(self.region_size % size != 0 for size in self.patch_sizes):
            raise ValueError("region_size must be divisible by all patch sizes")

        if self.embed_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("embed_dim, num_layers, and num_heads must be bigger than 0")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive number")

        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in range [0, 1].")

    @property
    def max_position_cell(self) -> int:
        """Maximum number of position cells."""
        smallest_patch = min(self.patch_sizes)
        return (self.image_size // smallest_patch) ** 2

@dataclass(frozen=True)
class ICSConfig:
    """Configuration for global ICS thresholds and an optional token budget."""

    low_percentile: float = 30.0
    high_percentile: float = 70.0

    # None preserves the existing fixed-percentile behavior.
    target_mean_tokens_per_image: float | None = None
    search_step: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.low_percentile <= 100:
            raise ValueError("low_percentile must be in range [0, 100].")
    
        if not 0 <= self.high_percentile <= 100:
            raise ValueError("high_percentile must be in range [0, 100].")
    
        if self.low_percentile >= self.high_percentile:
            raise ValueError("low_percentile must be smaller than high_percentile.")
    
        if self.target_mean_tokens_per_image is not None:
            if (
                not math.isfinite(self.target_mean_tokens_per_image)
                or self.target_mean_tokens_per_image <= 0
            ):
                raise ValueError(
                    "target_mean_tokens_per_image must be finite and positive."
                )
    
        if not math.isfinite(self.search_step) or not 0 < self.search_step <= 50:
            raise ValueError("search_step must be finite and in the range (0, 50].")

@dataclass(frozen=True)
class ICSCalibration:
    """Calibration ICS value used for mapping patch."""

    alpha: float
    beta: float
    p30: float
    p70: float
    low_percentile: float = 30.0
    high_percentile: float = 70.0
    estimated_tokens_per_image: float | None = None

    def __post_init__ (self) -> None:
        for name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("p30", self.p30),
            ("p70", self.p70),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite number"
        )

        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be greater than 0.")

        if self.p30 > self.p70:
            raise ValueError("p30 must be smaller than p70.")

@dataclass(frozen=True)
class PruningConfig:
    """Configuration for neuron MLP structured magnitude-based pruning."""

    target_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_ratio):
            raise ValueError("target_ratio must be finite number")

        if not 0 < self.target_ratio < 1:
            raise ValueError("target_ratio must be greater than 0 and smaller than 1.")