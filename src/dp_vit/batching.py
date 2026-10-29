
"""Batch sampler that groups images of the same token tier (no padding)."""

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class TierBatchSampler(Sampler[list[int]]):
    """
    Yield batches of dataset indices in which every image has the same tier.

    Images with the same tier have the same token count, so the encoder
    sequence needs no padding. Use as DataLoader(batch_sampler=...).

    Args:
        tiers: Tier index of every dataset item.
        batch_size: Maximum batch size.
        shuffle: Shuffle indices and batch order every epoch.
        weights: Optional per-item sampling weights (sampled with replacement).
        seed: Base seed; the epoch counter is added automatically.
    """

    def __init__(
        self,
        tiers: Sequence[int],
        batch_size: int,
        shuffle: bool = False,
        weights: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if weights is not None and not shuffle:
            raise ValueError("weights can only be used with shuffle=True.")

        self.tiers: np.ndarray = np.asarray(tiers, dtype=np.int64)

        if self.tiers.ndim != 1:
            raise ValueError("tiers must be a one-dimensional sequence.")

        if (self.tiers < 0).any():
            raise ValueError("tier indices must be non-negative.")

        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

        # Explicitly declare the attribute type.
        self.weights: np.ndarray | None = None

        if weights is not None:
            weight_array = np.asarray(weights, dtype=np.float64)

            if weight_array.ndim != 1:
                raise ValueError("weights must be one-dimensional.")

            if weight_array.shape != self.tiers.shape:
                raise ValueError(
                    "weights must match tiers in length."
                )

            if not np.isfinite(weight_array).all():
                raise ValueError("weights must contain finite values.")

            if (weight_array < 0).any() or weight_array.sum() <= 0:
                raise ValueError(
                    "weights must be non-negative and have a positive sum."
                )

            self.weights = weight_array / weight_array.sum()

    def _indices(self, rng: np.random.Generator) -> np.ndarray:
        count = len(self.tiers)

        if not self.shuffle:
            return np.arange(count, dtype=np.int64)

        if self.weights is not None:
            return rng.choice(
                count,
                size=count,
                replace=True,
                p=self.weights,
            )

        return rng.permutation(count)

    def _batches(
        self, rng: np.random.Generator
    ) -> list[list[int]]:
        indices = self._indices(rng)
        batches: list[list[int]] = []

        if len(indices) == 0:
            return batches

        selected_tiers = self.tiers[indices]

        for tier in np.unique(selected_tiers):
            members = indices[selected_tiers == tier]

            for start in range(0, len(members), self.batch_size):
                batch = members[start : start + self.batch_size]
                batches.append(batch.tolist())

        if self.shuffle and batches:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.shuffle:
            self.epoch += 1

        yield from self._batches(rng)

    def __len__(self) -> int:
        if len(self.tiers) == 0:
            return 0

        counts = np.bincount(self.tiers)

        return int(
            sum(
                (int(count) + self.batch_size - 1) // self.batch_size
                for count in counts
                if count > 0
            )
        )

    def eval_order(self) -> np.ndarray:
        """Dataset indices in the order an evaluation loader returns them."""
        if self.shuffle:
            raise ValueError(
                "eval_order is only defined for shuffle=False."
            )

        batches = self._batches(np.random.default_rng(0))

        if not batches:
            return np.empty(0, dtype=np.int64)

        return np.concatenate(
            [np.asarray(batch, dtype=np.int64) for batch in batches]
        )
