
"""Batch sampler that groups images with identical token signatures."""

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class SignatureBatchSampler(Sampler[list[int]]):
    """
    Yield batches of dataset indices with identical token signatures.

    Images with identical signatures have the same token counts
    across adaptive patch branches, allowing them to be grouped
    together and reducing unnecessary padding during batching.

    Use as DataLoader(batch_sampler=...).

    Args:
        signatures: Token signature of every dataset item.
        batch_size: Maximum batch size.
        shuffle: Shuffle indices and batch order every epoch.
        weights: Optional per-item sampling weights (sampled with replacement).
        seed: Base seed; the epoch counter is added automatically.
    """

    def __init__(
        self,
        signatures: Sequence[Sequence[int]],
        batch_size: int,
        shuffle: bool = False,
        weights: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError("batch_size must be an integer.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if weights is not None and not shuffle:
            raise ValueError("weights can only be used with shuffle=True.")

        raw_signatures = np.asarray(signatures)

        if raw_signatures.ndim != 2:
            raise ValueError("signatures must be a two-dimensional sequence.")

        if not np.issubdtype(raw_signatures.dtype, np.number):
            raise ValueError("signatures must contain numeric values.")

        if not np.isfinite(raw_signatures).all():
            raise ValueError("signatures must contain finite values.")

        if (raw_signatures < 0).any():
            raise ValueError("signature elements must be non-negative.")

        if not np.equal(raw_signatures, np.floor(raw_signatures)).all():
            raise ValueError("signature elements must be integers.")

        if (raw_signatures > np.iinfo(np.int64).max).any():
            raise ValueError("signature elements exceed the int64 range.")

        self.signatures: np.ndarray = raw_signatures.astype(np.int64)

        self.batch_size = int(batch_size)

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")

        self.shuffle = shuffle

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        
        self.seed = int(seed)
        self.epoch = 0

        # Explicitly declare the attribute type.
        self.weights: np.ndarray | None = None

        if weights is not None:
            weight_array = np.asarray(weights, dtype=np.float64)

            if weight_array.ndim != 1:
                raise ValueError("weights must be one-dimensional.")

            if weight_array.shape != (self.signatures.shape[0],):
                raise ValueError(
                    "weights must match signatures in length."
                )

            if not np.isfinite(weight_array).all():
                raise ValueError("weights must contain finite values.")

            if (weight_array < 0).any() or weight_array.sum() <= 0:
                raise ValueError(
                    "weights must be non-negative and have a positive sum."
                )

            max_weight = weight_array.max()

            if max_weight <= 0:
                raise ValueError(
                    "weights must be non-negative and have a positive sum."
                )

            weight_array = weight_array / max_weight
            self.weights = weight_array / weight_array.sum()

    def _indices(self, rng: np.random.Generator) -> np.ndarray:
        count = len(self.signatures)

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

        selected_signatures = self.signatures[indices]

        unique_signatures = np.unique(
            selected_signatures,
            axis=0,
        )

        for signature in unique_signatures:
            mask = np.all(selected_signatures == signature, axis=1)
            members = indices[mask]

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
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = self._indices(rng)

        if len(indices) == 0:
            return 0

        selected_signatures = self.signatures[indices]

        _, counts = np.unique(
            selected_signatures,
            axis=0,
            return_counts=True,
        )

        return int(
            sum(
                (int(count) + self.batch_size - 1) // self.batch_size
                for count in counts
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
