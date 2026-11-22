"""Batch samplers that group images by adaptive token signatures."""

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def _validate_signature_matrix(
    values: Sequence[Sequence[int]],
    name: str,
) -> np.ndarray:
    """Validate and convert token signatures to int64."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a rectangular two-dimensional sequence."
        ) from exc

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional sequence."
        )

    if array.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one branch."
        )

    if array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(
            f"{name} must contain numeric integer values."
        )

    if array.dtype.kind == "f":
        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} must contain finite values."
            )

        if (array < 0).any():
            raise ValueError(
                f"{name} must contain non-negative values."
            )

        if not np.equal(array, np.floor(array)).all():
            raise ValueError(
                f"{name} must contain integer values."
            )

        if (array >= 2**63).any():
            raise ValueError(
                f"{name} elements exceed the int64 range."
            )

    else:
        if (array < 0).any():
            raise ValueError(
                f"{name} must contain non-negative values."
            )

        if array.dtype.kind == "u":
            if (array > np.iinfo(np.int64).max).any():
                raise ValueError(
                    f"{name} elements exceed the int64 range."
                )

    return array.astype(np.int64)


def _validate_weights(
    weights: Sequence[float] | None,
    sample_count: int,
) -> np.ndarray | None:
    """Validate and normalize optional sampling weights."""

    if weights is None:
        return None

    try:
        weight_array = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "weights must contain valid numeric values."
        ) from exc

    if weight_array.ndim != 1:
        raise ValueError("weights must be one-dimensional.")

    if weight_array.shape != (sample_count,):
        raise ValueError(
            "weights must match signatures in length."
        )

    if weight_array.size == 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    if not np.isfinite(weight_array).all():
        raise ValueError(
            "weights must contain finite values."
        )

    if (weight_array < 0).any():
        raise ValueError(
            "weights must be non-negative."
        )

    max_weight = weight_array.max()

    if max_weight <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    # Scale first to avoid overflow when summing large weights.
    weight_array = weight_array / max_weight
    weight_array = weight_array / weight_array.sum()

    return weight_array


def _validate_batch_size(batch_size: int) -> int:
    """Validate batch size."""

    if isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    return int(batch_size)


def _validate_shuffle(shuffle: bool) -> None:
    """Validate shuffle flag."""

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")


def _validate_seed(seed: int) -> int:
    """Validate random seed."""

    if isinstance(seed, bool) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    return int(seed)


def _group_indices_by_signature(
    indices: np.ndarray,
    selected_signatures: np.ndarray,
) -> dict[tuple[int, ...], list[int]]:
    """Group dataset indices by signature tuple in O(N) time."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for index, sig_row in zip(
        indices.tolist(),
        selected_signatures.tolist(),
    ):
        groups[tuple(sig_row)].append(index)

    return groups


class SignatureBatchSampler(Sampler[list[int]]):
    """
    Yield batches of dataset indices with identical token signatures.

    Images with identical signatures have the same token counts
    across adaptive patch branches. Grouping them reduces unnecessary
    padding during batching.

    Use as DataLoader(batch_sampler=...).

    Args:
        signatures: Token signature of every dataset item.
        batch_size: Maximum number of samples per batch.
        shuffle: Shuffle indices and batch order every epoch.
        weights: Optional per-item sampling weights. When provided,
            samples are drawn with replacement.
        seed: Base random seed. The epoch counter is added automatically.
    """

    def __init__(
        self,
        signatures: Sequence[Sequence[int]],
        batch_size: int,
        shuffle: bool = False,
        weights: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.signatures = _validate_signature_matrix(
            signatures,
            "signatures",
        )

        self.batch_size = _validate_batch_size(batch_size)

        _validate_shuffle(shuffle)
        self.shuffle = shuffle

        if weights is not None and not shuffle:
            raise ValueError(
                "weights can only be used with shuffle=True."
            )

        self.seed = _validate_seed(seed)
        self.epoch = 0

        self.weights = _validate_weights(
            weights,
            len(self.signatures),
        )

        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] | None = None

    def _indices(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate dataset indices for the current epoch."""

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
        self,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Group selected indices by identical token signatures in O(N)."""

        indices = self._indices(rng)
        batches: list[list[int]] = []

        if len(indices) == 0:
            return batches

        selected_signatures = self.signatures[indices]
        groups = _group_indices_by_signature(
            indices,
            selected_signatures,
        )

        for signature in sorted(groups):
            members = groups[signature]

            for start in range(
                0,
                len(members),
                self.batch_size,
            ):
                batches.append(
                    members[start : start + self.batch_size]
                )

        if self.shuffle and batches:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]

        return batches

    def _get_batches_for_epoch(self, epoch: int) -> list[list[int]]:
        """Return cached batches for the given epoch, computing them once."""
        if self._cached_epoch != epoch or self._cached_batches is None:
            rng = np.random.default_rng(self.seed + epoch)
            self._cached_batches = self._batches(rng)
            self._cached_epoch = epoch

        return self._cached_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches for the current epoch."""

        batches = self._get_batches_for_epoch(self.epoch)

        if self.shuffle:
            self.epoch += 1

        yield from batches

    def __len__(self) -> int:
        """Return the number of batches for the current epoch seed."""

        return len(self._get_batches_for_epoch(self.epoch))

    def eval_order(self) -> np.ndarray:
        """
        Return dataset indices in the order an evaluation loader returns them.

        Only available when shuffle=False.
        """

        if self.shuffle:
            raise ValueError(
                "eval_order is only defined for shuffle=False."
            )

        batches = self._batches(
            np.random.default_rng(0)
        )

        if not batches:
            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.concatenate(
            [
                np.asarray(batch, dtype=np.int64)
                for batch in batches
            ]
        )


class ApproximateSignatureBatchSampler(Sampler[list[int]]):
    """
    Yield batches with approximately similar token signatures.

    Samples are first grouped by coarse signatures. Within each
    coarse group, samples are sorted by their exact signatures and
    grouped while respecting the maximum token difference per branch.

    Use as DataLoader(batch_sampler=...).

    Args:
        signatures: Exact token signature of every dataset item.
        coarse_signatures: Coarse token signature of every dataset item.
        batch_size: Maximum number of samples per batch.
        max_token_delta: Maximum allowed token-count difference
            per branch within a batch. Can be a single integer or
            a sequence containing one value per branch.
        shuffle: Shuffle indices and batch order every epoch.
        weights: Optional per-item sampling weights. When provided,
            samples are drawn with replacement.
        seed: Base random seed. The epoch counter is added automatically.
    """

    def __init__(
        self,
        signatures: Sequence[Sequence[int]],
        coarse_signatures: Sequence[Sequence[int]],
        batch_size: int,
        max_token_delta: int | Sequence[int],
        shuffle: bool = False,
        weights: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.signatures = _validate_signature_matrix(
            signatures,
            "signatures",
        )

        self.coarse_signatures = _validate_signature_matrix(
            coarse_signatures,
            "coarse_signatures",
        )

        if self.signatures.shape != self.coarse_signatures.shape:
            raise ValueError(
                "signatures and coarse_signatures must have the same shape."
            )

        self.batch_size = _validate_batch_size(batch_size)

        _validate_shuffle(shuffle)
        self.shuffle = shuffle

        if weights is not None and not shuffle:
            raise ValueError(
                "weights can only be used with shuffle=True."
            )

        self.max_token_delta = self._validate_max_token_delta(
            max_token_delta,
            self.signatures.shape[1],
        )

        self.seed = _validate_seed(seed)
        self.epoch = 0

        self.weights = _validate_weights(
            weights,
            len(self.signatures),
        )

        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] | None = None

    @staticmethod
    def _validate_max_token_delta(
        value: int | Sequence[int],
        num_branches: int,
    ) -> np.ndarray:
        """Validate maximum token difference per branch."""

        if isinstance(value, (int, np.integer)):
            if isinstance(value, bool):
                raise TypeError(
                    "max_token_delta must contain integer values."
                )

            if value < 0:
                raise ValueError(
                    "max_token_delta must be non-negative."
                )

            if value > np.iinfo(np.int64).max:
                raise ValueError(
                    "max_token_delta exceeds the int64 range."
                )

            return np.full(
                num_branches,
                int(value),
                dtype=np.int64,
            )

        try:
            delta_array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_token_delta must be a scalar or one-dimensional sequence."
            ) from exc

        if delta_array.ndim != 1:
            raise ValueError(
                "max_token_delta must be a scalar or one-dimensional sequence."
            )

        if delta_array.shape[0] != num_branches:
            raise ValueError(
                "max_token_delta must contain one value per branch."
            )

        if delta_array.dtype.kind not in ("i", "u", "f"):
            raise ValueError(
                "max_token_delta must contain numeric integer values."
            )

        if delta_array.dtype.kind == "f":
            if not np.isfinite(delta_array).all():
                raise ValueError(
                    "max_token_delta must contain finite values."
                )

            if (delta_array < 0).any():
                raise ValueError(
                    "max_token_delta must be non-negative."
                )

            if not np.equal(
                delta_array,
                np.floor(delta_array),
            ).all():
                raise ValueError(
                    "max_token_delta must contain integer values."
                )

            if (delta_array >= 2**63).any():
                raise ValueError(
                    "max_token_delta exceeds the int64 range."
                )

        else:
            if (delta_array < 0).any():
                raise ValueError(
                    "max_token_delta must be non-negative."
                )

            if delta_array.dtype.kind == "u":
                if (
                    delta_array > np.iinfo(np.int64).max
                ).any():
                    raise ValueError(
                        "max_token_delta exceeds the int64 range."
                    )

        return delta_array.astype(np.int64)

    def _indices(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate dataset indices for the current epoch."""

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
        self,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Build batches while respecting token-difference limits."""

        indices = self._indices(rng)
        batches: list[list[int]] = []

        if len(indices) == 0:
            return batches

        selected_coarse = self.coarse_signatures[indices]
        coarse_groups = _group_indices_by_signature(
            indices,
            selected_coarse,
        )

        for coarse_signature in sorted(coarse_groups):
            members = np.asarray(
                coarse_groups[coarse_signature],
                dtype=np.int64,
            )
            exact = self.signatures[members]

            # Sort samples lexicographically by exact signatures.
            sort_order = np.lexsort(
                tuple(
                    exact[:, branch]
                    for branch in reversed(
                        range(exact.shape[1])
                    )
                )
            )

            members = members[sort_order]
            exact = exact[sort_order]

            current_batch: list[int] = []
            batch_min: np.ndarray | None = None
            batch_max: np.ndarray | None = None

            for index, signature in zip(
                members,
                exact,
            ):
                if not current_batch:
                    current_batch = [int(index)]
                    batch_min = signature.copy()
                    batch_max = signature.copy()
                    continue

                assert batch_min is not None
                assert batch_max is not None

                new_min = np.minimum(
                    batch_min,
                    signature,
                )

                new_max = np.maximum(
                    batch_max,
                    signature,
                )

                within_tolerance = np.all(
                    new_max - new_min <= self.max_token_delta
                )

                has_capacity = (
                    len(current_batch) < self.batch_size
                )

                if within_tolerance and has_capacity:
                    current_batch.append(int(index))
                    batch_min = new_min
                    batch_max = new_max
                else:
                    batches.append(current_batch)

                    current_batch = [int(index)]
                    batch_min = signature.copy()
                    batch_max = signature.copy()

            if current_batch:
                batches.append(current_batch)

        if self.shuffle and batches:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]

        return batches

    def _get_batches_for_epoch(self, epoch: int) -> list[list[int]]:
        """Return cached batches for the given epoch, computing them once."""
        if self._cached_epoch != epoch or self._cached_batches is None:
            rng = np.random.default_rng(self.seed + epoch)
            self._cached_batches = self._batches(rng)
            self._cached_epoch = epoch

        return self._cached_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches for the current epoch."""

        batches = self._get_batches_for_epoch(self.epoch)

        if self.shuffle:
            self.epoch += 1

        yield from batches

    def __len__(self) -> int:
        """Return the number of batches for the current epoch seed."""

        return len(self._get_batches_for_epoch(self.epoch))

    def eval_order(self) -> np.ndarray:
        """
        Return dataset indices in the order an evaluation loader returns them.

        Only available when shuffle=False.
        """

        if self.shuffle:
            raise ValueError(
                "eval_order is only defined for shuffle=False."
            )

        batches = self._batches(
            np.random.default_rng(0)
        )

        if not batches:
            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.concatenate(
            [
                np.asarray(batch, dtype=np.int64)
                for batch in batches
            ]
        )