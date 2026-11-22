"""Structured magnitude-based pruning for DP-ViT MLP layers."""

import math
from numbers import Real
from typing import cast

import torch
from torch import nn

from .config import PruningConfig
from .model import DPViT


def _get_encoder_layers(
    model: DPViT,
) -> list[nn.TransformerEncoderLayer]:
    """Return and validate DP-ViT Transformer encoder layers."""

    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT.")

    if not isinstance(model.encoder, nn.TransformerEncoder):
        raise TypeError("model.encoder must be a TransformerEncoder.")

    layers = cast(
        list[nn.TransformerEncoderLayer],
        list(model.encoder.layers),
    )

    if not layers:
        raise ValueError("The model must contain at least one encoder layer.")

    for index, layer in enumerate(layers):
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise TypeError(
                f"Encoder layer {index} must be a TransformerEncoderLayer."
            )

        if not isinstance(layer.linear1, nn.Linear):
            raise TypeError(
                f"Encoder layer {index} linear1 must be nn.Linear."
            )

        if not isinstance(layer.linear2, nn.Linear):
            raise TypeError(
                f"Encoder layer {index} linear2 must be nn.Linear."
            )

        if layer.linear1.in_features != model.config.embed_dim:
            raise ValueError(
                f"Encoder layer {index} has an invalid linear1 input dimension."
            )

        if layer.linear2.out_features != model.config.embed_dim:
            raise ValueError(
                f"Encoder layer {index} has an invalid linear2 output dimension."
            )

        if layer.linear1.out_features != layer.linear2.in_features:
            raise ValueError(
                f"Encoder layer {index} has incompatible MLP dimensions."
            )

        if layer.linear1.out_features < 1:
            raise ValueError(
                f"Encoder layer {index} must retain at least one MLP neuron."
            )

    return layers


def _prune_layer_neurons(
    layer: nn.TransformerEncoderLayer,
    keep_indices: torch.Tensor,
) -> None:
    """Physically remove selected neurons from an encoder layer's MLP."""

    old_linear1 = layer.linear1
    old_linear2 = layer.linear2

    if keep_indices.ndim != 1:
        raise ValueError("keep_indices must be a one-dimensional tensor.")

    keep_indices = keep_indices.to(
        device=old_linear1.weight.device,
        dtype=torch.long,
    )

    if keep_indices.numel() == 0:
        raise ValueError("At least one neuron must be retained.")

    if (
        torch.any(keep_indices < 0)
        or torch.any(keep_indices >= old_linear1.out_features)
    ):
        raise ValueError("keep_indices contains an out-of-range index.")

    if torch.unique(keep_indices).numel() != keep_indices.numel():
        raise ValueError("keep_indices must not contain duplicate indices.")

    new_width = int(keep_indices.numel())

    new_linear1 = nn.Linear(
        in_features=old_linear1.in_features,
        out_features=new_width,
        bias=old_linear1.bias is not None,
        device=old_linear1.weight.device,
        dtype=old_linear1.weight.dtype,
    )

    new_linear2 = nn.Linear(
        in_features=new_width,
        out_features=old_linear2.out_features,
        bias=old_linear2.bias is not None,
        device=old_linear2.weight.device,
        dtype=old_linear2.weight.dtype,
    )

    with torch.no_grad():
        new_linear1.weight.copy_(
            old_linear1.weight.index_select(0, keep_indices)
        )

        if old_linear1.bias is not None:
            new_linear1.bias.copy_(
                old_linear1.bias.index_select(0, keep_indices)
            )

        new_linear2.weight.copy_(
            old_linear2.weight.index_select(1, keep_indices)
        )

        if old_linear2.bias is not None:
            new_linear2.bias.copy_(old_linear2.bias)

    # Preserve gradient settings.
    new_linear1.weight.requires_grad_(old_linear1.weight.requires_grad)
    new_linear2.weight.requires_grad_(old_linear2.weight.requires_grad)

    if old_linear1.bias is not None:
        new_linear1.bias.requires_grad_(old_linear1.bias.requires_grad)

    if old_linear2.bias is not None:
        new_linear2.bias.requires_grad_(old_linear2.bias.requires_grad)

    # Preserve training/evaluation state.
    new_linear1.train(old_linear1.training)
    new_linear2.train(old_linear2.training)

    layer.linear1 = new_linear1
    layer.linear2 = new_linear2


def prune_mlp_by_magnitude(
    model: DPViT,
    config: PruningConfig,
) -> tuple[DPViT, dict]:
    """
    Globally prune MLP neurons using combined incoming and outgoing
    L1 weight magnitudes.

    At least one neuron is retained in each encoder layer.
    """

    if not isinstance(config, PruningConfig):
        raise TypeError("config must be a PruningConfig instance.")

    raw_ratio = config.target_ratio

    if (
        isinstance(raw_ratio, bool)
        or not isinstance(raw_ratio, Real)
        or not math.isfinite(float(raw_ratio))
        or not 0.0 <= float(raw_ratio) < 1.0
    ):
        raise ValueError("target_ratio must be in the range [0, 1).")

    target_ratio = float(raw_ratio)

    layers = _get_encoder_layers(model)

    widths_before = [
        int(layer.linear1.out_features) for layer in layers
    ]

    total_neurons = int(sum(widths_before))
    target_count = int(math.floor(target_ratio * total_neurons))

    parameters_before = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    # No physical modification is needed for a zero pruning ratio.
    if target_count == 0:
        return model, {
            "target_ratio": target_ratio,
            "target_neurons_to_prune": 0,
            "actual_neuron_prune_ratio": 0.0,
            "neurons_before": total_neurons,
            "neurons_pruned": 0,
            "neurons_after": total_neurons,
            "mlp_widths_before": widths_before,
            "mlp_widths_after": widths_before.copy(),
            "parameters_before": parameters_before,
            "parameters_after": parameters_before,
        }

    ranked_neurons = []

    # Calculate the magnitude score of every MLP neuron.
    for layer_index, layer in enumerate(layers):
        incoming_scores = (
            layer.linear1.weight.detach().abs().sum(dim=1)
        )

        outgoing_scores = (
            layer.linear2.weight.detach().abs().sum(dim=0)
        )

        scores = incoming_scores + outgoing_scores

        if not torch.isfinite(scores).all().item():
            raise ValueError(
                f"Non-finite magnitude scores in encoder layer {layer_index}."
            )

        for neuron_index, score in enumerate(scores.cpu().tolist()):
            ranked_neurons.append(
                (float(score), layer_index, neuron_index)
            )

    # Global ascending order: smallest magnitude is pruned first.
    ranked_neurons.sort(
        key=lambda item: (item[0], item[1], item[2])
    )

    keep_masks = [
        torch.ones(
            width,
            dtype=torch.bool,
            device=layer.linear1.weight.device,
        )
        for layer, width in zip(layers, widths_before)
    ]

    remaining_per_layer = widths_before.copy()
    pruned_count = 0

    # Select neurons while retaining at least one neuron per layer.
    for _, layer_index, neuron_index in ranked_neurons:
        if pruned_count >= target_count:
            break

        if remaining_per_layer[layer_index] <= 1:
            continue

        keep_masks[layer_index][neuron_index] = False
        remaining_per_layer[layer_index] -= 1
        pruned_count += 1

    # Apply physical pruning after all masks are determined.
    for layer, keep_mask in zip(layers, keep_masks):
        keep_indices = torch.where(keep_mask)[0]
        _prune_layer_neurons(layer, keep_indices)

    widths_after = [
        int(layer.linear1.out_features) for layer in layers
    ]

    parameters_after = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    report = {
        "target_ratio": target_ratio,
        "target_neurons_to_prune": target_count,
        "actual_neuron_prune_ratio": float(pruned_count / total_neurons),
        "neurons_before": total_neurons,
        "neurons_pruned": int(pruned_count),
        "neurons_after": int(total_neurons - pruned_count),
        "mlp_widths_before": widths_before,
        "mlp_widths_after": widths_after,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
    }

    return model, report