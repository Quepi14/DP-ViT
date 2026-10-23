"""Structured magnitude-based pruning for DP-ViT MLP layers."""

import torch 
from torch import nn

from .config import PruningConfig
from .model import DPViT

def _get_encoder_layers(model: DPViT) -> list[nn.TransformerEncoderLayer]:
    """return the Transformer encoder layers from DP-ViT model."""
    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT.")

    if not isinstance(model.encoder, nn.TransformerEncoderLayer):
        raise TypeError("model.encoder must be a TransformerEncoder.")

    layers = list(model.encoder.layers)

    if not layers:
        raise ValueError("the model must contain at least one encoder layer.")

    if not all(
        isinstance(layer, nn.TransformerEncoderLayer)
        for layer in layers
    ):
        raise TypeError("All encoder layers must be TransformerEncoderLayer instances.")

    return [
        layer
        for layer in layers
        if isinstance(layer, nn.TransformerEncoderLayer)
    ]

def _prune_layer_neurons(
    layer: nn.TransformerEncoderLayer,
    keep_indices: torch.Tensor,
) -> None:
    """Physically remove selected neurons from an encoder layer's MLP."""
    old_linear1 = layer.linear1
    old_linear2 = layer.linear2

    keep_indices = keep_indices.to(old_linear1.weight.device)

    new_linear1 = nn.Linear(
        in_features=old_linear1.in_features,
        out_features=keep_indices.numel(),
        bias=old_linear1.bias is not None,
        device=old_linear1.weight.device,
        dtype=old_linear1.weight.dtype,
    )

    new_linear2 = nn.Linear(
        in_features=keep_indices.numel(),
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

    layer.linear1 = new_linear1
    layer.linear2 = new_linear2

def prune_mlp_by_magnitude(
    model: DPViT,
    config: PruningConfig,
) -> tuple[DPViT, dict]:
    """
    Prune MLP neurons with the lowest incoming-weight L1 magnitudes.

    Pruning is applied globally across all encoder layers. At least one
    neuron in retained in each layer.
    """
    layers = _get_encoder_layers(model)

    if not isinstance(config, PruningConfig):
        raise TypeError("config must be a PruningConfig instance.")

    total_neurons = sum(layer.linear1.out_features for layer in layers)
    target_count = int(config.target_ratio * total_neurons)

    scores_by_layer = []
    ranked_neurons = []

    for layer_index, layer in enumerate(layers):
        # Each score is the L1 magnitude of one neuron's incoming weights.
        scores = layer.linear1.weight.detach().abs().sum(dim=1)
        scores_by_layer.append(scores)

        for neuron_index, score in enumerate(scores.cpu().tolist()):
            ranked_neurons.append(
                (float(score), layer_index, neuron_index)
            )

    ranked_neurons.sort(key=lambda item: item[0])

    keep_masks = [
        torch.ones(
            layer.linear1.out_features,
            dtype=torch.bool,
            device=layer.linear1.weight.device,
        )
        for layer in layers
    ]

    remaining_per_layer = [
        layer.linear1.out_features for layer in layers
    ]

    pruned_count = 0

    for _, layer_index, neuron_index in ranked_neurons:
        if pruned_count >= target_count:
            break

        # Keep at least one MLP neruon in every encoder layer.
        if remaining_per_layer[layer_index] <= 1:
            continue

        keep_masks[layer_index][neuron_index] = False
        remaining_per_layer[layer_index] -= 1
        pruned_count += 1

    widths_before = [
        layer.linear1.out_features for layer in layers
    ]
    parameters_before = sum(
        parameter.numel() for parameter in model.parameters()
    )

    for layer, keep_masks in zip(layers, keep_masks):
        keep_indices = torch.where(keep_masks)[0]
        _prune_layer_neurons(layer, keep_indices)

    parameters_after = sum(
        parameter.numel() for parameter in model.parameters()
    )
    widths_after = [
        layer.linear1.out_features for layer in layers
    ]

    report = {
        "target_ratio": config.target_ratio,
        "actual_neuron_prune_ratio": pruned_count / total_neurons,
        "neurons_before": total_neurons,
        "neurons_pruned": pruned_count,
        "neurons_after": total_neurons - pruned_count,
        "mlp_widths_before": widths_before,
        "mlp_widths_after": widths_after,
        "parameters_after": parameters_after,
    }

    return model, report
