"""Checkpoint saving and loading for DP-ViT."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .config import DPViTConfig, ICSCalibration
from .model import DPViT

CHECKPOINT_FORMAT_VERSION = 1

def _get_mlp_widths(model: DPViT) -> list[int]:
    """Return the hidden width of every encoder MLP."""
    return [
        int(getattr(getattr(layer, "linear1"), "out_features"))
        for layer in model.encoder.layers
    ]

def _resize_mlp_layer(
    layer: nn.TransformerEncoderLayer,
    hidden_width: int,
) -> None:
    """Create MLP layers with a target width before leading checkpoint weights"""
    if hidden_width <= 0:
        raise ValueError("Each MLP hidden width must be greater than zero.")

    old_linear1 = layer.linear1
    old_linear2 = layer.linear2

    new_linear1 = nn.Linear(
        in_features=old_linear1.in_features,
        out_features=hidden_width,
        bias=old_linear1.bias is not None,
        device=old_linear1.weight.device,
        dtype=old_linear1.weight.dtype,
    )
    new_linear2 = nn.Linear(
        in_features=hidden_width,
        out_features=old_linear2.out_features,
        bias=old_linear2.bias is not None,
        device=old_linear2.weight.device,
        dtype=old_linear2.weight.dtype,
    )

    layer.linear1 = new_linear1
    layer.linear2 = new_linear2

def save_checkpoint(
    path: str | Path,
    model: DPViT,
    ics_calibration: ICSCalibration,
    class_names: list[str] | None = None,
    pruning_report: dict[str, Any] | None = None,
) -> Path:
    """Save a DP-ViT checkpoint and the metadata needed to rebuild it."""
    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT")

    if not isinstance(ics_calibration, ICSCalibration):
        raise TypeError("ics_calibration must be an ICSCalibration instance.")

    if class_names is not None:
        if len(class_names) != model.config.num_classes:
            raise ValueError(
                "The number of class names must match num_classes."
            )

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": asdict(model.config),
        "ics_calibration": asdict(ics_calibration),
        "class_names": class_names,
        "mlp_widths": _get_mlp_widths(model),
        "pruning_report": pruning_report,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)

    return checkpoint_path

def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[DPViT, ICSCalibration, dict[str, Any]]:
    """
    Load a DP-ViT checkpoint and reconstruct its architecture.

    Returns:
        model: The reconstructured DP-ViT model.
        ics_calibration: The ICS parameters saved with the model.
        metadata: Class names, MLP widths, and pruning report.
    """
    checkpoint_path = Path(path).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported or missing DP-ViT checkpoint format version.")

    config = DPViTConfig(**payload["model_config"])
    calibration = ICSCalibration(**payload["ics_calibration"])

    model = DPViT(config)

    mlp_widths = payload["mlp_widths"]
    encoder_layers = list(model.encoder.layers)

    if len(mlp_widths) != len(encoder_layers):
        raise ValueError("Checkpoint MLP widths do not match the encoder depth.")

    for layer, hidden_width in zip(encoder_layers, mlp_widths):
        _resize_mlp_layer(
            cast(nn.TransformerEncoderLayer, layer),
            int(hidden_width),
        )

    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)

    metadata = {
        "class_names": payload.get("class_names"),
        "mlp_widths": mlp_widths,
        "pruning_report": payload.get("pruning_report"),
    }

    return model, calibration, metadata