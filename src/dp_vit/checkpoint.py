"""Checkpoint saving and loading for DP-ViT."""

from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import DPViTConfig, ICSCalibration
from .model import DPViT


CHECKPOINT_FORMAT_VERSION = 1


def _sanitize_for_weights_only(value: Any) -> Any:
    """Recursively convert metadata values into weights_only-safe primitives."""
    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return float(value)

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return _sanitize_for_weights_only(value.tolist())

    if isinstance(value, dict):
        return {
            str(k): _sanitize_for_weights_only(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_sanitize_for_weights_only(item) for item in value]

    raise TypeError(
        f"Unsupported metadata type for weights_only checkpoint: {type(value).__name__}"
    )


def _get_mlp_widths(model: DPViT) -> list[int]:
    """Return the hidden width of every encoder MLP."""

    if not isinstance(model, DPViT):
        raise TypeError("model must be an instance of DPViT.")

    if not isinstance(model.encoder, nn.TransformerEncoder):
        raise TypeError("model.encoder must be a TransformerEncoder.")

    layers = list(model.encoder.layers)

    if not layers:
        raise ValueError(
            "The model must contain at least one encoder layer."
        )

    widths = []

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

        if layer.linear1.out_features != layer.linear2.in_features:
            raise ValueError(
                f"Encoder layer {index} has incompatible MLP dimensions."
            )

        if layer.linear1.in_features != model.config.embed_dim:
            raise ValueError(
                f"Encoder layer {index} has an invalid MLP input dimension."
            )

        if layer.linear2.out_features != model.config.embed_dim:
            raise ValueError(
                f"Encoder layer {index} has an invalid MLP output dimension."
            )

        if layer.linear1.out_features <= 0:
            raise ValueError(
                f"Encoder layer {index} must retain at least one neuron."
            )

        widths.append(int(layer.linear1.out_features))

    return widths


def _validate_mlp_widths(
    mlp_widths: Any,
    expected_depth: int,
) -> list[int]:
    """Validate hidden widths stored in a checkpoint."""

    if not isinstance(mlp_widths, (list, tuple)):
        raise TypeError(
            "mlp_widths must be a list or tuple."
        )

    if len(mlp_widths) != expected_depth:
        raise ValueError(
            "Checkpoint MLP widths do not match the encoder depth."
        )

    validated_widths = []

    for index, width in enumerate(mlp_widths):
        if (
            not isinstance(width, Integral)
            or isinstance(width, bool)
            or width <= 0
        ):
            raise ValueError(
                f"MLP width at index {index} must be a positive integer."
            )

        validated_widths.append(int(width))

    return validated_widths


def _resize_mlp_layer(
    layer: nn.TransformerEncoderLayer,
    hidden_width: int,
) -> None:
    """Resize an encoder layer's MLP before loading checkpoint weights."""

    if not isinstance(layer, nn.TransformerEncoderLayer):
        raise TypeError(
            "layer must be a TransformerEncoderLayer instance."
        )

    if (
        not isinstance(hidden_width, Integral)
        or isinstance(hidden_width, bool)
        or hidden_width <= 0
    ):
        raise ValueError(
            "hidden_width must be a positive integer."
        )

    if not isinstance(layer.linear1, nn.Linear):
        raise TypeError("layer.linear1 must be nn.Linear.")

    if not isinstance(layer.linear2, nn.Linear):
        raise TypeError("layer.linear2 must be nn.Linear.")

    old_linear1 = layer.linear1
    old_linear2 = layer.linear2

    if old_linear1.out_features != old_linear2.in_features:
        raise ValueError(
            "The original MLP layers have incompatible dimensions."
        )

    hidden_width = int(hidden_width)

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

    # Preserve gradient settings.
    new_linear1.weight.requires_grad_(
        old_linear1.weight.requires_grad
    )

    new_linear2.weight.requires_grad_(
        old_linear2.weight.requires_grad
    )

    if old_linear1.bias is not None:
        new_linear1.bias.requires_grad_(
            old_linear1.bias.requires_grad
        )

    if old_linear2.bias is not None:
        new_linear2.bias.requires_grad_(
            old_linear2.bias.requires_grad
        )

    # Preserve training/evaluation state.
    new_linear1.train(old_linear1.training)
    new_linear2.train(old_linear2.training)

    layer.linear1 = new_linear1
    layer.linear2 = new_linear2


def _validate_checkpoint_payload(payload: Any) -> None:
    """Validate the top-level checkpoint structure."""

    if not isinstance(payload, dict):
        raise TypeError(
            "Checkpoint payload must be a dictionary."
        )

    version = payload.get("format_version")

    if (
        not isinstance(version, Integral)
        or isinstance(version, bool)
        or version != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported or missing DP-ViT checkpoint format version."
        )

    required_keys = {
        "model_config",
        "ics_calibration",
        "mlp_widths",
        "model_state_dict",
    }

    missing_keys = required_keys.difference(payload)

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    if not isinstance(payload["model_config"], dict):
        raise TypeError(
            "model_config must be a dictionary."
        )

    if not isinstance(payload["ics_calibration"], dict):
        raise TypeError(
            "ics_calibration must be a dictionary."
        )

    if not isinstance(payload["model_state_dict"], dict):
        raise TypeError(
            "model_state_dict must be a dictionary."
        )

    if not all(
        isinstance(name, str)
        and isinstance(value, torch.Tensor)
        for name, value in payload["model_state_dict"].items()
    ):
        raise TypeError(
            "model_state_dict must contain string keys and tensor values."
        )

    if not isinstance(payload["mlp_widths"], (list, tuple)):
        raise TypeError(
            "mlp_widths must be a list or tuple."
        )

    class_names = payload.get("class_names")

    if class_names is not None:
        if (
            not isinstance(class_names, list)
            or not all(
                isinstance(name, str) and name.strip()
                for name in class_names
            )
        ):
            raise ValueError(
                "class_names must be a list of non-empty strings or None."
            )

    pruning_report = payload.get("pruning_report")

    if (
        pruning_report is not None
        and not isinstance(pruning_report, dict)
    ):
        raise TypeError(
            "pruning_report must be a dictionary or None."
        )


def _validate_class_names(
    class_names: list[str] | None,
    num_classes: int,
) -> None:
    """Validate class names against the model output dimension."""

    if class_names is None:
        return

    if not isinstance(class_names, list):
        raise TypeError(
            "class_names must be a list or None."
        )

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names must contain one name for every model class."
        )

    if not all(
        isinstance(name, str) and name.strip()
        for name in class_names
    ):
        raise ValueError(
            "class_names must contain non-empty strings."
        )


def save_checkpoint(
    path: str | Path,
    model: DPViT,
    ics_calibration: ICSCalibration,
    class_names: list[str] | None = None,
    pruning_report: dict[str, Any] | None = None,
) -> Path:
    """Save a DP-ViT checkpoint and reconstruction metadata."""

    if not isinstance(model, DPViT):
        raise TypeError(
            "model must be an instance of DPViT."
        )

    if not isinstance(ics_calibration, ICSCalibration):
        raise TypeError(
            "ics_calibration must be an ICSCalibration instance."
        )

    _validate_class_names(
        class_names,
        model.config.num_classes,
    )

    if (
        pruning_report is not None
        and not isinstance(pruning_report, dict)
    ):
        raise TypeError(
            "pruning_report must be a dictionary or None."
        )

    mlp_widths = _get_mlp_widths(model)

    try:
        checkpoint_path = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "path must be a valid filesystem path."
        ) from exc

    if not str(checkpoint_path).strip() or str(checkpoint_path) == ".":
        raise ValueError(
            "path must specify a checkpoint filename."
        )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": _sanitize_for_weights_only(asdict(model.config)),
        "ics_calibration": _sanitize_for_weights_only(asdict(ics_calibration)),
        "class_names": (
            class_names.copy()
            if class_names is not None
            else None
        ),
        "mlp_widths": mlp_widths,
        "pruning_report": (
            _sanitize_for_weights_only(pruning_report)
            if pruning_report is not None
            else None
        ),
        "model_state_dict": model_state_dict,
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    try:
        torch.save(
            payload,
            temporary_path,
        )

        temporary_path.replace(checkpoint_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[DPViT, ICSCalibration, dict[str, Any]]:
    """
    Load a DP-ViT checkpoint and reconstruct its architecture.

    Returns:
        model: Reconstructed DP-ViT model.
        ics_calibration: Saved ICS calibration parameters.
        metadata: Class names, MLP widths, and pruning report.
    """

    checkpoint_path = Path(path).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {checkpoint_path}"
        )

    try:
        device = torch.device(map_location)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "map_location must be a valid device or device string."
        ) from exc

    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    _validate_checkpoint_payload(payload)

    # Reconstruct the original model configuration.
    config = DPViTConfig(
        **payload["model_config"]
    )

    # Reconstruct ICS calibration (with fallback for legacy p30/p70 keys).
    cal_data = dict(payload["ics_calibration"])
    if "p30" in cal_data and "t_low" not in cal_data:
        cal_data["t_low"] = cal_data.pop("p30")
    if "p70" in cal_data and "t_high" not in cal_data:
        cal_data["t_high"] = cal_data.pop("p70")

    calibration = ICSCalibration(**cal_data)

    # Reconstruct the base DP-ViT architecture.
    model = DPViT(config)

    encoder_layers = list(model.encoder.layers)

    # Restore the MLP dimensions after physical pruning.
    mlp_widths = _validate_mlp_widths(
        payload["mlp_widths"],
        expected_depth=len(encoder_layers),
    )

    for layer, hidden_width in zip(
        encoder_layers,
        mlp_widths,
    ):
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise TypeError(
                "Encoder layers must be TransformerEncoderLayer instances."
            )

        _resize_mlp_layer(
            layer,
            hidden_width,
        )

    # Load model parameters strictly.
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.to(device)

    class_names = payload.get("class_names")

    if class_names is not None:
        _validate_class_names(
            class_names,
            config.num_classes,
        )

    metadata = {
        "class_names": class_names,
        "mlp_widths": mlp_widths,
        "pruning_report": payload.get("pruning_report"),
    }

    return model, calibration, metadata