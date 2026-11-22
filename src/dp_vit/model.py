
"""DP-ViT model with adaptive patch embedding and a ViT encoder."""

import torch
from torch import Tensor, nn

from .config import DPViTConfig
from .patching import AdaptivePatchEmbedding


class DPViT(nn.Module):
    """Dynamic Patch Vision Transformer for image classification."""

    def __init__(self, config: DPViTConfig) -> None:
        super().__init__()

        if not isinstance(config, DPViTConfig):
            raise TypeError("config must be a DPViTConfig instance.")

        self.config = config
        self.patch_embedding = AdaptivePatchEmbedding(config)

        # Learnable classification token.
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.embed_dim)
        )

        # Transformer encoder configuration.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(
                config.embed_dim * config.mlp_ratio
            ),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # Initialize each encoder layer independently.
        for layer in self.encoder.layers:
            for name, parameter in layer.named_parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

                elif name.endswith("weight") and "norm" in name:
                    nn.init.ones_(parameter)

                else:
                    nn.init.zeros_(parameter)

        # Final normalization and classification head.
        self.norm = nn.LayerNorm(config.embed_dim)

        self.classifier = nn.Linear(
            config.embed_dim,
            config.num_classes,
        )

        # Initialize the classification token.
        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Run adaptive patch tokens through the Transformer encoder.

        Args:
            batch:
                Output dictionary from adaptive_collate_fn
                or fast_adaptive_collate_fn.

        Returns:
            Class logits with shape (batch_size, num_classes).
        """
        # Adaptive patch projection and positional embedding.
        tokens, valid_mask = self.patch_embedding(batch)

        batch_size = tokens.shape[0]

        # Insert CLS token at the beginning of the sequence.
        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            (class_token, tokens),
            dim=1,
        )

        # True represents a valid token.
        class_token_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=valid_mask.device,
        )

        full_valid_mask = torch.cat(
            (class_token_mask, valid_mask),
            dim=1,
        )

        # TransformerEncoder uses True to identify padded tokens.
        padding_mask = ~full_valid_mask

        # Apply Transformer encoder with padding mask.
        encoded = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        # Use only the normalized CLS representation.
        class_representation = self.norm(
            encoded[:, 0]
        )

        # Generate classification logits.
        logits = self.classifier(
            class_representation
        )

        return logits
