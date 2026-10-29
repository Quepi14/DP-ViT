"""DP-ViT model with adaptive patch embedding and a ViT encoder."""

import torch
from torch import Tensor, nn

from .config import DPViTConfig
from .patching import AdaptivePatchEmbedding

class DPViT(nn.Module):
    """Dynamic Patch Vision Transformer for image classification."""

    def __init__(self, config: DPViTConfig) -> None:
        super().__init__()

        self.config = config
        self.patch_embedding = AdaptivePatchEmbedding(config)

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
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

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(
            config.embed_dim,
            config.num_classes,
        )

        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Run the adaptive tokens through the encoder and classification head.

        Args:
            batch: Output from adaptive_collate_fn.

        Returns:
            Class logits with shape (batch_size, num_classes).
        """
        tokens, valid_mask = self.patch_embedding(batch)

        batch_size = tokens.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)

        #TransformerEncoder uses True to mark padded tokens.
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
        padding_mask = ~full_valid_mask

        # With fixed-budget patch maps nothing is padded. Passing no mask lets
        # PyTorch use its faster attention path (one small GPU sync per call).
        if not bool(padding_mask.any()):
            padding_mask = None

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        class_representation = self.norm(encoded[:, 0])
        logits = self.classifier(class_representation)

        return logits