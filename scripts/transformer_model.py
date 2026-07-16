# transformer_model.py

import torch
import torch.nn as nn


class StrokeTransformer(nn.Module):

    def __init__(
        self,
        input_dim=15,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        num_classes=62,
        max_strokes=3,
        dropout=0.15,
    ):

        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_strokes, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask=None):

        if mask is None:
            mask = torch.any(x != 0, dim=-1)

        x = self.input_norm(x)
        x = self.embedding(x)
        x = x + self.position_embedding[:, : x.size(1)]

        padding_mask = ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        mask_float = mask.unsqueeze(-1).float()
        x = x * mask_float
        pooled = x.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(pooled)

        return logits
