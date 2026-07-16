import math

import torch
import torch.nn as nn


class PointSequenceTransformer(nn.Module):

    def __init__(
        self,
        input_dim=5,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=62,
        max_points=96,
        dropout=0.20,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_points + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        x = self.input_norm(x)
        x = self.embedding(x) * math.sqrt(x.size(-1))

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embedding[:, : x.size(1)]

        cls_mask = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=~full_mask)
        logits = self.classifier(x[:, 0])

        return logits
