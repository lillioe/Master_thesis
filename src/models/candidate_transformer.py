from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateTransformer(nn.Module):
    """
    Transformer encoder for splice site candidate classification.

    Input:
        x: candidate feature vectors, shape (B, N, D)
           B = batch size
           N = number of candidates in gene/region
           D = candidate vector dimension, (NT dim)

        attention_mask: shape (B, N)
           1 / True = valid candidate
           0 / False = padding

        labels: optional, shape (B, N)
           class indices per candidate
           use -100 for padded positions

    Output:
        During training:
            {
                "loss": scalar,
                "logits": (B, N, C)
            }

        During inference:
            {
                "logits": (B, N, C),
                "predictions": (B, N)
            }
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        num_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        use_input_projection: bool = True,
        model_dim: int | None = None,
        activation: str = "gelu",
    ):
        super().__init__()

        if model_dim is None:
            model_dim = input_dim

        if dim_feedforward is None:
            dim_feedforward = 4 * model_dim

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes

        if use_input_projection or input_dim != model_dim:
            self.input_projection = nn.Linear(input_dim, model_dim)
        else:
            self.input_projection = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        x:
            Float tensor of shape (B, N, D)

        attention_mask:
            Bool/int tensor of shape (B, N)
            1 = valid candidate, 0 = padding

        labels:
            Long tensor of shape (B, N)
            Labels should be 0 to num_classes-1.
            Padded labels should be -100.
        """

        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, N, D), got {tuple(x.shape)}")

        h = self.input_projection(x)

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape (B, N), "
                    f"got {tuple(attention_mask.shape)}"
                )

            # PyTorch Transformer expects True at padded positions.
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        h = self.encoder(
            h,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.norm(h)
        logits = self.classifier(h)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        else:
            output["predictions"] = logits.argmax(dim=-1)

        return output


class CandidateMLPBaseline(nn.Module):
    """
    classify each candidate independently.

    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.net(x)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        else:
            output["predictions"] = logits.argmax(dim=-1)

        return output