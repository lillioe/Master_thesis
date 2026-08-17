from __future__ import annotations

import torch
from torch import nn


class NTCandidateMLP(nn.Module):
    """
    Candidate classifier for precomputed NT embeddings
    Input:
        x: [batch, input_dim]
    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 1,
        dropout: float = 0.2,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        layers: list[nn.Module] = []

        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            dim = input_dim

            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(dim, hidden_dim))
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                dim = hidden_dim

            layers.append(nn.Linear(dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"logits": logits}