from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.models.candidate_transformer import CandidateTransformer


class TrainableCenterAttentionPooling(nn.Module):
    """
    Candidate-window pooling

    Input:
        hidden_states: (M, T, H)
        attention_mask: (M, T)

    Output:
        pooled candidate vectors: (M, 2H)

    It returns:
        concat(center_token_embedding, attention_pooled_embedding)
    """

    def __init__(self, hidden_dim: int, attn_hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()

        if attn_hidden_dim is None:
            attn_hidden_dim = max(128, hidden_dim // 2)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        center_token_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be (M,T,H), got {tuple(hidden_states.shape)}")

        m, t, h = hidden_states.shape

        if center_token_index >= t:
            center_token_index = t // 2

        center_vec = hidden_states[:, center_token_index, :]

        scores = self.scorer(hidden_states).squeeze(-1)  # (M, T)
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)

        weights = torch.softmax(scores, dim=1)  # (M, T)
        attn_vec = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        pooled = torch.cat([center_vec, attn_vec], dim=-1)

        return pooled, weights


class NTPoolCandidateTransformer(nn.Module):
    """
    End- to- end model:
        DNA windows, NT, trainable pooling, 
        candidate transformer, candidate lables

    receives list of genes, where each gene is a list
    of candidate DNA windows
    """

    def __init__(
        self,
        model_id: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        nt_layer: int = -1,
        freeze_nt: bool = True,
        unfreeze_last_nt_layers: int = 0,
        window_size: int = 600,
        num_classes: int = 2,
        transformer_model_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        append_position_features: bool = True,
        pooling_dropout: float = 0.1,
    ):
        super().__init__()

        self.model_id = model_id
        self.nt_layer = nt_layer
        self.window_size = window_size
        self.append_position_features = append_position_features

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.nt = AutoModelForMaskedLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.set_nt_trainability(
            freeze_nt=freeze_nt,
            unfreeze_last_nt_layers=unfreeze_last_nt_layers,
        )

        # Infer hidden dimension cheaply from config if possible.
        hidden_dim = getattr(self.nt.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.nt.config, "d_model", None)
        if hidden_dim is None:
            raise ValueError("Could not infer NT hidden dimension from config.")

        self.nt_hidden_dim = int(hidden_dim)

        self.pool = TrainableCenterAttentionPooling(
            hidden_dim=self.nt_hidden_dim,
            dropout=pooling_dropout,
        )

        candidate_dim = 2 * self.nt_hidden_dim

        if append_position_features:
            # position_norm, is_donor, is_acceptor
            candidate_dim += 3

        self.candidate_transformer = CandidateTransformer(
            input_dim=candidate_dim,
            num_classes=num_classes,
            num_layers=transformer_layers,
            n_attention_heads=transformer_heads,
            dropout=transformer_dropout,
            model_dim=transformer_model_dim,
            use_input_projection=True,
        )

    def set_nt_trainability(self, freeze_nt: bool, unfreeze_last_nt_layers: int = 0) -> None:
        for p in self.nt.parameters():
            p.requires_grad = False

        if freeze_nt and unfreeze_last_nt_layers <= 0:
            self.nt.eval()
            return

        if unfreeze_last_nt_layers <= 0:
            for p in self.nt.parameters():
                p.requires_grad = True
            self.nt.train()
            return

        # unfreeze last N encoder layers by parsing parameter names
        layer_ids = set()

        for name, _ in self.nt.named_parameters():
            for pattern in [
                r"encoder\.layer\.(\d+)\.",
                r"encoder\.layers\.(\d+)\.",
                r"layers\.(\d+)\.",
                r"layer\.(\d+)\.",
            ]:
                m = re.search(pattern, name)
                if m:
                    layer_ids.add(int(m.group(1)))

        if not layer_ids:
            raise ValueError(
                "Could not identify NT layer numbers "
            )

        max_layer = max(layer_ids)
        keep_from = max_layer - unfreeze_last_nt_layers + 1

        for name, p in self.nt.named_parameters():
            unfreeze = False

            for pattern in [
                r"encoder\.layer\.(\d+)\.",
                r"encoder\.layers\.(\d+)\.",
                r"layers\.(\d+)\.",
                r"layer\.(\d+)\.",
            ]:
                m = re.search(pattern, name)
                if m and int(m.group(1)) >= keep_from:
                    unfreeze = True
                    break

            if unfreeze:
                p.requires_grad = True

        self.nt.train()

        n_trainable = sum(p.numel() for p in self.nt.parameters() if p.requires_grad)
        print(f"NT trainable parameters: {n_trainable:,}")

    def tokenize_flat_windows(
        self,
        windows: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            windows,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in tokens.items()}

    def center_token_index(self) -> int:
        # CLS token offset = 1, NT token size roughly 6 bp.
        center_nt_index = self.window_size // 2
        return 1 + (center_nt_index // 6)

    def encode_candidate_windows(
        self,
        flat_windows: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        # Encode candidate windows through NT in smaller internal chunks
        # preserves the full candidate sequence for the downstream candidate transformer
        # avoids 600 bp windows through NT at once
        nt_chunk_size = getattr(self, "nt_chunk_size", 64)

        need_grad = any(p.requires_grad for p in self.nt.parameters())
        pooled_chunks = []

        for start in range(0, len(flat_windows), nt_chunk_size):
            sub_windows = flat_windows[start : start + nt_chunk_size]
            tokens = self.tokenize_flat_windows(sub_windows, device=device)

            if need_grad:
                out = self.nt(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    output_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    out = self.nt(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"],
                        output_hidden_states=True,
                    )

            hidden = out.hidden_states[self.nt_layer]

            pooled, _weights = self.pool(
                hidden_states=hidden,
                attention_mask=tokens["attention_mask"],
                center_token_index=self.center_token_index(),
            )

            pooled_chunks.append(pooled)

        return torch.cat(pooled_chunks, dim=0)

    def forward(
        self,
        batch_windows: list[list[str]],
        candidate_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = candidate_mask.device
        batch_size, max_candidates = candidate_mask.shape

        flat_windows = []
        flat_positions = []

        for b in range(batch_size):
            valid_n = int(candidate_mask[b].sum().item())
            for j in range(valid_n):
                flat_windows.append(batch_windows[b][j])
                flat_positions.append((b, j))

        flat_vecs = self.encode_candidate_windows(flat_windows, device=device)

        candidate_dim = flat_vecs.shape[-1]

        if self.append_position_features:
            if position_features is None:
                raise ValueError("position_features required when append_position_features=True")
            candidate_dim += position_features.shape[-1]

        x = torch.zeros(
            (batch_size, max_candidates, candidate_dim),
            dtype=flat_vecs.dtype,
            device=device,
        )

        for k, (b, j) in enumerate(flat_positions):
            if self.append_position_features:
                x[b, j] = torch.cat(
                    [flat_vecs[k], position_features[b, j].to(flat_vecs.dtype)],
                    dim=0,
                )
            else:
                x[b, j] = flat_vecs[k]

        return self.candidate_transformer(
            x=x.float(),
            attention_mask=candidate_mask,
            labels=labels,
        )