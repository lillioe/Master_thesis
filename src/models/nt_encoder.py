import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM


class NucleotideTransformerEncoder(nn.Module):
    """
    Wrapper around pretrained NT

    freeze=True:
        NT is used as fixed feature extractor
    freeze=False:
        NT weights can be fine-tuned together with downstream model
    """

    def __init__(
        self,
        model_id: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        freeze: bool = True,
        layer: int = -1,
    ):
        super().__init__()

        self.model_id = model_id
        self.layer = layer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.set_freeze(freeze)

    def set_freeze(self, freeze: bool):
        self.freeze = freeze

        for p in self.model.parameters():
            p.requires_grad = not freeze

        if freeze:
            self.model.eval()
        else:
            self.model.train()

    def tokenize(self, sequences, device=None):
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}

        return tokens

    def forward(self, input_ids, attention_mask):
        if self.freeze:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = out.hidden_states[self.layer]

        return {
            "hidden_states": hidden,
            "attention_mask": attention_mask,
        }


def mean_pool_embeddings(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def center_plus_mean_pool(hidden_states, attention_mask, window_size):
    mean_vec = mean_pool_embeddings(hidden_states, attention_mask)

    center_nt_index = window_size // 2
    center_token_index = 1 + (center_nt_index // 6)

    center_vec = hidden_states[:, center_token_index, :]

    return torch.cat([center_vec, mean_vec], dim=-1)