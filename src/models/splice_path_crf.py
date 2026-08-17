
"""
Splice-path CRF model

Model uses an existing linear-chain CRF implementation from torchcrf
The model assumes candidate-level NT logits have already been exported
Input logits are binary NT usage logits:
    site_logits[..., 0] = logit_not_used
    site_logits[..., 1] = logit_used

One CRF training example is one transcript-specific path over all candidates
of a gene,

model returns the negative CRF log-likelihood of the
annotated transcript path during training. 
During initial inference, the model uses Viterbi
decoding to return the single highest-scoring path. Posterior sampling from
the CRF distribution can be added later

Shapes:
    B = batch size
    L = number of candidates after padding
    K = number of CRF states

Inputs:
    site_logits:        (B, L, 2)
    candidate_type_ids: (B, L), 0 = donor candidate, 1 = acceptor candidate
    attention_mask:     (B, L), True/1 = valid candidate
    labels:             (B, L), optional, -1 at padding positions
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchcrf import CRF


DONOR_CANDIDATE = 0
ACCEPTOR_CANDIDATE = 1
NEG_INF = -10_000.0


LABELS_3 = {
    "skip": 0,
    "donor": 1,
    "acceptor": 2,
}


LABELS_4 = {
    "S_D": 0,
    "D": 1,
    "S_A": 2,
    "A": 3,
}


@dataclass(frozen=True)
class SpliceCRFConfig:
    label_mode: str = "3"  # "3" or "4"
    use_logit_calibration: bool = True
    enforce_transition_constraints: bool = False


class SplicePathCRFModel(nn.Module):
    """
    Linear-chain CRF for transcript-specific splice-path prediction,
    class maps binary NT logits to CRF emissions,
    applies splice specific emission constraints,
    optional transcition constraints,
    torchcrf for crf likelihood, viterbi decoding.
    """

    def __init__(self, config: SpliceCRFConfig):
        super().__init__()

        if config.label_mode not in {"3", "4"}:
            raise ValueError("label_mode must be '3' or '4'")

        self.config = config
        self.label_mode = config.label_mode

        self.label_to_id = LABELS_3 if self.label_mode == "3" else LABELS_4
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        if config.use_logit_calibration:
            # Calibrates [logit_not_used, logit_used].
            self.logit_scale = nn.Parameter(torch.ones(2))
            self.logit_bias = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("logit_scale", torch.ones(2))
            self.register_buffer("logit_bias", torch.zeros(2))

        transition_mask = self.build_transition_mask()
        self.register_buffer("transition_mask", transition_mask)

    def build_transition_mask(self) -> torch.Tensor:
        """
        Return a boolean matrix of allowed label-to-label transitions.
        
        Rows are previous labels, columns are next labels.

        For 3-state mode, all transitions are allowed 

        For 4-state mode, hard donor/acceptor ordering enforced with:
            S_D: S_D or D
            D: S_A or A
            S_A: S_A or A
            A: S_D or D
        """

        mask = torch.ones(self.num_labels, self.num_labels, dtype=torch.bool)

        if self.label_mode == "3" or not self.config.enforce_transition_constraints:
            return mask

        mask[:] = False

        S_D = LABELS_4["S_D"]
        D = LABELS_4["D"]
        S_A = LABELS_4["S_A"]
        A = LABELS_4["A"]

        mask[S_D, S_D] = True
        mask[S_D, D] = True

        mask[D, S_A] = True
        mask[D, A] = True

        mask[S_A, S_A] = True
        mask[S_A, A] = True

        mask[A, S_D] = True
        mask[A, D] = True

        return mask

    def apply_transition_constraints(self) -> None:
        """
        torchcrf does not expose a transition mask argument,
        invalid transitions avoided before CRF calls
        """

        if not self.config.enforce_transition_constraints:
            return

        with torch.no_grad():
            invalid = ~self.transition_mask
            self.crf.transitions.data[invalid] = NEG_INF

    def build_emissions(
        self,
        site_logits: torch.Tensor,
        candidate_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map binary NT logits to CRF emissions.

        site_logits[..., 0] = not used logit
        site_logits[..., 1] = used logit

        For donor candidates:
            used: donor/D emission
            acceptor/A emission is forbidden

        For acceptor candidates:
            used:  acceptor/A emission
            donor/D emission is forbidden
        """

        if site_logits.ndim != 3 or site_logits.shape[-1] != 2:
            raise ValueError("site_logits must have shape (B, L, 2)")

        if candidate_type_ids.shape != site_logits.shape[:2]:
            raise ValueError("candidate_type_ids must have shape (B, L)")

        calibrated = site_logits * self.logit_scale + self.logit_bias

        logit_not_used = calibrated[..., 0]
        logit_used = calibrated[..., 1]

        B, L, _ = site_logits.shape

        emissions = site_logits.new_full(
            (B, L, self.num_labels),
            fill_value=NEG_INF,
        )

        donor_mask = candidate_type_ids == DONOR_CANDIDATE
        acceptor_mask = candidate_type_ids == ACCEPTOR_CANDIDATE

        if self.label_mode == "3":
            skip = LABELS_3["skip"]
            donor = LABELS_3["donor"]
            acceptor = LABELS_3["acceptor"]

            emissions[..., skip] = logit_not_used

            emissions[..., donor] = torch.where(
                donor_mask,
                logit_used,
                site_logits.new_full((B, L), NEG_INF),
            )

            emissions[..., acceptor] = torch.where(
                acceptor_mask,
                logit_used,
                site_logits.new_full((B, L), NEG_INF),
            )

        else:
            S_D = LABELS_4["S_D"]
            D = LABELS_4["D"]
            S_A = LABELS_4["S_A"]
            A = LABELS_4["A"]

            emissions[..., S_D] = logit_not_used
            emissions[..., S_A] = logit_not_used

            emissions[..., D] = torch.where(
                donor_mask,
                logit_used,
                site_logits.new_full((B, L), NEG_INF),
            )

            emissions[..., A] = torch.where(
                acceptor_mask,
                logit_used,
                site_logits.new_full((B, L), NEG_INF),
            )

        return emissions

    def forward(
        self,
        site_logits: torch.Tensor,
        candidate_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decode_mode: str = "viterbi",
    ) -> dict:
        """
        Forward pass.

        If labels are provided, the model returns the negative CRF
        log-likelihood loss for the gold transcript path.

        If labels are not provided, the model runs inference.

        decode_mode:
            "viterbi":
                Return the single highest-scoring path under the CRF.
                This uses torchcrf.decode().
        """

        emissions = self.build_emissions(
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
        )

        self.apply_transition_constraints()

        if labels is not None:
            crf_mask = labels != -1

            safe_labels = labels.clone()
            safe_labels[safe_labels == -1] = 0

            log_likelihood = self.crf(
                emissions,
                safe_labels,
                mask=crf_mask.bool(),
                reduction="none",
            )

            return {
                "loss": -log_likelihood.mean(),
                "log_likelihood": log_likelihood,
                "emissions": emissions,
            }

        if decode_mode == "viterbi":
            predictions = self.crf.decode(
                emissions,
                mask=attention_mask.bool(),
            )

            return {
                "predictions": predictions,
                "decode_mode": "viterbi",
                "emissions": emissions,
            }

        if decode_mode == "sample":
            raise NotImplementedError(
                "Posterior sampling not implemented yet. "
                "The current torchcrf-based implementation supports training "
                "with CRF negative log-likelihood and Viterbi decoding. "
            )

        raise ValueError(
            f"Unknown decode_mode={decode_mode!r}. "
            "Expected 'viterbi' or 'sample'."
        )