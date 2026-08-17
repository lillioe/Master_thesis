"""
Splice-path CRF model.

Module defines a linear-chain CRF for transcript specific splice site
path prediction over ordered donor/acceptor candidates.

The model assumes candidate NT-transformer logits have already been
computed.

One CRF training example is one transcript path over all candidates of a gene.

Shape notation:
    B = batch size
    L = number of candidates in the gene
    K = number of CRF labels
        K = 3 for {skip, donor, acceptor}
        K = 4 for {S_D, D, S_A, A}

Inputs:
    site_logits:
        Tensor of shape (B, L, 2)
        Raw NT-transformer logits.
        Column 0 = donor logit
        Column 1 = acceptor logit

    candidate_type_ids:
        Tensor of shape (B, L)
        0 = donor candidate
        1 = acceptor candidate

    attention_mask:
        Tensor of shape (B, L)
        1/True = valid candidate
        0/False = padding

    labels:
        Tensor of shape (B, L), optional
        Ground-truth CRF label ids.
        Padding positions should be -1.

Outputs:
    If labels are provided:
        {"loss": scalar, "emissions": Tensor}

    If labels are not provided:
        {"predictions": list[list[int]], "emissions": Tensor}
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
    """
    Configuration for the splice-path CRF.

    label_mode:
        "3", labels {skip, donor, acceptor}
        "4", labels {S_D, D, S_A, A}

    use_logit_calibration:
        If True, learn a scale and bias for donor/acceptor NT logits before
        using them as CRF emissions.

    enforce_emission_constraints:
        If True, forbid impossible label/candidate-type combinations.
        Example: an acceptor candidate cannot emit the donor label.

    enforce_transition_constraints:
        If True, apply hard transition constraints.
    """

    label_mode: str = "3"
    use_logit_calibration: bool = True
    enforce_emission_constraints: bool = True
    enforce_transition_constraints: bool = False


class SplicePathCRF(nn.Module):
    """
    Linear-chain CRF wrapper for splice-path prediction.

    This follows the same role as the DeepCDS LinearChainCRF class:
    it receives emissions and labels, computes CRF NLL during training,
    and decodes the most likely path during inference.
    """

    def __init__(
        self,
        num_labels: int,
        transition_mask: torch.Tensor | None = None,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.crf = CRF(num_tags=num_labels, batch_first=True)

            self.register_buffer("transition_mask", transition_mask.bool())
        else:
            self.transition_mask = None

    def apply_transition_constraints(self) -> None:
        """
        Assign impossible transitions to a very negative value.
        This is mainly for the 4-state model
        """

        if self.transition_mask is None:
            return

        with torch.no_grad():
            invalid = ~self.transition_mask
            self.crf.transitions.data[invalid] = NEG_INF

    def forward(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Run CRF training or decoding.

        Args:
            emissions:
                Tensor of shape (B, L, K)

            attention_mask:
                Tensor of shape (B, L), 1/True for valid candidates

            labels:
                Optional tensor of shape (B, L)
                Padding positions should be -1.

        Returns:
            During training:
                {"loss": scalar, "emissions": emissions}

            During inference:
                {"predictions": list[list[int]], "emissions": emissions}
        """

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

            loss = -log_likelihood.mean()

            return {
                "loss": loss,
                "emissions": emissions,
            }

        crf_mask = attention_mask.bool()
        predictions = self.crf.decode(emissions, mask=crf_mask)

        return {
            "predictions": predictions,
            "emissions": emissions,
        }


class SplicePathCRFModel(nn.Module):
    """
    CRF model for splice-site path prediction from NT-transformer logits.
    Expects raw donor/acceptor logits as input.
    """

    def __init__(self, config: SpliceCRFConfig):
        super().__init__()

        if config.label_mode not in {"3", "4"}:
            raise ValueError("label_mode must be '3' or '4'")

        self.config = config
        self.label_mode = config.label_mode

        if self.label_mode == "3":
            self.label_to_id = LABELS_3
        else:
            self.label_to_id = LABELS_4

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        if config.use_logit_calibration:
            # Separate calibration for donor and acceptor logits.
            # Start as identity transform:
            # calibrated_logit = 1 * raw_logit + 0
            self.logit_scale = nn.Parameter(torch.ones(2))
            self.logit_bias = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("logit_scale", torch.ones(2))
            self.register_buffer("logit_bias", torch.zeros(2))

        # Learnable skip-state scores.
        # For 3 labels: one skip score.
        # For 4 labels: separate S_D and S_A skip-context scores.
        if self.label_mode == "3":
            self.skip_bias = nn.Parameter(torch.zeros(1))
        else:
            self.skip_bias = nn.Parameter(torch.zeros(2))

        transition_mask = None
        if config.enforce_transition_constraints:
            transition_mask = self.build_transition_mask()

        self.crf = SplicePathCRF(
            num_labels=self.num_labels,
            transition_mask=transition_mask,
        )

    def build_transition_mask(self) -> torch.Tensor:
        """
        Build hard transition mask.

        For label_mode="3", no strong biological ordering enforced

        For label_mode="4", enforce:

            S_D -> S_D or D
            D   -> S_A or A
            S_A -> S_A or A
            A   -> S_D or D
        """

        mask = torch.zeros(self.num_labels, self.num_labels, dtype=torch.bool)

        if self.label_mode == "3":
            mask[:, :] = True
            return mask

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

    def build_emissions(
        self,
        site_logits: torch.Tensor,
        candidate_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw donor/acceptor logits into CRF emissions.

        Args:
            site_logits:
                Tensor of shape (B, L, 2)
                site_logits[..., 0] = donor logit
                site_logits[..., 1] = acceptor logit

            candidate_type_ids:
                Tensor of shape (B, L)
                0 = donor candidate
                1 = acceptor candidate

        Returns:
            emissions:
                Tensor of shape (B, L, K)
        """

        calibrated = site_logits * self.logit_scale + self.logit_bias
        donor_logit = calibrated[..., 0]
        acceptor_logit = calibrated[..., 1]

        B, L, _ = site_logits.shape

        emissions = site_logits.new_full(
            (B, L, self.num_labels),
            fill_value=0.0,
        )

        if self.label_mode == "3":
            skip = LABELS_3["skip"]
            donor = LABELS_3["donor"]
            acceptor = LABELS_3["acceptor"]

            emissions[..., skip] = self.skip_bias[0]
            emissions[..., donor] = donor_logit
            emissions[..., acceptor] = acceptor_logit

        else:
            S_D = LABELS_4["S_D"]
            D = LABELS_4["D"]
            S_A = LABELS_4["S_A"]
            A = LABELS_4["A"]

            emissions[..., S_D] = self.skip_bias[0]
            emissions[..., D] = donor_logit
            emissions[..., S_A] = self.skip_bias[1]
            emissions[..., A] = acceptor_logit

        if self.config.enforce_emission_constraints:
            emissions = self.apply_emission_constraints(
                emissions=emissions,
                candidate_type_ids=candidate_type_ids,
            )

        return emissions

    def apply_emission_constraints(
        self,
        emissions: torch.Tensor,
        candidate_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forbid impossible candidate-type/state combinations

        Donor candidate:
            3-state allowed: skip, donor
            4-state allowed: S_D, D, S_A

        Acceptor candidate:
            3-state allowed: skip, acceptor
            4-state allowed: S_D, S_A, A
        """

        donor_candidate_mask = candidate_type_ids == DONOR_CANDIDATE
        acceptor_candidate_mask = candidate_type_ids == ACCEPTOR_CANDIDATE

        emissions = emissions.clone()

        if self.label_mode == "3":
            donor_label = LABELS_3["donor"]
            acceptor_label = LABELS_3["acceptor"]

            # Acceptor candidates cannot be selected as donors.
            emissions[acceptor_candidate_mask, donor_label] = NEG_INF

            # Donor candidates cannot be selected as acceptors.
            emissions[donor_candidate_mask, acceptor_label] = NEG_INF

        else:
            D = LABELS_4["D"]
            A = LABELS_4["A"]

            # Acceptor candidates cannot be selected as D.
            emissions[acceptor_candidate_mask, D] = NEG_INF

            # Donor candidates cannot be selected as A.
            emissions[donor_candidate_mask, A] = NEG_INF

        return emissions

    def forward(
        self,
        site_logits: torch.Tensor,
        candidate_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            site_logits:
                Raw NT-transformer logits, shape (B, L, 2).

            candidate_type_ids:
                0 = donor candidate, 1 = acceptor candidate, shape (B, L).

            attention_mask:
                1/True for valid candidates, 0/False for padding, shape (B, L).

            labels:
                Optional CRF ground-truth label ids, shape (B, L).
                Padding positions should be -1.

        Returns:
            dict with loss during training or decoded predictions during inference.
        """

        emissions = self.build_emissions(
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
        )

        return self.crf(
            emissions=emissions,
            attention_mask=attention_mask,
            labels=labels,
        )
