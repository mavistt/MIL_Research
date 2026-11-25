from __future__ import annotations

import torch
import torch.nn as nn

from ..common import unwrap_torchmil_logits


class TorchMILCLAMSB(nn.Module):
    """Official TorchMIL CLAM-SB with its instance-level clustering loss."""

    def __init__(self, input_dim: int, att_dim: int, k_sample: int, gated: bool, inst_loss_name: str) -> None:
        super().__init__()
        from torchmil.models import CLAM_SB

        self.model = CLAM_SB(
            in_shape=(input_dim,),
            att_dim=att_dim,
            k_sample=k_sample,
            gated=gated,
            inst_loss_name=inst_loss_name,
            criterion=nn.BCEWithLogitsLoss(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return unwrap_torchmil_logits(self.model(x, mask=mask))

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, losses = self.model.compute_loss(Y=y, X=x, mask=mask)
        return unwrap_torchmil_logits(logits), losses
