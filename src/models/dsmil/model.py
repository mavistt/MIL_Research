from __future__ import annotations

import torch
import torch.nn as nn

from ..common import unwrap_torchmil_logits


class TorchMILDSMIL(nn.Module):
    """Official TorchMIL DSMIL with critical-instance and bag losses."""

    def __init__(self, input_dim: int, att_dim: int, nonlinear_q: bool, nonlinear_v: bool, dropout: float) -> None:
        super().__init__()
        from torchmil.models import DSMIL

        self.model = DSMIL(
            in_shape=(input_dim,),
            att_dim=att_dim,
            nonlinear_q=nonlinear_q,
            nonlinear_v=nonlinear_v,
            dropout=dropout,
            criterion=nn.BCEWithLogitsLoss(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return unwrap_torchmil_logits(self.model(x, mask=mask))

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, losses = self.model.compute_loss(Y=y, X=x, mask=mask)
        return unwrap_torchmil_logits(logits), losses
