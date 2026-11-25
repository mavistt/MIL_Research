from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from ..common import unwrap_torchmil_logits


class TorchMILABMIL(nn.Module):
    """Thin wrapper around TorchMIL's standard or gated ABMIL."""

    def __init__(self, input_dim: int, att_dim: int, gated: bool, dropout: float) -> None:
        super().__init__()
        try:
            from torchmil.models import ABMIL
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torchmil is required; install requirements.txt first") from exc
        criterion = nn.BCEWithLogitsLoss()
        kwargs = {"in_shape": (input_dim,), "att_dim": att_dim, "gated": gated, "criterion": criterion}
        if "dropout" in inspect.signature(ABMIL).parameters:
            kwargs["dropout"] = dropout
        self.model = ABMIL(**kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return unwrap_torchmil_logits(self.model(x, mask=mask))

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits = self(x, mask)
        return logits, {"bag_loss": torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())}

    def compute_training_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, losses = self.compute_loss(y, x, mask)
        return logits, losses["bag_loss"], losses

    def compute_validation_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits = self(x, mask)
        bag_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        return logits, bag_loss, {"bag_loss": bag_loss}
