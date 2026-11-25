from __future__ import annotations

import torch
import torch.nn as nn

from ..common import unwrap_torchmil_logits


class TorchMILTransMIL(nn.Module):
    """Official TorchMIL TransMIL, preserving its sequence-squaring behavior."""

    def __init__(
        self,
        input_dim: int,
        att_dim: int,
        n_layers: int,
        n_heads: int,
        n_landmarks: int | None,
        pinv_iterations: int,
        dropout: float,
        use_mlp: bool,
    ) -> None:
        super().__init__()
        from torchmil.models import TransMIL

        self.model = TransMIL(
            in_shape=(input_dim,),
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
            criterion=nn.BCEWithLogitsLoss(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.all()):
            raise ValueError("TransMIL requires an unpadded bag; use batch_size=1")
        return unwrap_torchmil_logits(self.model(x))

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits = self(x, mask)
        return logits, {"bag_loss": torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())}
