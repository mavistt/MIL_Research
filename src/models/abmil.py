from __future__ import annotations

import inspect

import torch
import torch.nn as nn


class TorchMILABMIL(nn.Module):
    """Thin compatibility wrapper around TorchMIL's standard ABMIL."""

    def __init__(self, input_dim: int, att_dim: int, gated: bool, dropout: float) -> None:
        super().__init__()
        try:
            from torchmil.models import ABMIL
        except Exception as exc:  # pragma: no cover - exercised only before install
            raise RuntimeError("torchmil is required; install requirements.txt first") from exc
        criterion = nn.BCEWithLogitsLoss()
        kwargs = {"in_shape": (input_dim,), "att_dim": att_dim, "gated": gated, "criterion": criterion}
        if "dropout" in inspect.signature(ABMIL).parameters:
            kwargs["dropout"] = dropout
        self.model = ABMIL(**kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = self.model(x, mask=mask)
        if isinstance(result, dict):
            for key in ("Y_hat", "logits", "y_hat", "output"):
                if key in result:
                    result = result[key]
                    break
        if isinstance(result, (tuple, list)):
            result = result[0]
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Unexpected ABMIL output type: {type(result)!r}")
        return result.reshape(-1)

