from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolingMIL(nn.Module):
    """Mean-pool valid patch features, then apply a linear bag classifier."""

    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled)).reshape(-1)

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits = self(x, mask)
        return logits, {"bag_loss": torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())}

    def compute_training_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, losses = self.compute_loss(y, x, mask)
        return logits, losses["bag_loss"], losses
