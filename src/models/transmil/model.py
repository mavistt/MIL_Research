from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import two_class_positive_logit, unpad_single_bag
from .official_transmil import TransMIL


class OfficialTransMIL(nn.Module):
    """Official TransMIL encoder/classifier with the paper's two-class CE head."""

    def __init__(self, input_dim: int, n_classes: int = 2, **_: object) -> None:
        super().__init__()
        if input_dim != 1024 or n_classes != 2:
            raise ValueError("The vendored Camelyon16 TransMIL implementation expects 1024-D features and 2 classes")
        self.model = TransMIL(n_classes=n_classes)

    def _official_forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.model(data=unpad_single_bag(x, mask).unsqueeze(0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        results = self._official_forward(x, mask)
        return two_class_positive_logit(results["logits"])

    def compute_training_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        results = self._official_forward(x, mask)
        labels = y.reshape(-1).long()
        bag_loss = F.cross_entropy(results["logits"], labels)
        return two_class_positive_logit(results["logits"]), bag_loss, {"bag_loss": bag_loss}

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, loss, details = self.compute_training_loss(y, x, mask)
        return logits, {"total_loss": loss, **details}
