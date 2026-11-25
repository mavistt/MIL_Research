from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import unpad_single_bag
from .official_dsmil import BClassifier, FCLayer, MILNet


class OfficialDSMIL(nn.Module):
    """Official DSMIL FCLayer/BClassifier/MILNet adapter for WSI bags."""

    def __init__(
        self,
        input_dim: int,
        output_class: int = 1,
        dropout_node: float = 0.0,
        nonlinear: bool = True,
        passing_v: bool = False,
    ) -> None:
        super().__init__()
        if output_class != 1:
            raise ValueError("The CAMELYON16 binary adapter uses output_class=1")
        self.model = MILNet(
            FCLayer(input_dim, output_class),
            BClassifier(
                input_size=input_dim,
                output_class=output_class,
                dropout_v=dropout_node,
                nonlinear=nonlinear,
                passing_v=passing_v,
            ),
        )

    def _official_forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.model(unpad_single_bag(x, mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, bag_prediction, _, _ = self._official_forward(x, mask)
        return bag_prediction.reshape(-1)

    def compute_training_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        instance_prediction, bag_prediction, _, _ = self._official_forward(x, mask)
        target = y.reshape(-1).float()
        bag_logits = bag_prediction.reshape(-1)
        max_instance_logits = instance_prediction.reshape(-1).max().reshape(1)
        bag_loss = F.binary_cross_entropy_with_logits(bag_logits, target)
        max_loss = F.binary_cross_entropy_with_logits(max_instance_logits, target)
        total = 0.5 * bag_loss + 0.5 * max_loss
        return bag_logits, total, {"bag_loss": bag_loss, "max_loss": max_loss, "total_loss": total}

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, total, details = self.compute_training_loss(y, x, mask)
        return logits, {"total_loss": total, **{k: v for k, v in details.items() if k != "total_loss"}}
