from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import two_class_positive_logit, unpad_single_bag
from .official_model_clam import CLAM_SB


class OfficialCLAMSB(nn.Module):
    """Adapter around the official CLAM single-branch implementation."""

    def __init__(
        self,
        input_dim: int,
        model_size: str = "small",
        gated: bool = True,
        dropout: float = 0.25,
        k_sample: int = 8,
        n_classes: int = 2,
        subtyping: bool = False,
        bag_weight: float = 0.7,
    ) -> None:
        super().__init__()
        if n_classes != 2:
            raise ValueError("This CAMELYON16 adapter expects two bag classes")
        self.model = CLAM_SB(
            gate=gated,
            size_arg=model_size,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=nn.CrossEntropyLoss(),
            subtyping=subtyping,
            embed_dim=input_dim,
        )
        self.bag_weight = float(bag_weight)

    def _official_forward(self, x: torch.Tensor, mask: torch.Tensor, *, instance_eval: bool, y: torch.Tensor | None = None):
        h = unpad_single_bag(x, mask)
        label = None if y is None else y.reshape(-1).long()[0]
        return self.model(h, label=label, instance_eval=instance_eval)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, _, _, _, _ = self._official_forward(x, mask, instance_eval=False)
        return two_class_positive_logit(logits)

    def compute_training_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, _, _, _, results = self._official_forward(x, mask, instance_eval=True, y=y)
        bag_loss = F.cross_entropy(logits, y.reshape(-1).long())
        instance_loss = results.get("instance_loss", torch.zeros((), device=logits.device))
        if not isinstance(instance_loss, torch.Tensor):
            instance_loss = torch.tensor(float(instance_loss), device=logits.device)
        instance_loss = instance_loss.reshape(())
        total = self.bag_weight * bag_loss + (1.0 - self.bag_weight) * instance_loss
        return two_class_positive_logit(logits), total, {
            "bag_loss": bag_loss,
            "instance_loss": instance_loss,
            "total_loss": total,
        }

    def compute_validation_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, _, _, _, _ = self._official_forward(x, mask, instance_eval=False)
        bag_loss = F.cross_entropy(logits, y.reshape(-1).long())
        return two_class_positive_logit(logits), bag_loss, {"bag_loss": bag_loss}

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        logits, total, details = self.compute_training_loss(y, x, mask)
        return logits, {"total_loss": total, **{k: v for k, v in details.items() if k != "total_loss"}}
