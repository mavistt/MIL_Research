from __future__ import annotations

import torch


def unwrap_torchmil_logits(result: object) -> torch.Tensor:
    """Normalize TorchMIL 1.0.2 forward return values to [batch]."""
    if isinstance(result, dict):
        for key in ("Y_hat", "logits", "y_hat", "output"):
            if key in result:
                result = result[key]
                break
    if isinstance(result, (tuple, list)):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"Unexpected TorchMIL output type: {type(result)!r}")
    return result.reshape(-1)


def sum_loss_terms(loss_terms: object) -> torch.Tensor:
    """Sum official TorchMIL loss terms while preserving gradients."""
    if isinstance(loss_terms, torch.Tensor):
        return loss_terms
    if not isinstance(loss_terms, dict) or not loss_terms:
        raise TypeError(f"Unexpected TorchMIL loss return value: {type(loss_terms)!r}")
    values = [value for value in loss_terms.values() if isinstance(value, torch.Tensor)]
    if not values:
        raise TypeError("TorchMIL returned no tensor loss terms")
    return torch.stack([value.reshape(()) for value in values]).sum()


def unpad_single_bag(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return one variable-length bag without collate padding."""
    if x.ndim != 3 or mask.ndim != 2 or x.shape[:2] != mask.shape:
        raise ValueError(f"Expected X [B,N,D] and mask [B,N], got {tuple(x.shape)} and {tuple(mask.shape)}")
    if x.shape[0] != 1:
        raise ValueError("The official WSI implementations are adapted with batch_size=1")
    valid = x[0][mask[0]]
    if valid.shape[0] == 0:
        raise ValueError("A bag must contain at least one valid instance")
    return valid


def two_class_positive_logit(logits: torch.Tensor) -> torch.Tensor:
    """Convert official two-class logits to the scalar score used by metrics."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits [B,2], got {tuple(logits.shape)}")
    return (logits[:, 1] - logits[:, 0]).reshape(-1)
