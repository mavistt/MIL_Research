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
