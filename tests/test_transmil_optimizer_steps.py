from __future__ import annotations

import torch

from scripts.train import build_optimizer
from src.models import build_model


def test_transmil_runs_seven_finite_optimizer_steps_and_updates_lookahead():
    config = {
        "model": {"name": "transmil", "input_dim": 1024, "n_classes": 2},
        "training": {
            "learning_rate": 2e-4,
            "weight_decay": 1e-5,
            "betas": [0.9, 0.999],
            "optimizer": "lookahead_radam",
            "lookahead_alpha": 0.5,
            "lookahead_k": 6,
        },
    }
    torch.manual_seed(2025)
    model = build_model(config)
    optimizer = build_optimizer(model, config["training"])
    x = torch.randn(1, 64, 1024)
    mask = torch.ones(1, 64, dtype=torch.bool)
    y = torch.tensor([1.0])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    for _ in range(7):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model.compute_training_loss(y=y, x=x, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert changed
    assert all("slow_buffer" in state for state in optimizer.state.values())
