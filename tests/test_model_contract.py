from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchmil")

from src.models import build_model  # noqa: E402


def _config(name: str) -> dict:
    model = {"name": name, "input_dim": 16, "att_dim": 16, "dropout": 0.0}
    if name == "abmil":
        model["gated"] = False
    if name == "clam_sb":
        model.update({"k_sample": 4, "inst_loss_name": "SmoothTop1SVM", "gated": False})
    if name == "transmil":
        model.update({"n_layers": 2, "n_heads": 4, "n_landmarks": 4, "pinv_iterations": 2, "use_mlp": False})
    return {"model": model}


@pytest.mark.parametrize("name", ["abmil", "gated_abmil", "mean_pooling", "clam_sb", "dsmil", "transmil"])
def test_all_baselines_follow_forward_and_loss_contract(name: str):
    model = build_model(_config(name))
    x = torch.randn(1, 16, 16)
    mask = torch.ones(1, 16, dtype=torch.bool)
    y = torch.tensor([1.0])
    logits = model(x, mask)
    assert logits.shape == (1,)
    logits, losses = model.compute_loss(y=y, x=x, mask=mask)
    assert logits.shape == (1,)
    assert losses
    loss = sum(losses.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
