from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchmil")
pytest.importorskip("nystrom_attention")

from src.models import build_model  # noqa: E402


def _config(name: str) -> dict:
    model = {"name": name, "input_dim": 1024, "att_dim": 128, "dropout": 0.0}
    if name == "abmil":
        model["gated"] = False
    if name == "gated_abmil":
        model["dropout"] = 0.0
    if name == "clam_sb":
        model.update({"model_size": "small", "k_sample": 4, "n_classes": 2, "subtyping": False, "gated": True, "bag_weight": 0.7, "dropout": 0.25})
    if name == "dsmil":
        model.update({"output_class": 1, "nonlinear": True, "passing_v": False, "dropout_node": 0.0})
    if name == "transmil":
        model.update({"n_classes": 2})
    return {"model": model}


@pytest.mark.parametrize("name", ["abmil", "gated_abmil", "mean_pooling", "clam_sb", "dsmil", "transmil"])
def test_all_baselines_follow_forward_and_official_loss_contract(name: str):
    model = build_model(_config(name))
    x = torch.randn(1, 64, 1024)
    mask = torch.ones(1, 64, dtype=torch.bool)
    y = torch.tensor([1.0])
    logits = model(x, mask)
    assert logits.shape == (1,)
    logits, loss, details = model.compute_training_loss(y=y, x=x, mask=mask)
    assert logits.shape == (1,)
    assert loss.ndim == 0
    assert details
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
