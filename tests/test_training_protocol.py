from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from scripts.train import selection_score  # noqa: E402
from src.models import build_model  # noqa: E402


def test_clam_training_and_validation_losses_are_separate():
    config = {
        "model": {
            "name": "clam_sb",
            "input_dim": 1024,
            "model_size": "small",
            "gated": True,
            "dropout": 0.25,
            "k_sample": 4,
            "n_classes": 2,
            "subtyping": False,
            "bag_weight": 0.7,
        }
    }
    model = build_model(config)
    x = torch.randn(1, 64, 1024)
    mask = torch.ones(1, 64, dtype=torch.bool)
    y = torch.tensor([1.0])
    _, training_loss, training_details = model.compute_training_loss(y, x, mask)
    expected = 0.7 * training_details["bag_loss"] + 0.3 * training_details["instance_loss"]
    torch.testing.assert_close(training_loss, expected)
    _, validation_loss, validation_details = model.compute_validation_loss(y, x, mask)
    torch.testing.assert_close(validation_loss, validation_details["bag_loss"])
    assert not torch.equal(training_loss.detach(), validation_loss.detach())


def test_project_checkpoint_selection_uses_validation_auroc():
    score, lower_is_better = selection_score(0.8, {"auroc": 0.73, "auprc": 0.70}, "val_auroc")
    assert score == pytest.approx(0.73)
    assert lower_is_better is False
