from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.models.dsmil.model import OfficialDSMIL  # noqa: E402


def test_dsmil_official_sparse_init_is_orthogonal_and_zero_bias():
    torch.manual_seed(1234)
    model = OfficialDSMIL(input_dim=1024, output_class=1, nonlinear=True, passing_v=False)
    checked = 0
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            continue
        checked += 1
        weight = module.weight.detach().reshape(module.weight.shape[0], -1)
        if weight.shape[0] <= weight.shape[1]:
            gram = weight @ weight.t()
            identity = torch.eye(weight.shape[0], dtype=weight.dtype)
        else:
            gram = weight.t() @ weight
            identity = torch.eye(weight.shape[1], dtype=weight.dtype)
        torch.testing.assert_close(gram, identity, rtol=1e-5, atol=1e-5)
        if module.bias is not None:
            assert torch.equal(module.bias.detach(), torch.zeros_like(module.bias.detach()))
    assert checked >= 4
