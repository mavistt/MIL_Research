from __future__ import annotations

import importlib.metadata
import inspect

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nystrom_attention")

from src.models.transmil.nystrom_compat import NystromAttention  # noqa: E402


def _reference_class(*, tail: bool):
    import nystrom_attention.nystrom_attention as official_module

    source = inspect.getsource(official_module)
    # PyTorch 2.3 compatibility only: remove in-place writes on views.
    source = source.replace("q *= self.scale", "q = q * self.scale")
    source = source.replace("out += self.res_conv(v)", "out = out + self.res_conv(v)")
    if tail:
        # Project/TransMIL padding correction: official project behavior keeps
        # the final n positions after left padding.
        source = source.replace("out = out[:, :n]", "out = out[:, -n:]")
    namespace: dict[str, object] = {}
    exec(compile(source, "nystrom_attention_0_0_9_reference.py", "exec"), namespace)
    return namespace["NystromAttention"]


@pytest.mark.parametrize("length", [145, 256, 8101, 44522])
def test_compat_matches_nystrom_0_0_9_tail_reference_forward_and_backward(length: int):
    assert importlib.metadata.version("nystrom-attention") == "0.0.9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Reference = _reference_class(tail=True)
    torch.manual_seed(2025)
    reference = Reference(dim=32, dim_head=8, heads=4, num_landmarks=256, pinv_iterations=6, residual=True, dropout=0.0).to(device)
    compat = NystromAttention(dim=32, dim_head=8, heads=4, num_landmarks=256, pinv_iterations=6, residual=True, dropout=0.0).to(device)
    compat.load_state_dict(reference.state_dict())
    torch.manual_seed(9000 + length)
    x_reference = torch.randn(1, length, 32, device=device, requires_grad=True)
    x_compat = x_reference.detach().clone().requires_grad_(True)

    output_reference = reference(x_reference)
    output_compat = compat(x_compat)
    torch.testing.assert_close(output_compat, output_reference, rtol=1e-5, atol=1e-6)
    output_reference.square().mean().backward()
    output_compat.square().mean().backward()
    torch.testing.assert_close(x_compat.grad, x_reference.grad, rtol=1e-5, atol=1e-6)
    for (_, reference_parameter), (_, compat_parameter) in zip(reference.named_parameters(), compat.named_parameters()):
        if reference_parameter.grad is None:
            assert compat_parameter.grad is None
        else:
            torch.testing.assert_close(compat_parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", [145, 8101, 44522])
def test_project_tail_correction_is_not_the_unpatched_left_pad_slice(length: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HeadReference = _reference_class(tail=False)
    TailReference = _reference_class(tail=True)
    torch.manual_seed(2026)
    head = HeadReference(dim=32, dim_head=8, heads=4, num_landmarks=256, pinv_iterations=6, residual=True, dropout=0.0).to(device)
    tail = TailReference(dim=32, dim_head=8, heads=4, num_landmarks=256, pinv_iterations=6, residual=True, dropout=0.0).to(device)
    tail.load_state_dict(head.state_dict())
    x = torch.randn(1, length, 32, device=device)
    head_output = head(x)
    tail_output = tail(x)
    assert not torch.allclose(head_output, tail_output, rtol=1e-5, atol=1e-6)
