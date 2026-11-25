"""Minimal PyTorch 2.3 compatibility copy of nystrom-attention 0.0.9.

The upstream 0.0.9 implementation performs ``q *= scale`` on a view returned
by ``chunk``. PyTorch 2.3 rejects that in-place autograd operation. The
attention equations and default parameters remain unchanged; only the
in-place operations are made out-of-place.
"""

from math import ceil

import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, reduce


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))
    identity = rearrange(torch.eye(x.shape[-1], device=device), "i j -> () i j")
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * identity - (xz @ (15 * identity - (xz @ (7 * identity - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, residual_conv_kernel=33, eps=1e-8, dropout=0.0):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (residual_conv_kernel, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if mask is not None:
                mask = F.pad(mask, (padding, 0), value=False)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if mask is not None:
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
        q = q * self.scale
        l = ceil(n / m)
        landmark_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_eq, "sum", l=l)
        divisor = l
        if mask is not None:
            mask_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_sum[..., None] + eps
            mask_landmarks = mask_sum > 0
        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor
        eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(eq, q, k_landmarks)
        sim2 = einsum(eq, q_landmarks, k_landmarks)
        sim3 = einsum(eq, q_landmarks, k)
        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1 = sim1.masked_fill(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2 = sim2.masked_fill(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3 = sim3.masked_fill(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out = out + self.res_conv(v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, :n]
        if return_attn:
            return out, attn1 @ attn2_inv @ attn3
        return out
