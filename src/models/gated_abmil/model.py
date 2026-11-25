from __future__ import annotations

from ..abmil.model import TorchMILABMIL


class GatedABMIL(TorchMILABMIL):
    """独立的 Gated ABMIL 入口，底层复用标准 ABMIL 适配器。"""

    def __init__(self, input_dim: int, att_dim: int, dropout: float) -> None:
        super().__init__(input_dim=input_dim, att_dim=att_dim, gated=True, dropout=dropout)
