from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    if name != "abmil":
        raise ValueError(f"Only the first baseline is implemented; got model={name}")
    from .abmil import TorchMILABMIL

    return TorchMILABMIL(
        input_dim=int(model_cfg["input_dim"]),
        att_dim=int(model_cfg["att_dim"]),
        gated=bool(model_cfg.get("gated", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

