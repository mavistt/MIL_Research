from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    input_dim = int(model_cfg["input_dim"])
    att_dim = int(model_cfg.get("att_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.0))
    if name == "abmil":
        from .abmil import TorchMILABMIL

        return TorchMILABMIL(input_dim, att_dim, bool(model_cfg.get("gated", False)), dropout)
    if name == "gated_abmil":
        from .gated_abmil import GatedABMIL

        return GatedABMIL(input_dim, att_dim, dropout)
    if name in {"mean", "mean_pooling", "mean_pool"}:
        from .mean_pooling import MeanPoolingMIL

        return MeanPoolingMIL(input_dim, dropout)
    if name in {"clam", "clam_sb"}:
        from .clam_sb import OfficialCLAMSB

        return OfficialCLAMSB(
            input_dim=input_dim,
            model_size=str(model_cfg.get("model_size", "small")),
            k_sample=int(model_cfg.get("k_sample", 8)),
            gated=bool(model_cfg.get("gated", True)),
            dropout=dropout,
            n_classes=int(model_cfg.get("n_classes", 2)),
            subtyping=bool(model_cfg.get("subtyping", False)),
            bag_weight=float(model_cfg.get("bag_weight", 0.7)),
        )
    if name == "dsmil":
        from .dsmil import OfficialDSMIL

        return OfficialDSMIL(
            input_dim=input_dim,
            output_class=int(model_cfg.get("output_class", 1)),
            dropout_node=float(model_cfg.get("dropout_node", 0.0)),
            nonlinear=bool(model_cfg.get("nonlinear", True)),
            passing_v=bool(model_cfg.get("passing_v", False)),
        )
    if name == "transmil":
        from .transmil import OfficialTransMIL

        return OfficialTransMIL(
            input_dim=input_dim,
            n_classes=int(model_cfg.get("n_classes", 2)),
        )
    raise ValueError(f"Unsupported model={name}; expected abmil, mean_pooling, clam_sb, dsmil or transmil")

