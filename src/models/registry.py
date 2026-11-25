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
        from .clam_sb import TorchMILCLAMSB

        return TorchMILCLAMSB(
            input_dim=input_dim,
            att_dim=att_dim,
            k_sample=int(model_cfg.get("k_sample", 8)),
            gated=bool(model_cfg.get("gated", False)),
            inst_loss_name=str(model_cfg.get("inst_loss_name", "SmoothTop1SVM")),
        )
    if name == "dsmil":
        from .dsmil import TorchMILDSMIL

        return TorchMILDSMIL(
            input_dim=input_dim,
            att_dim=att_dim,
            nonlinear_q=bool(model_cfg.get("nonlinear_q", False)),
            nonlinear_v=bool(model_cfg.get("nonlinear_v", False)),
            dropout=dropout,
        )
    if name == "transmil":
        from .transmil import TorchMILTransMIL

        return TorchMILTransMIL(
            input_dim=input_dim,
            att_dim=att_dim,
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            n_landmarks=model_cfg.get("n_landmarks"),
            pinv_iterations=int(model_cfg.get("pinv_iterations", 6)),
            dropout=dropout,
            use_mlp=bool(model_cfg.get("use_mlp", False)),
        )
    raise ValueError(f"Unsupported model={name}; expected abmil, mean_pooling, clam_sb, dsmil or transmil")

