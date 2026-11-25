# Gated ABMIL

Gated ABMIL 是 ABMIL 的门控注意力变体。它复用 `src/models/abmil/` 中的官方 TorchMIL 适配器，通过 `gated: true` 启用门控分支，不复制一份相同模型代码。

- 配置：`configs/experiments/abmil_gated.yaml`
- 输出：`outputs/camelyon16_uni/gated_abmil/`
- 日志：`logs/gated_abmil/`
