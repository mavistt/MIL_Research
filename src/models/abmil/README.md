# ABMIL

这里实现 TorchMIL 1.0.2 的标准 ABMIL。Gated ABMIL 的独立入口位于 `src/models/gated_abmil/`，底层复用本目录适配器，不复制模型代码。

- 标准 ABMIL 配置：`configs/experiments/abmil.yaml`
- Gated ABMIL 配置：`configs/experiments/abmil_gated.yaml`
- 标准 ABMIL 输出：`outputs/camelyon16_uni/abmil_ungated/`
- Gated ABMIL 输出：`outputs/camelyon16_uni/gated_abmil/`
