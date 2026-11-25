# TransMIL

调用 TorchMIL 1.0.2 的官方 TransMIL，包括 Nyström Transformer 和 PPEG。由于官方 forward 没有外部 mask 参数，项目要求 batch size 1，并显式拒绝外部 padding。

- 配置：`configs/experiments/transmil.yaml`
- 预留输出：`outputs/camelyon16_uni/transmil/`
- 预留日志：`logs/transmil/`
