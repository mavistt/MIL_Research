# TransMIL

调用固定 commit 的作者官方 TransMIL，包括 512 维两层 Nyström Transformer、CLS token 和 PPEG。官方 forward 没有外部 mask 参数，项目只在进入官方实现前去除 collate padding，并保留官方序列补方，不截断 bag。`nystrom_compat.py` 是 PyTorch 2.3 的最小 out-of-place 兼容层。

- 配置：`configs/experiments/transmil.yaml`
- 预留输出：`outputs/camelyon16_uni/transmil/`
- 预留日志：`logs/transmil/`
