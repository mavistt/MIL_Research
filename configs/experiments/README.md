# 实验配置

`abmil.yaml` 是标准 ABMIL baseline 的配置，包含输入维度、注意力维度、优化器、训练轮数、早停和随机种子。

- `abmil_gated.yaml`：TorchMIL Gated ABMIL。
- `mean_pooling.yaml`：masked mean pooling 对照模型。
- `clam_sb.yaml`：作者官方 CLAM-SB 的严格适配配置（200 epochs、官方 bag/instance loss）。
- `dsmil.yaml`：作者官方 DSMIL 的严格适配配置（50 epochs、官方 max-instance/bag loss）。
- `transmil.yaml`：作者官方 TransMIL 的严格适配配置（Lookahead-RAdam、两类 CE）。

所有配置都使用 `split_seed: 1234` 对应的固定划分；`training_seed` 可以由 `scripts/train.py --seed` 覆盖。新实验输出不要写入历史 `outputs/abmil_seed1234/`。
