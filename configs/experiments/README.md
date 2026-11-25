# 实验配置

`abmil.yaml` 是标准 ABMIL baseline 的配置，包含输入维度、注意力维度、优化器、训练轮数、早停和随机种子。

- `abmil_gated.yaml`：TorchMIL Gated ABMIL。
- `mean_pooling.yaml`：masked mean pooling 对照模型。
- `clam_sb.yaml`：TorchMIL 官方 CLAM-SB。
- `dsmil.yaml`：TorchMIL 官方 DSMIL。
- `transmil.yaml`：TorchMIL 官方 TransMIL。

所有配置都使用 `split_seed: 1234` 对应的固定划分；`training_seed` 可以由 `scripts/train.py --seed` 覆盖。新实验输出不要写入历史 `outputs/abmil_seed1234/`。
