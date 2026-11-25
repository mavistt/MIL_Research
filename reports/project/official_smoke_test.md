# 官方 baseline 真实 feature smoke test

运行脚本：`scripts/official_smoke_test.py`  
设备：香港服务器 GPU6（进程内显示为 `cuda:0`）

## Bag 覆盖

| 级别 | bag | label | instances |
|---|---|---:|---:|
| small | `normal_042` | 0 | 140 |
| medium | `tumor_062` | 1 | 7,984 |
| large / max | `test_069` | 1 | 44,402 |

## 结果

所有模型在三种 bag 长度上均成功 forward，没有截断或 OOM。这里的 score 是用于后续 ROC/PR 计算的 scalar logit，不是最终概率。

| 模型 | small 峰值显存 MB | medium 峰值显存 MB | large 峰值显存 MB |
|---|---:|---:|---:|
| Mean Pooling | 8.68 | 134.17 | 418.35 |
| ABMIL | 10.25 | 47.62 | 226.03 |
| Gated ABMIL | 11.32 | 52.71 | 248.72 |
| CLAM-SB | 14.29 | 113.05 | 577.31 |
| DSMIL | 11.44 | 80.41 | 402.42 |
| TransMIL | 49.13 | 536.71 | 2,814.28 |

TransMIL 对 `test_069` 按作者实现补齐 119 个位置到平方序列；这属于官方 sequence-squaring 行为，不是项目截断。完整耗时、score 和补方数量保存在同目录的 JSON 文件中。
