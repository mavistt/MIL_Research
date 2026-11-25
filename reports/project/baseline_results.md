# 已完成 Baseline 结果

## 实验状态

项目共包含 6 个 Baseline，正式训练均使用固定 split `seed1234`、UNI 1024 维 feature、训练 seed `0/1/2` 和测试阈值 `0.5`。

| Baseline | 正式实验状态 | seed |
|---|---|---:|
| 标准 ABMIL | 已完成；另保留历史 seed1234，形成独立五 Seed 汇总 | 0, 1, 2 |
| Gated ABMIL | 已完成 | 0, 1, 2 |
| Mean Pooling | 已完成 | 0, 1, 2 |
| CLAM-SB | 已完成 | 0, 1, 2 |
| DSMIL | 已完成 | 0, 1, 2 |
| TransMIL | 已完成 | 0, 1, 2 |

## Common seed 0/1/2 测试集结果

下表为 Mean ± Sample Std（ddof=1），每个指标有效 seed 数量均为 3。完整 8 项指标见 [`all_baselines_comparison.md`](all_baselines_comparison.md) 和 [`all_baselines_comparison.csv`](all_baselines_comparison.csv)。

| Baseline | AUROC | AUPRC | Accuracy | F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| ABMIL | 0.977381 ± 0.017132 | 0.977558 ± 0.014799 | 0.950904 ± 0.008951 | 0.930797 ± 0.013371 | 0.935374 ± 0.011783 |
| Gated ABMIL | 0.976276 ± 0.022678 | 0.973405 ± 0.021844 | 0.943152 ± 0.016137 | 0.920743 ± 0.023130 | 0.929124 ± 0.018595 |
| Mean Pooling | 0.542857 ± 0.024531 | 0.517908 ± 0.016304 | 0.640827 ± 0.016137 | 0.279732 ± 0.029888 | 0.552253 ± 0.016095 |
| CLAM-SB | 0.984779 ± 0.008493 | 0.981782 ± 0.007359 | 0.935401 ± 0.024919 | 0.915583 ± 0.023847 | 0.929464 ± 0.008898 |
| DSMIL | 0.962585 ± 0.015693 | 0.959662 ± 0.011634 | 0.919897 ± 0.034955 | 0.889765 ± 0.050999 | 0.907738 ± 0.040913 |
| TransMIL | 0.958759 ± 0.047688 | 0.954957 ± 0.052745 | 0.914729 ± 0.055900 | 0.880271 ± 0.076974 | 0.896981 ± 0.058398 |

## 独立历史 ABMIL 五 Seed 结果

标准 ABMIL 的新 seed `0–3` 加历史 seed `1234` 汇总为 AUROC `0.971837 ± 0.014991`、AUPRC `0.967828 ± 0.017104`。该五 Seed 汇总保留在 [`reports/abmil_ungated/summary.md`](../abmil_ungated/summary.md)，不混入上面的 common-seed 主比较。

## 计算开销和可追溯性

正式 CLAM、DSMIL、TransMIL 的参数量、实际 epoch、最佳 checkpoint epoch、总训练时间、峰值显存、单 WSI 测试模型计算时间和代码 commit 见 [`all_baselines_compute.csv`](all_baselines_compute.csv)。历史 ABMIL、Gated ABMIL 和 Mean Pooling 运行时尚未生成本轮 metadata，因此对应字段标记为 `legacy/unknown`。

所有模型的测试 WSI ID、label 和 split 文件已通过一致性检查；固定 split SHA256 为 `dd07e2e746c88effb7b74b3e71df3257a986b936a3a166717042a7e9ebe7710e`。
