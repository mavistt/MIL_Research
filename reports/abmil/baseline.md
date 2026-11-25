# ABMIL Baseline 结果

## 实验条件

- 数据集：CAMELYON16
- 特征：预提取 UNI patch embedding，1024 维
- 模型：标准 ABMIL，不是 SmABMIL
- 官方划分：270 train、129 test；train 内按 WSI 标签分层拆出 216 train 和 54 validation
- 随机种子：1234
- 训练输入：只使用 patch 特征 `X` 和 WSI 级标签 `Y`
- 未使用：原始 WSI、UNI 权重、ResNet 特征、`coords`、`y_inst`

## 测试结果

在固定阈值 0.5 下，官方 test 集 129 个 WSI 的结果为：

| 指标 | 数值 |
|---|---:|
| AUROC | 0.9699 |
| AUPRC | 0.9566 |
| Accuracy | 0.9147 |
| Precision | 0.9524 |
| Recall / Sensitivity | 0.8163 |
| Specificity | 0.9750 |
| F1 | 0.8791 |
| Balanced Accuracy | 0.8957 |

混淆矩阵为 `[[78, 2], [9, 40]]`，即 TN=78、FP=2、FN=9、TP=40。

## 如何查看

- `outputs/camelyon16_uni/abmil_ungated/seed_1234/metrics_test.json`：历史训练脚本生成的 test 指标。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/evaluated_recheck/`：如重新运行独立评估脚本，复核结果会放在这里。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/training_history.csv`：每个 epoch 的训练和验证曲线数据。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/predictions_test.csv`：每个 WSI 的标签、预测概率和预测类别。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/roc_curve.png`：ROC 曲线。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/pr_curve.png`：PR 曲线。
- `outputs/camelyon16_uni/abmil_ungated/seed_1234/confusion_matrix.png`：混淆矩阵图。

当前结果是单个随机种子的 baseline，不应直接视为最终稳定性能。后续应在相同划分上运行多个 seeds，并报告均值和标准差。
