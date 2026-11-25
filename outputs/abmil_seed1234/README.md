# ABMIL seed1234 实验

这是标准 ABMIL baseline 的完整输出，使用固定 `seed1234.csv` 划分。

- `config.yaml`：本次运行的配置。
- `split_used.csv`：本次运行使用的固定划分。
- `checkpoint_best.pt`：验证集选择的最佳模型。
- `training_history.csv`：每个 epoch 的训练/验证损失和指标。
- `metrics_val.json`、`metrics_test.json`：验证和测试指标。
- `predictions_test.csv`：WSI ID、标签、预测概率和预测类别。
- `roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`：测试可视化。

当前 test 结果：AUROC `0.9699`，AUPRC `0.9566`，Accuracy `0.9147`，F1 `0.8791`。
