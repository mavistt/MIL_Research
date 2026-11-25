# 已完成 Baseline 结果

## 实现状态

项目目前有 6 个 baseline：

| baseline | 代码 | 完整实验 |
|---|---|---|
| 标准 ABMIL | 已实现 | 已完成 4 个新 seed + 历史 seed1234 |
| Gated ABMIL | 已实现 | 已完成 3 个 seed |
| Mean Pooling | 已实现并 smoke test | 尚未完整训练 |
| CLAM-SB | 已实现并 smoke test | 尚未完整训练 |
| DSMIL | 已实现并 smoke test | 尚未完整训练 |
| TransMIL | 已实现并 smoke test | 尚未完整训练 |

## 当前可比较结果

标准 ABMIL 的五次汇总（新 seed 0–3 加历史 seed1234）为 AUROC `0.971837 ± 0.014991`、AUPRC `0.967828 ± 0.017104`。

Gated ABMIL 的三次汇总为 AUROC `0.976276 ± 0.022678`、AUPRC `0.973405 ± 0.021844`。

由于两组 seed 数量不同，这里只记录结果，不把小幅差异直接解释为稳定的模型优势。后续正式比较时应统一 seed 数量，并同时查看 AUROC、AUPRC、Recall/Sensitivity、Specificity 和 F1。
