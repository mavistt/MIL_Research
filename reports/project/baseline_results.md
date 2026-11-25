# 已完成 Baseline 结果

## 实现状态

项目目前有 6 个 baseline：

| baseline | 代码 | 完整实验 |
|---|---|---|
| 标准 ABMIL | 已实现 | 已完成 4 个新 seed + 历史 seed1234（协议已核对） |
| Gated ABMIL | 已实现 | 已完成 3 个 seed |
| Mean Pooling | 已实现并 smoke test | 已完成 3 个 seed |
| CLAM-SB | 已实现并 smoke test | 尚未完整训练 |
| DSMIL | 已实现并 smoke test | 尚未完整训练 |
| TransMIL | 已实现并 smoke test | 尚未完整训练 |

## 当前可比较结果

标准 ABMIL 的五次汇总（新 seed 0–3 加历史 seed1234）为 AUROC `0.971837 ± 0.014991`、AUPRC `0.967828 ± 0.017104`。已对比保存的 config 和 split：模型、30 epoch、Adam、学习率、阈值一致，`selection_metric: auroc` 与当前 `val_auroc` 的选择逻辑等价，因此可以作为同条件第五次记录。

Gated ABMIL 的三次汇总为 AUROC `0.976276 ± 0.022678`、AUPRC `0.973405 ± 0.021844`。

Mean Pooling 的三个 seed 汇总为 AUROC `0.542857 ± 0.024531`、AUPRC `0.517908 ± 0.016304`。它是项目级聚合对照，不属于论文作者模型。

本轮旧的 CLAM-SB seed 0 训练已中止，原因是它使用本轮修正前的代码和实验协议；其部分 checkpoint 只作为历史残留，不作为有效 baseline 结果。

由于两组 seed 数量不同，这里只记录结果，不把小幅差异直接解释为稳定的模型优势。后续正式比较时应统一 seed 数量，并同时查看 AUROC、AUPRC、Recall/Sensitivity、Specificity 和 F1。
