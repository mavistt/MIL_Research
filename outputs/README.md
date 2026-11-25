# 实验输出

这里保存训练检查点、训练历史、指标 JSON、预测 CSV 和曲线图。每个实验使用独立子目录，避免不同随机种子或模型互相覆盖。

- `camelyon16_uni/abmil_ungated/seed_0/` 至 `seed_3/`：标准 ABMIL 多 seed 结果。
- `camelyon16_uni/abmil_ungated/seed_1234/`：历史单 seed ABMIL 完整结果，保留用于复核。
- `camelyon16_uni/<experiment>/seed_<seed>/`：多 seed 实验的隔离输出，每个目录包含 config、split、checkpoint、训练历史、指标、预测和图表。
