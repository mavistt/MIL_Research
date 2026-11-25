# Smoke Test 报告

当前 ABMIL 基线的 GPU smoke test 已通过。

- 设备：`cuda:0`
- 输入特征形状：`[1, 7893, 1024]`
- padding mask 形状：`[1, 7893]`
- 切片标签形状：`[1]`
- ABMIL 输出形状：`[1]`
- 单步 BCEWithLogitsLoss：`0.375167`
- 获得梯度的参数张量：5
- 训练路径未包含 `coords` 或 `y_inst`

此外，项目自动化测试结果为 `2 passed`。
