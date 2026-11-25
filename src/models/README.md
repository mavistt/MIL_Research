# 模型代码

- `abmil/`：标准 ABMIL 的 TorchMIL 适配器。
- `gated_abmil/`：Gated ABMIL 的独立入口；底层复用 `abmil/`，避免复制相同实现。
- `mean_pooling/`：不带注意力的 masked mean pooling + 线性分类器。
- `clam_sb/`：CLAM-SB 的 TorchMIL 官方实现适配器。
- `dsmil/`：DSMIL 的 TorchMIL 官方实现适配器。
- `transmil/`：TransMIL 的 TorchMIL 官方实现适配器。
- `common.py`：各模型共享的输出和 loss 辅助函数。
- `registry.py`：根据 YAML 中的模型名称创建模型。
- `pooling/`、`torchmil_baselines.py`：兼容旧导入路径的薄封装，不包含模型实现。
- `__init__.py`：导出模型构造接口。

所有模型都遵循 `forward(x, mask)` 和 `compute_loss(y, x, mask)` 接口。训练 batch size 保持为 1，因此变长 bag 不会产生跨 bag padding；TransMIL 额外保留 TorchMIL 官方的序列补方行为。
