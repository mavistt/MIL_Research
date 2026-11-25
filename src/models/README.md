# 模型代码

- `abmil/`：标准 ABMIL 的 TorchMIL 适配器。
- `gated_abmil/`：Gated ABMIL 的独立入口；底层复用 `abmil/`，避免复制相同实现。
- `mean_pooling/`：不带注意力的 masked mean pooling + 线性分类器。
- `clam_sb/`：固定 commit 的作者官方 CLAM-SB 核心实现及项目适配器。
- `dsmil/`：固定 commit 的作者官方 DSMIL 核心实现及项目适配器。
- `transmil/`：固定 commit 的作者官方 TransMIL 核心实现、RAdam/Lookahead 和 PyTorch 兼容层。
- `common.py`：各模型共享的输出和 loss 辅助函数。
- `registry.py`：根据 YAML 中的模型名称创建模型。
- `pooling/`、`torchmil_baselines.py`：兼容旧导入路径的薄封装，不包含模型实现。
- `__init__.py`：导出模型构造接口。

所有模型都遵循 `forward(x, mask)` 和 `compute_training_loss(y, x, mask)` 接口。训练 batch size 保持为 1，因此变长 bag 不会产生跨 bag padding；TransMIL 保留作者官方的序列补方行为。实现来源、loss 和超参数详见 `reports/project/official_implementation_audit.md`。
