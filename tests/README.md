# 测试代码

- `test_project_contract.py`：检查项目关键目录、配置和代码接口是否存在，防止基础结构被意外破坏。
- `test_model_contract.py`：在合成 bag 上验证 Mean Pooling、ABMIL、Gated ABMIL、CLAM-SB、DSMIL 和 TransMIL 的统一前向、损失与反向传播接口。

运行方式：`pytest -q`。
