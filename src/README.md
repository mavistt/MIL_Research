# 可复用代码包

`src/` 保存不依赖具体命令行入口的核心代码。

- `datasets/`：按需读取 WSI bag，并处理变长 patch 序列。
- `models/`：模型构造、注册和 ABMIL 实现。
- `metrics/`：分类指标、曲线和结果保存。
- `utils/`：配置读取、随机种子和通用工具。
