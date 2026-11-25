# 模型代码

- `abmil.py`：封装 TorchMIL 的标准 ABMIL，输入 patch 特征和 mask，输出 WSI 级 logit。
- `registry.py`：根据 YAML 中的模型名称创建模型，方便未来加入其他聚合器。
- `__init__.py`：导出模型构造接口。
