# 数据集代码

- `camelyon16.py`：读取官方 split 和四类 `.npy` 文件的轻量 CAMELYON16 bag 适配器，提供变长 bag 的 padding mask。
- `__init__.py`：导出数据集类和 collate 函数。

训练默认只请求 `X` 和 `Y`；`coords`、`y_inst` 可供后续分析使用。
