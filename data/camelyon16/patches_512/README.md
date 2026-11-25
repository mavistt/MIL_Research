# 512×512 patch 数据

这里保存从 512×512 patch 网格得到的 WSI 级数组。四个子目录使用相同的 WSI ID 命名，便于逐个匹配。

- `features/`：UNI patch embedding。
- `labels/`：WSI 级二分类标签。
- `coords/`：patch 坐标。
- `patch_labels/`：patch 级标注，仅用于后续实例级分析。
