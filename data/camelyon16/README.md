# 解压后的 CAMELYON16 数据

这是训练和审计实际读取的数据根目录。`splits.csv` 保存官方 train/test 划分，`patches_512/` 保存每个 WSI 对应的特征、标签、坐标和 patch 级标注。

这里不放原始 WSI；每个 WSI 以若干 `.npy` 数组表示。
