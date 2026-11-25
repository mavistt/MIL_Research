# UNI 特征

这里每个 `.npy` 文件对应一个 WSI，数组形状为 `[N, 1024]`，其中 N 是该 WSI 的 patch 数量。ABMIL 的训练输入只读取这些特征和对应的 WSI 级标签。
