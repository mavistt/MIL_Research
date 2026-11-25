# 数据说明

本目录保存 CAMELYON16 的项目数据、固定划分和数据说明。原始 WSI、UNI 权重和 ResNet 特征不在本项目范围内。

## 数据来源与下载内容

来源：Hugging Face 数据集 `torchmil/Camelyon16_MIL`。

只下载以下五项：

| 文件 | 大小 |
|---|---:|
| `downloads/camelyon16_hf/dataset/patches_512/features/features_UNI.tar.gz` | 13,736,145,354 bytes |
| `downloads/camelyon16_hf/dataset/patches_512/labels.tar.gz` | 4,834 bytes |
| `downloads/camelyon16_hf/dataset/patches_512/coords.tar.gz` | 7,059,298 bytes |
| `downloads/camelyon16_hf/dataset/patches_512/patch_labels.tar.gz` | 110,041 bytes |
| `downloads/camelyon16_hf/dataset/splits.csv` | 6,300 bytes |

归档文件保留在 `downloads/`，解压后的数据位于 `camelyon16/`。下载器使用 Range 请求和 `.part` 断点文件，完成后才会原子重命名。

## 数据审计结果

审计依据官方 `splits.csv`，结果如下：

- WSI 总数：399
- 官方 train：270；官方 test：129
- 阳性 WSI：160；阴性 WSI：239
- train/test 重叠：0
- 重复 WSI ID：0
- 特征：所有文件均为二维矩阵 `[N, 1024]`
- 特征值：未发现 NaN、Inf 或空特征文件
- 坐标：与特征数量匹配，形状为 `[N, 2]`
- patch 标签：与对应特征数量匹配
- patch 数量：最小 140，最大 44,402，均值约 9,066.62，中位数 7,984
- 缺失文件：0
- 数据审计报告：`../reports/data_audit.md`

## 弱监督约束

ABMIL 训练只使用 patch 特征 `X` 和切片级标签 `Y`。`coords` 和 `y_inst` 仅为后续定位、可视化和实例级评估保留，不参与当前 baseline 的训练监督。

## 固定划分

`splits/seed1234.csv` 是在官方 train 分区内按切片标签分层、随机种子 1234 生成的 80/20 train/validation 划分；官方 test 分区保持不变。

| 分区 | 总数 | 阴性 | 阳性 |
|---|---:|---:|---:|
| train | 216 | 127 | 89 |
| validation | 54 | 32 | 22 |
| test | 129 | 80 | 49 |

具体划分规则见 `splits/README.md`。测试集不用于早停、模型选择或阈值优化。
