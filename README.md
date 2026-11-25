# 基于基础模型特征的 WSI-MIL 研究基线

本项目提供一个隔离、可复现的 CAMELYON16 全切片图像（WSI）弱监督多实例学习（MIL）基线。首个实验使用预先计算好的 UNI patch 特征和标准 ABMIL 完成切片级二分类，不下载原始 WSI，也不重新提取特征。

首个实验只使用切片级标签。坐标和 patch 级标签保留给后续定位与实例级分析，不会作为 ABMIL 的训练监督。

## 项目结构

```text
WSI-MIL-Research/
├── configs/                 数据集与实验配置
├── data/downloads/          指定的 Hugging Face 数据文件
├── data/camelyon16/         解压后的 CAMELYON16 数据
├── data/splits/             固定随机种子的训练/验证划分
├── src/                     数据集、模型、指标和工具代码
├── scripts/                 下载、解压、审计、测试、训练和评估脚本
├── outputs/                 检查点、指标、预测结果和图表
├── reports/                 环境与数据审计报告
└── .venv/                   项目专用 Python 虚拟环境
```

## 环境安装

请在项目根目录创建独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/print_environment.py
```

当前基线使用 TorchMIL 1.0.2 和标准 `torchmil.models.ABMIL`。代码会检查实际安装的 TorchMIL API、PyTorch、CUDA 和 GPU 信息，不依赖旧版 `Camelyon16MIL` 或 2048 维示例。

## 数据下载

只从 `torchmil/Camelyon16_MIL` 下载以下五项：

- `dataset/patches_512/features/features_UNI.tar.gz`
- `dataset/patches_512/labels.tar.gz`
- `dataset/patches_512/coords.tar.gz`
- `dataset/patches_512/patch_labels.tar.gz`
- `dataset/splits.csv`

这些是处理好的 UNI 特征及其元数据，不是原始 WSI、UNI 权重或 ResNet 特征。项目不会下载完整的 Hugging Face 仓库。

```bash
python scripts/download_data.py --project-root .
```

下载器使用固定大小的 HTTP Range 请求，将未完成文件保存为 `.part`，完成后再原子重命名。重新执行命令会按照已有 `.part` 文件大小断点续传，连接中断不会丢弃已有数据。

如果计算节点需要网络代理，请只在当前下载任务的进程环境中设置 `HTTP_PROXY` 和 `HTTPS_PROXY`；代理地址、账号、密钥和隧道配置不应写入仓库。

## 解压与数据审计

```bash
python scripts/extract_data.py --project-root .
python scripts/audit_data.py --project-root .
python scripts/make_splits.py --project-root .
```

审计会检查官方 train/test 划分、重复或重叠的 WSI ID、特征与标签对应关系、特征维度（应为 1024）、有限值、坐标形状、实例标签长度、切片标签和 patch 数量统计，并写入 `reports/data_audit.md`。

训练使用官方训练分区中的分层 80/20 train/validation 划分，随机种子为 1234；官方 test 分区保持不变，不用于早停、模型选择或阈值优化。

## Smoke test 与 ABMIL 基线

```bash
python scripts/smoke_test.py --project-root .
python scripts/train.py --project-root . --config configs/experiments/abmil.yaml
python scripts/evaluate.py --project-root . --config configs/experiments/abmil.yaml \
  --checkpoint outputs/abmil_seed1234/checkpoint_best.pt
```

Smoke test 会加载一个 WSI、检查变长 bag 的 padding mask、执行 ABMIL 前向和反向传播，并确认训练路径不包含 `coords` 或 `y_inst`。初始训练使用 batch size 1、Adam、BCEWithLogitsLoss、验证集选最优检查点和早停。

基线输出包括：

- `checkpoint_best.pt`
- `training_history.csv`
- `metrics_val.json` 和 `metrics_test.json`
- `predictions_test.csv`
- ROC、PR 和混淆矩阵图

主要指标为 AUROC 和 AUPRC，同时记录 Accuracy、Precision、Recall、Specificity、F1 和 Balanced Accuracy。

## 可扩展设计

数据适配器、特征变换、MIL 聚合器、分类头和实验配置保持相对独立。当前只实现可运行的 ABMIL 基线，不预先添加空的未来模型占位符。后续可以在不重写下载、数据审计、训练和评估流程的前提下加入 Gated ABMIL、CLAM、DSMIL、TransMIL、Mamba 或其他聚合方法。

## 版本控制约定

数据、归档、缓存、虚拟环境、日志、输出、检查点和临时文件均由 `.gitignore` 排除。仓库只保存可复用代码、配置、测试和不包含私密运维信息的项目文档。
