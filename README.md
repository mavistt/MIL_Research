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

每个 baseline 都按同一层级隔离：

```text
src/models/<baseline>/                 独立模型实现与说明
configs/experiments/<baseline>.yaml    独立超参数配置
outputs/camelyon16_uni/<baseline>/     独立 checkpoint、指标和图表
logs/<baseline>/                       独立调度日志与 seed 日志
```

当前 baseline 目录为 `abmil`、`gated_abmil`、`mean_pooling`、`clam_sb`、`dsmil` 和 `transmil`。Gated ABMIL 与标准 ABMIL 共享同一个 TorchMIL 适配器，仅通过配置开关启用门控注意力；实验输出和日志仍完全分开。

各目录内都有更细的中文说明：

- `configs/README.md`：配置文件如何控制数据集和实验。
- `data/README.md`：下载文件、解压数据、审计结果和固定划分。
- `src/README.md`：数据集、模型、指标和工具代码的分工。
- `scripts/README.md`：每个命令行脚本的用途。
- `reports/README.md`：环境、数据审计和 smoke test 报告。
- `outputs/README.md`：训练结果和评估结果的说明。
- `tests/README.md`：自动化测试说明。

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

审计会检查官方 train/test 划分、重复或重叠的 WSI ID、特征与标签对应关系、特征维度（应为 1024）、有限值、坐标形状、实例标签长度、切片标签和 patch 数量统计，并写入 `reports/project/data_audit.md`。

训练使用官方训练分区中的分层 80/20 train/validation 划分，随机种子为 1234；官方 test 分区保持不变，不用于早停、模型选择或阈值优化。

## Smoke test 与多 Seed 基线

```bash
python scripts/smoke_test.py --project-root .
python scripts/train.py --project-root . --config configs/experiments/abmil.yaml
python scripts/evaluate.py --project-root . --config configs/experiments/abmil.yaml \
  --checkpoint outputs/abmil_seed1234/checkpoint_best.pt
```

Smoke test 会加载一个 WSI、检查变长 bag 的 padding mask、执行 ABMIL 前向和反向传播，并确认训练路径不包含 `coords` 或 `y_inst`。初始训练使用 batch size 1、Adam、BCEWithLogitsLoss、验证集选最优检查点和早停。

原始 `abmil_seed1234/` 历史结果保持不变。新的多 seed 实验使用独立目录和日志：

```bash
nohup python scripts/run_seeds.py \
  --project-root . \
  --config configs/experiments/abmil.yaml \
  --experiment-name abmil_ungated \
  --seeds 0 1 2 3 --skip-complete \
  > logs/abmil_ungated_seeds.log 2>&1 &
python scripts/summarize_results.py --project-root . --experiment-name abmil_ungated
```

每个 seed 的结果位于 `outputs/camelyon16_uni/<实验名>/seed_<seed>/`，对应日志位于 `logs/`。`run_seeds.py` 按 seed 串行运行，支持跳过已完成的 seed；汇总脚本会输出每个 seed、均值、样本标准差和有效样本数。

当前已实现并保持统一训练接口的模型包括：Mean Pooling、标准 ABMIL、Gated ABMIL、CLAM-SB、DSMIL 和 TransMIL。CLAM-SB、DSMIL、TransMIL 使用各自作者 commit 的核心实现；ABMIL/Gated ABMIL 使用 TorchMIL 1.0.2 适配器。实现来源和严格 loss/optimizer 配置见 reports/project/official_implementation_audit.md。

基线输出包括：

- `checkpoint_best.pt`
- `training_history.csv`
- `metrics_val.json` 和 `metrics_test.json`
- `predictions_test.csv`
- ROC、PR 和混淆矩阵图

主要指标为 AUROC 和 AUPRC，同时记录 Accuracy、Precision、Recall、Specificity、F1 和 Balanced Accuracy。

## 可扩展设计

数据适配器、特征变换、MIL 聚合器、分类头和实验配置保持相对独立。新增模型只需实现统一的 forward(x, mask) 和 compute_training_loss(y, x, mask) 接口，并在 registry 中注册；不会改变固定 split 或特征文件。

## 版本控制约定

数据、归档、缓存、虚拟环境、日志、输出、检查点和临时文件均由 `.gitignore` 排除。仓库只保存可复用代码、配置、测试和不包含私密运维信息的项目文档。
