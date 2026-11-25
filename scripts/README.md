# 命令行脚本

- `download_data.py`：只下载五个指定 Hugging Face 文件，支持 Range 断点续传。
- `extract_data.py`：安全检查并解压四个 tar.gz 归档，复制官方 `splits.csv`。
- `audit_data.py`：检查 WSI 对应关系、特征维度、标签、坐标、缺失值和 patch 数量，并生成审计报告。
- `make_splits.py`：在官方 train 分区内生成固定随机种子的分层 train/validation 划分。
- `print_environment.py`：打印 Python、PyTorch、TorchMIL、CUDA、GPU 和磁盘信息。
- `smoke_test.py`：加载一个 bag，执行 ABMIL 前向、loss 和反向传播。
- `train.py`：执行训练、验证、早停、最佳 checkpoint 保存和基础评估；`--seed` 只覆盖训练随机种子，不改变固定 split。
- `evaluate.py`：读取指定 checkpoint，在 test 集生成独立评估结果和图表。
- `run_seeds.py`：按 seed 串行启动独立训练，每个 seed 单独记录日志和输出目录。
- `summarize_results.py`：汇总多个 seed 的 test 指标，计算均值、样本标准差和有效数量。
