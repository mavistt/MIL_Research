# 官方 baseline 第二轮实现审计

## 审计结论

本轮只做代码修正和验证，不启动新的 CLAM、DSMIL、TransMIL 多 seed 正式训练。上一轮遗留的 CLAM seed 0 旧代码训练已停止；其部分 checkpoint 不作为本轮结果使用。

审计将“作者模型/损失”和“项目实验协议”分开记录。模型核心尽量保持作者实现；统一的固定 split、验证 AUROC、测试指标和输入接口属于项目级适配，不能表述为作者原始完整训练协议。

## 来源与模型架构

| Baseline | 架构忠实度 | 固定来源 | 当前项目适配 |
|---|---|---|---|
| ABMIL / Gated ABMIL | 部分忠实：使用 TorchMIL 1.0.2 ABMIL，而不是作者 WSI 训练仓库 | [AttentionDeepMIL commit `eb0434b`](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/tree/eb0434ba2795711a45d693d60120ae53532b1b93) | 统一输入/输出 shape、BCE score；历史结果保留 |
| Mean Pooling | 非论文官方模型，项目控制对照 | 项目代码 | masked mean + linear classifier |
| CLAM-SB | 忠实于作者 `CLAM_SB` 核心网络 | [CLAM commit `53e2409`](https://github.com/mahmoodlab/CLAM/tree/53e2409d4a8189c682c173382964a85f114f923c) | 去除 padded-bag、二分类 score；核心 attention、classifier、instance sampling 未简化 |
| DSMIL | 忠实于作者 `FCLayer/BClassifier/MILNet` | [DSMIL commit `80465ed`](https://github.com/binli123/dsmil-wsi/tree/80465ed534ac49b5e550833da56c27936f0b9440) | 去除 padded-bag、二分类 score；补齐作者初始化 |
| TransMIL | 忠实于作者 `_fc1`、两层 TransLayer、PPEG、CLS、`_fc2` | [TransMIL commit `9d6aee5`](https://github.com/szc19990412/TransMIL/tree/9d6aee57c7c72375fb9132dc58cd8c9b0f0a949c) | 去除 padded-bag、`.to(device)` 替代硬编码 `.cuda()`、Nyström PyTorch 2.3 兼容层和左侧 padding 后的尾部取回 |

Nyström 兼容层以 `nystrom-attention==0.0.9` 的源码为数值参考；只对 PyTorch 2.3 不允许的 view inplace 写法做 out-of-place 兼容，并按本项目的左侧 padding 语义从末尾取回原始 `n` 个位置。测试还单独证明了未修正的 head slice 与 tail slice 在非整除长度上确实不同。因此，这里不声称与未修正的原始 0.0.9 wheel 字节级一致，而是明确验证“官方公式 + PyTorch 兼容 + 本项目要求的 padding 修正”。

## Loss、初始化和优化器

| Baseline | Training Loss | Validation Loss | Initialization | Optimizer / scheduler |
|---|---|---|---|---|
| ABMIL / Gated | Bag BCEWithLogits | Bag BCEWithLogits | TorchMIL 默认 | Adam，沿用现有配置 |
| Mean Pooling | Bag BCEWithLogits | Bag BCEWithLogits | PyTorch 默认 | Adam |
| CLAM-SB | `0.7 * bag CrossEntropy + 0.3 * instance CrossEntropy` | 只计算 Bag CrossEntropy | 作者默认初始化 | Adam，lr `1e-4`，wd `1e-5` |
| DSMIL | `0.5 * bag BCE + 0.5 * max-instance BCE` | Bag BCE；评估仍使用 bag branch | 与作者 `apply_sparse_init` 一致：Linear/Conv1d/Conv2d orthogonal，bias 0 | Adam，betas `(0.5, 0.9)`，wd `1e-3`，CosineAnnealingLR，eta_min `5e-6` |
| TransMIL | 二分类 CrossEntropy | 二分类 CrossEntropy | 作者默认初始化 | Lookahead(RAdam)，lr `2e-4`，wd `1e-5` |

代码中已明确区分 `compute_training_loss` 与 `compute_validation_loss`。CLAM 的训练总 loss 不再被误当作验证选择 loss。

## Checkpoint 选择与作者协议差异

当前六种配置统一使用项目级 `val_auroc` 选择 checkpoint，测试阈值和指标计算保持一致。这是为了让后续聚合器比较使用同一评价协议，不等于作者原始训练脚本协议。

| 方法 | 作者原始选择/训练设置 | 本项目决策 | 状态 |
|---|---|---|---|
| CLAM | `EarlyStopping` 使用 Validation Bag Loss | 保留 Bag Loss 作为 validation loss，但 checkpoint 选择统一使用 `val_auroc`；当前 strict 配置不启用额外早停 | Project-level Training Adaptation |
| DSMIL | 使用 `(Accuracy + AUC) / 2` 的 current score；`average=False` 时使用 bag prediction | 保留 `average=False` 的 bag branch；checkpoint 选择统一使用 `val_auroc` | Project-level Training Adaptation；bag branch 已保留 |
| TransMIL | `fp16=True`、gradient accumulation `2`；作者 `CamelData` 的 `data_shuffle=False` 表示不打乱 WSI 内部 patch 顺序 | `patch_shuffle=False` 保留原始 patch 顺序；训练 DataLoader 的 WSI 顺序统一为 `wsi_shuffle=True`，验证/测试为 `False`；当前项目 `fp16=False`、accumulation `1`，原因是统一的小 batch 验证框架尚未实现作者 AMP/累积器 | Project-level Training Adaptation |

上述决策都写入 YAML 和每次运行的 `run_metadata.json`，不再把项目适配称作完整作者训练协议。

## 数据和输入协议

- 固定使用 `data/splits/seed1234.csv`，不重新随机划分。
- 训练只请求 `X, Y`；`y_inst` 和 `coords` 仅用于审计或后续分析。
- 所有模型使用同一份 1024 维 UNI feature、同一 test threshold 和同一套 test metrics。
- `training.wsi_shuffle` 只控制训练集 WSI 的 DataLoader 顺序；`training.patch_shuffle=false` 明确保持每个 WSI 内部的原始 patch 顺序。Smoke test 只从固定 split 的 train partition 取真实 Bag。
- `scripts/audit_data.py` 额外审计 399/400 manifest gap、`test_049`、WSI/patch label 一致性、patch label 合法值、未引用 feature 文件和多候选路径。

## 验证状态

| 项目 | 状态 |
|---|---|
| DSMIL 官方初始化存在且正交/零 bias | 已验证：4 个 Linear/Conv1d 层通过正交性和零 bias 检查 |
| CLAM training loss 与 validation bag loss 分离 | 已验证：11 个基础/协议测试通过 |
| TransMIL Nyström forward/backward 数值对照 | 已验证：7 个长度/前后向案例通过，含 145、256、8101、44522 |
| 六模型真实 small/medium/max Forward、Loss、Backward、finite gradient、optimizer step、checkpoint reload、evaluation output | 已验证：18/18 通过，6 模型 × 3 个真实 Bag；TransMIL Lookahead 另做 PyTorch 2.3 hook 兼容修正 |
| 数据审计扩展 | 已验证：399 条 manifest、0 缺失必需文件、0 标签不一致、0 非法 patch label、0 额外 feature、0 重复候选路径 |
| 历史 ABMIL seed1234 是否与新 seed 0–3 完全同协议 | 已确认：split_used.csv SHA256 相同；保存 config 的模型、epoch、Adam、学习率、阈值一致；`auroc` 与当前 `val_auroc` 在选择逻辑中等价 |

验收未通过前不允许启动正式多 seed 训练；本轮列出的验收项均已通过，因此可以在单独的正式训练任务中按这些已记录的协议启动后续 baseline。

## 本轮实际验收记录

- 基础模型、项目契约、DSMIL 初始化和训练协议：`11 passed`。
- Nyström 数值对照：`7 passed`；同一权重、同一输入，比较 forward、输入梯度和参数梯度。
- 完整 pytest：`19 passed`；新增 TransMIL 连续 7-step Lookahead 预检覆盖第 6 步 slow-buffer 更新。
- 真实数据 smoke：`18 passed, 0 failed`；使用固定 split 的 train partition 真实 CAMELYON16 + UNI feature small/medium/max Bag，覆盖六个模型的 loss、backward、finite gradient、optimizer step、checkpoint save/reload 和 evaluation output。
- 数据审计：通过；详细结果见 `reports/project/data_audit.md`。

因此，本轮审查验收通过，可以进入正式训练前的实验排程；本轮本身没有启动 CLAM、DSMIL 或 TransMIL 的正式多 seed 训练。

## 正式三 Baseline 训练结果

后续正式训练已按本报告记录的协议完成：CLAM-SB、DSMIL、TransMIL 均完成 seed `0/1/2`，三个模型的每个 seed 均正常退出并生成 checkpoint、training history、验证/测试 metrics、predictions 和 run metadata。

- CLAM-SB：AUROC `0.984779 ± 0.008493`，AUPRC `0.981782 ± 0.007359`。
- DSMIL：AUROC `0.962585 ± 0.015693`，AUPRC `0.959662 ± 0.011634`。
- TransMIL：AUROC `0.958759 ± 0.047688`，AUPRC `0.954957 ± 0.052745`。

六个 Baseline 的 common seed 0/1/2 总比较见 `reports/project/all_baselines_comparison.md`；测试 WSI ID、label 和 split 一致性检查通过，固定 split SHA256 为 `dd07e2e746c88effb7b74b3e71df3257a986b936a3a166717042a7e9ebe7710e`。
