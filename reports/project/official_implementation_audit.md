# 官方 baseline 实现审计

## 结论

本仓库的目标是做“严格 baseline”，因此正式 baseline 不再把 TorchMIL 的同名模型当作作者官方代码。当前分层如下：

- ABMIL / Gated ABMIL：继续使用 TorchMIL 1.0.2 的标准 ABMIL 适配器。它与 AttentionDeepMIL 的 attention 结构一致，但不是作者的 WSI 训练仓库；之前已经完成的结果保留，不因这次审计重跑。
- Mean Pooling：项目自有的简单聚合对照，不宣称是论文官方 baseline。
- CLAM-SB、DSMIL、TransMIL：已切换为对应作者仓库指定 commit 的核心实现，并仅增加了 padded-bag 到官方输入形状、二分类 score 和项目数据接口的适配。

## 来源与核对范围

| Baseline | 固定来源 commit | 使用的官方核心文件 | 项目中的适配 | 许可证记录 |
|---|---|---|---|---|
| ABMIL / Gated ABMIL | [AttentionDeepMIL `eb0434b`](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/tree/eb0434ba2795711a45d693d60120ae53532b1b93) | 官方 attention MIL 结构；当前运行实现为 TorchMIL 1.0.2 | 输出 shape、BCE score | TorchMIL 依赖许可证 |
| CLAM-SB | [CLAM `53e2409`](https://github.com/mahmoodlab/CLAM/tree/53e2409d4a8189c682c173382964a85f114f923c) | `models/model_clam.py` 的 `CLAM_SB`，包括 gated attention、instance classifiers、top-k sampling | `src/models/clam_sb/model.py` 处理变长 bag；不改变核心网络 | `src/models/clam_sb/THIRD_PARTY_LICENSE.md` |
| DSMIL | [DSMIL `80465ed`](https://github.com/binli123/dsmil-wsi/tree/80465ed534ac49b5e550833da56c27936f0b9440) | `dsmil.py` 的 `FCLayer`、`BClassifier`、`MILNet` | `src/models/dsmil/model.py` 适配 padded bag；保留 instance critical selection 和 bag branch | `src/models/dsmil/THIRD_PARTY_LICENSE.md` |
| TransMIL | [TransMIL `9d6aee5`](https://github.com/szc19990412/TransMIL/tree/9d6aee57c7c72375fb9132dc58cd8c9b0f0a949c) | `models/TransMIL.py` 的 `_fc1`、两层 `TransLayer`、PPEG、CLS token、`_fc2` | `src/models/transmil/model.py` 适配输入与二分类 CE；`nystrom_compat.py` 只修复 PyTorch 2.3 的 inplace autograd 兼容性 | TransMIL/依赖许可证按源仓库保留 |

## 训练目标与配置

### CLAM-SB

使用官方 small 结构 `1024 -> 512 -> 256`、gated attention、dropout 0.25、`k_sample=8`、`subtyping=false`。训练目标严格采用官方训练循环的

`0.7 * bag CrossEntropy + 0.3 * instance CrossEntropy`。

配置为 Adam、learning rate `1e-4`、weight decay `1e-5`、batch size 1、最多 200 epochs；验证集只用于选择 checkpoint，不启用额外的项目自定义早停。

### DSMIL

使用官方 binary path `output_class=1`，保留 instance classifier、critical-instance selection、query/attention 和 bag classifier。训练目标为

`0.5 * bag BCEWithLogits + 0.5 * max-instance BCEWithLogits`。

配置为 Adam `(betas=(0.5, 0.9), weight_decay=1e-3)`、learning rate `1e-4`、CosineAnnealingLR `eta_min=5e-6`、最多 50 epochs、patience 10。评估使用官方 bag branch，不取 instance prediction 平均。

### TransMIL

使用官方 `1024 -> 512` 投影、两层 8-head Nystrom attention、256 landmarks、6 次 pseudo-inverse、residual=True、dropout 0.1、PPEG 的 7/5/3 depth-wise convolution 和二分类 logits。bag 序列补方仍由官方实现完成，不截断最长 bag。

配置为 Lookahead(RAdam)、learning rate `2e-4`、weight decay `1e-5`、batch size 1、最多 200 epochs、patience 10，loss 为二分类 CrossEntropy。

### 项目共同约束

- 固定 `data/splits/seed1234.csv`，不因不同 baseline 改变划分。
- 输入固定为已下载的 1024 维 UNI feature；训练只读取 `X, Y`，不使用坐标和 instance label。
- 训练接口用 scalar `logit_1 - logit_0` 统一计算 ROC/PR；这等价于两类 softmax 的正类概率经过 sigmoid。
- 每个 baseline、seed、log、output、report 均隔离在自己的子目录。

## 验证记录

远端香港服务器的项目 venv 中已通过：

```text
8 passed, 1 warning
```

覆盖 6 个模型入口和项目契约测试。另有 `nystrom_compat.py`：固定依赖 `nystrom-attention==0.0.9` 在 PyTorch 2.3 上对 `chunk` view 执行 inplace `q *= scale` 会触发 autograd 错误，因此只把该操作改为 out-of-place，未改变 attention 计算公式、层数或超参数。

真实 CAMELYON16 feature 的 small / medium / large bag smoke test 完成后，会把样本数量、显存和耗时补到本报告；正式多 seed 结果写入对应 `reports/<baseline>/`，不与本审计报告混合。
