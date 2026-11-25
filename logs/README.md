# 日志目录

这里保存长时间实验的运行日志，并按实验名称分子目录：

```text
logs/
├── abmil_ungated/
│   ├── run.log
│   └── seed_0.log ... seed_3.log
├── gated_abmil/
│   ├── run.log
│   └── seed_0.log ... seed_2.log
├── mean_pooling/       # 后续实验
├── clam_sb/            # 后续实验
├── dsmil/              # 后续实验
└── transmil/           # 后续实验
```

每个 seed 单独记录，`run.log` 记录该实验的调度状态。下载和安装完成后的临时日志不保留；训练结束后的正式结果以 `outputs/` 和 `reports/` 为准。
