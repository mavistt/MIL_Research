# 报告目录

报告也按实验和用途隔离，根目录只保留本说明：

```text
reports/
├── project/          数据审计、环境、smoke test 和通用 baseline 说明
├── abmil/            历史单 seed 标准 ABMIL 说明
├── abmil_ungated/    标准 ABMIL 多 seed 汇总和逐 seed 指标
├── gated_abmil/      Gated ABMIL 汇总和逐 seed 指标
├── mean_pooling/     后续实验预留
├── clam_sb/          后续实验预留
├── dsmil/            后续实验预留
└── transmil/         后续实验预留
```

每个已运行实验目录包含 `per_seed.csv`、`summary.csv`、`summary.md`；后台调度完成状态保存在对应目录的 `seeds_status.json`。未运行的 baseline 目录只保留说明文件。

这里保存可长期查看的报告，不保存临时运行日志；日志统一放在 `logs/<baseline>/`。
