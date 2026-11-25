# 项目级报告

这里保存不属于某一个单独 baseline 的资料：数据审计、环境记录、官方实现审计、真实 feature smoke test、模型实现说明和整体结果比较。

- `official_implementation_audit.md`：记录每个 baseline 的作者 commit、核心文件、loss、optimizer 和适配边界。
- `official_smoke_test.json`：由 `scripts/official_smoke_test.py` 生成，记录不同 bag 长度的真实 forward、耗时和显存。
