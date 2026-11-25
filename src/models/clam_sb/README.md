# CLAM-SB

调用固定 commit 的作者官方 `CLAM_SB`，包括 gated attention、top/bottom instance sampling、instance classifiers 和官方 bag/instance loss。项目 wrapper 只负责去除 collate padding 与统一二分类 score。

- 配置：`configs/experiments/clam_sb.yaml`
- 预留输出：`outputs/camelyon16_uni/clam_sb/`
- 预留日志：`logs/clam_sb/`
