# Mean Pooling

对有效 patch feature 做 masked mean pooling，再接一个线性分类器。它是无注意力聚合的简单对照 baseline。

- 配置：`configs/experiments/mean_pooling.yaml`
- 预留输出：`outputs/camelyon16_uni/mean_pooling/`
- 预留日志：`logs/mean_pooling/`
