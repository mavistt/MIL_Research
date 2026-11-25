# 固定数据划分

`seed1234.csv` 保存当前 ABMIL baseline 使用的固定划分。它由官方 CAMELYON16 train/test 划分生成：

- 官方 test 129 个 WSI 全部保留在 test；
- 官方 train 270 个 WSI 按切片级标签分层拆分为 80% train 和 20% validation；
- 随机种子：`1234`；
- train：216 个 WSI（阴性 127，阳性 89）；
- validation：54 个 WSI（阴性 32，阳性 22）；
- test：129 个 WSI（阴性 80，阳性 49）。

后续模型应复用该文件，不重新随机划分。test 只用于最终评估，不参与早停、模型选择或阈值优化。
