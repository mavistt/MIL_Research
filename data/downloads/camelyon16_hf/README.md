# CAMELYON16 Hugging Face 下载目录

这里对应数据集 `torchmil/Camelyon16_MIL` 的本地下载镜像。`dataset/` 下保存五个指定文件：UNI 特征、切片标签、坐标、patch 标签和官方 split 文件。

下载脚本支持固定 Range 请求和 `.part` 断点续传；归档完成后才会去掉 `.part` 后缀。
