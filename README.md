<div align="center">
<img src="./docs/taffy.gif" alt="Ace Taffy" width="200">
</div>

<h1 align="center">taffy-emoji</h1>

表情包图像二分类器。判断一张图片是否为[永雏塔菲](https://space.bilibili.com/1265680561)。

用于我的 bot。

## Model

Very Simple Binary Classifier.

EfficientNet-B0 backbone（ImageNet 预训练）提取 1280-dim 特征，接自定义头：

```
Dropout(0.4) -> Linear(1280 → 1)
```

推理时最后接 sigmoid 输出概率。

训练时作了数据增强：随机裁剪、翻转、颜色抖动、旋转。

## Install

```bash
uv sync
```

## Train

```bash
uv run python -m src.train           # 从 ImageNet 权重开始
uv run python -m src.train --resume  # 从 checkpoints/best.pth 继续
```

数据集：[`homearchbishop/ace-taffy-images`](https://huggingface.co/datasets/homearchbishop/ace-taffy-images)（首次运行自动下载）

## 推理

```bash
uv run python -m src.predict path/to/img.jpg              # 单张图片 -> 打印结果
uv run python -m src.predict https://example.com/x.jpg   # URL -> 打印结果
uv run python -m src.predict --dir data/mydir             # 目录 -> outputs/<dir>_predictions.json
uv run python -m src.predict --threshold 0.6              # 自定义阈值（默认 0.5）
uv run python -m src.predict --val                        # 验证集阈值扫描，输出各阈值指标
uv run python -m src.predict --ckpt checkpoints/best.pth  # 指定 checkpoint   
```

# License

AGPL-3.0

# Acknowledgments

文档顶部动图来自 [春也Haruya](https://space.bilibili.com/3280)。
