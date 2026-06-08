# 实验五：多模态大模型赋能的小样本视觉分类

本项目根据实验五指导书实现 SemFew 风格的小样本学习流程，用语义描述增强视觉原型分类。代码覆盖：

- `compute_center.py`：计算视觉 center 和语义 center；
- `train.py`：在 episode 上训练 SemAlign，并用交叉熵 loss 优化 Query 分类；
- `test.py`：在测试 episode 上预测 Query 类别并报告 Accuracy；
- `semantic_evolution.py`：生成语义描述 JSON，可替换为 WordNet + 大模型生成的类别视觉描述。

## 环境配置

```bash
cd prj_5
pip install -r requirements.txt
```

建议使用 Python 3.10+、PyTorch 2.x 和 NVIDIA GPU。首次使用 `pretrained: true` 会下载 ResNet-18 预训练权重；如果服务器不能联网，可以在配置中改为 `pretrained: false`。

## 数据准备

miniImageNet 建议整理为 ImageFolder 结构：

```text
data/miniImageNet/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_c/*.jpg
  test/
    class_d/*.jpg
```

每个类别至少需要 `k_shot + q_query` 张图片。默认配置是 5-way 1-shot，每类 15 张 Query 图片。

## 语义进化

先生成可运行的语义描述模板：

```bash
python semantic_evolution.py --data-root data/miniImageNet --split train --output outputs/semantic_descriptions.json
```

实验报告中可以自行选取大模型，把 `outputs/semantic_descriptions.json` 中每个类别的描述替换为更细的视觉描述。脚本只要求 JSON 键为类别目录名，值为描述文本。

## 计算 Center

```bash
python compute_center.py --config configs/semfew_miniimagenet.yaml --split train --output outputs/train_centers.pt
```

输出包含：

- `classes`：类别名称；
- `visual_centers`：ResNet 图像特征 center；
- `semantic_centers`：语义描述编码 center；
- `counts`：每类图片数量。

## 训练

```bash
python train.py --config configs/semfew_miniimagenet.yaml
```

主要输出在 `outputs/semfew_resnet18_miniimagenet/`：

```text
best_model.pt
last_model.pt
config.yaml
summary.json
```

如果要改成 5-way 5-shot：

```bash
python train.py --config configs/semfew_miniimagenet.yaml --k-shot 5
```

## 测试

```bash
python test.py --checkpoint outputs/semfew_resnet18_miniimagenet/best_model.pt --split test
```

输出 `outputs/test/metrics.json`，包含平均 Accuracy 和 95% 置信区间。

## 实验报告可写要点

1. 小样本学习按 episode 组织：Support 集计算类别原型，Query 集用于分类与评估。
2. SemFew 的语义进化把类别标签扩展为包含形状、颜色、纹理、部件和场景上下文的描述，减少只用类别名带来的语义偏差。
3. SemAlign 使用两层全连接网络融合视觉 center 与语义 center，得到增强后的类别 prototype。
4. Query 预测通过 Query 特征与各类别 prototype 的余弦相似度完成，训练 loss 为交叉熵。
5. 不同类别性能差异可从类内变化、背景干扰、类别间相似度、语义描述是否具体等角度分析。
