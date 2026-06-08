# 实验五报告记录

## 语义进化

- 选用的大模型：
- 输入类别来源：
- 生成描述策略：
- 语义描述文件：`outputs/semantic_descriptions.json`

## 训练与测试

```bash
python semantic_evolution.py --data-root data/miniImageNet --split train
python train.py --config configs/semfew_miniimagenet.yaml
python test.py --checkpoint outputs/semfew_resnet18_miniimagenet/best_model.pt
```

## 结果

- N-way：
- K-shot：
- Query 数：
- 平均 Accuracy：
- 95% CI：

## 问题与措施

- 

## 不同类别性能差异分析

- 
