# Caltech-101Classity

## 一、项目简介
基于PyTorch完成的一个101类别图像分类。使用`卷积神经网络（CNN）`，自行设计`train_mine.py`与`ResNet-18`对比结果。并实现单张图片预测`predict.py`。

## 二、数据集
* 来源:([Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02?utm_source=chatgpt.com))
* 结构:包含101个不同物体的文件夹，每个文件夹中有几十---几百张图片。
* 图片格式:分辨率不同的RGB图。
* 划分：随机80%作为训练集，剩下20%作为测试集。

## 三、任务目标
* 自行设计的CNN网络在测试集上准确率达到85%以上
* 修改ResNet-18的分类层
* 保存自设计CNN和ResNet-18的最优模型，在`predict.py`中加载并完成单张图片预测。

## 四、处理流程
### 1.数据加载
删除background类，使用`ImageFolder`读取图片，统一resize为128*128方便后续处理，并进行归一化。
### 2.模型
#### 自建mineCNN:
* 共5层卷积(64->128->256->512->512)
```python
stage(3,   64),    # 128 → 64
stage(64,  128),   # 64 → 32
stage(128, 256),   # 32  → 16
stage(256, 512),   # 16  → 8
stage(512, 512, pool=False),  # 8 → 8（保留空间信息）
```
* 每个stage的结构为:
```python
nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
nn.BatchNorm2d(out_ch),
nn.ReLU(inplace=True),
ResBlock(out_ch),
```
* 每个阶段均含有ResBlock，防止梯度消失。
```python
nn.Conv2d(channels, channels, 3, padding=1, bias=False),
nn.BatchNorm2d(channels),
nn.ReLU(inplace=True),
nn.Conv2d(channels, channels, 3, padding=1, bias=False),
nn.BatchNorm2d(channels),
```
* 使用Global Average Pooling(GAP)替代直接全连接，减少计算量并降低过拟合概率。
```python
self.gap = nn.AdaptiveAvgPool2d(1)
```
* 使用kaiming初始化卷积核。
#### ResNet18:
* 加载ImageNet预训练权重。
* 修改最后的全连接层，使模型输出101维(num_classes)的概率分布。
```python
nn.Linear(in_features, 256),
nn.ReLU(inplace=True),
nn.Dropout(0.4),
nn.Linear(256, num_classes),
```
* stage1(10epoch)冻结主干，只训练全连接层。
```python
for param in model.parameters():
    param.requires_grad = False
```
* stage2(20epoch)全网络微调。
```python
for param in model.parameters():
    param.requires_grad = True
```
### 3.训练策略
* AdamW优化器(weight_decay=1e-4)
* 线性Warmup(5epoch)+余弦退火学习率
* 使用Label Smoothing=0.1防止模型过度自信
* 梯度裁剪，max_norm=5.0防止梯度爆炸
* 最后保存最优模型
### 4.单张图片推理
输入单张图片，使用两个模型的最优参数分别预测，输出预测结果和置信度。

## 五、评估指标
* 模型在测试集上的准确率
* 单张图片的预测准确率与置信度

## 六、结果
最终自建模型mineCNN准确率在87%左右。ResNet18准确率在94%左右。
对umbrella.jpg单张图片的预测结果：
```
图片：umbrella.jpg
mineCNN  → umbrella             置信度: 0.9545
ResNet18     → umbrella             置信度: 0.9021
```
## 七、核心结论
* 1.在使用了残差结构与GAP后，准确率显著提升(从简易版的31%提升到87%)，这说明残差快对深层网络有非常大的帮助，与此同时GAP也有效缓解了模型的过拟合。
* 2.ResNet18的迁移学习速度明显更快(自建CNN完整流程需要30分钟左右，而ResNet仅需5分钟左右)，并且在多次单张图片预测中ResNet18的准确率和置信度都优于自建CNN，这说明ResNet18的泛化能力更强，体现出了预训练权重的优势。
* 3.LabelSmoothing、AdamW、Warmup+余弦退火学习率这些优化器和训练技巧对模型也有很大的提升，尤其是对不均衡数据集。
