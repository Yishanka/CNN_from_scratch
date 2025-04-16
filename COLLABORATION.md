# CNN from Scratch 开发计划

## 📋 待实现模块

### 网络层 (layer/)
- [x] Linear 层
- [ ] **Conv2d 层** - 优先级高
- [ ] BatchNorm 层
- [ ] MaxPool/AvgPool 层
- [ ] Dropout 层
- [ ] Flatten 层
- [ ] 激活函数层完善

### 损失函数 (loss/)
- [ ] **CrossEntropyLoss** - 优先级高
- [ ] MSELoss
- [ ] BCELoss
- [ ] L1Loss

### 优化器 (optimizer/)
- [ ] **Adam** - 优先级高
- [ ] SGD
- [ ] RMSProp
- [ ] Momentum
- [ ] 学习率调度器

### 数据处理 (data/)
- [x] Dataset 基类与实现
- [x] DataLoader 实现
- [x] 数据变换工具
- [ ] 更多数据集支持

### 其他功能
- [ ] 模型评估工具
- [ ] 模型保存与加载
- [ ] 示例模型实现 (LeNet, SimpleVGG)

## 💬 当前工作分配

| 模块 | 负责人 | 状态 | 计划完成日期 |
|------|--------|------|------------|
| core/tensor.py | @username | 已完成 | 2025-04-01 |
| data/ | @username | 已完成 | 2025-04-16 |
| layer/conv2d.py | @username | 进行中 | 2025-04-30 |
| optimizer/adam.py | - | 待认领 | - |
| loss/crossentropyloss.py | - | 待认领 | - |
| layer/batchnorm.py | - | 待认领 | - |
| layer/pooling.py | - | 待认领 | - |

## 📅 近期目标

1. 完成 Conv2d 层实现
2. 实现 Adam 优化器
3. 完成 CrossEntropyLoss 损失函数
4. 构建一个完整的 CNN 分类模型示例

---

最后更新: 2025-04-16