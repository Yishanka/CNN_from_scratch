import numpy as np
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from cnn.data.loader import FashionMNIST, DataLoader
from cnn.model import SimpleCNN
from cnn.loss.cross_entropy_loss import CrossEntropyLoss
from cnn.optimizer.adam import Adam

# Fashion MNIST类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def train(model, train_loader, criterion, optimizer, epochs=5):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数，默认5
    
    返回:
        训练损失和准确率历史记录
    """
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (inputs, targets) in progress_bar:
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            criterion.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.data.item()
            
            # 计算准确率（对于非one-hot标签）
            predicted = np.argmax(outputs.data, axis=1)
            total += targets.shape[0]
            if len(targets.shape) > 1:  # 如果是one-hot编码
                actual = np.argmax(targets.data, axis=1)
            else:
                actual = targets.data.astype(np.int64)
            correct += np.sum(predicted == actual)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return train_losses, train_accs

def test(model, test_loader):
    """
    测试模型
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
    
    返回:
        准确率和预测结果
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    
    for batch_idx, (inputs, targets) in progress_bar:
        # 前向传播
        outputs = model(inputs)
        
        # 计算准确率
        predicted = np.argmax(outputs.data, axis=1)
        total += targets.shape[0]
        if len(targets.shape) > 1:  # 如果是one-hot编码
            actual = np.argmax(targets.data, axis=1)
        else:
            actual = targets.data.astype(np.int64)
        correct += np.sum(predicted == actual)
        
        # 保存预测结果和真实标签
        all_preds.extend(predicted)
        all_targets.extend(actual)
        
        # 更新进度条
        progress_bar.set_postfix({
            'acc': 100. * correct / total
        })
    
    # 计算总准确率
    accuracy = 100. * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_targets

def visualize_results(train_losses, train_accs, test_acc, all_preds, all_targets, num_samples=10):
    """
    可视化训练结果和预测样本
    
    参数:
        train_losses: 训练损失历史
        train_accs: 训练准确率历史
        test_acc: 测试准确率
        all_preds: 所有预测结果
        all_targets: 所有真实标签
        num_samples: 可视化样本数量，默认10
    """
    # 1. 绘制训练损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # 2. 显示混淆矩阵
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    for i in range(len(all_preds)):
        confusion_matrix[all_targets[i], all_preds[i]] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 3. 直接使用已有的预测结果来显示样本
    # 从预测和真实标签中随机选择样本，而不是从整个测试集中选择
    num_samples = min(num_samples, len(all_preds))
    sample_indices = np.random.choice(len(all_preds), num_samples, replace=False)
    
    # 加载测试数据以获取图像
    test_dataset = FashionMNIST(train=False)
    test_images, _ = test_dataset.get_data()
    
    # 获取在main函数中使用的相同测试子集的索引
    np.random.seed(42)  # 确保与main函数中使用相同的随机种子
    test_dataset = FashionMNIST(train=False)
    _, test_labels = test_dataset.get_data()
    subset_size_test = 1000
    test_indices = np.random.choice(len(test_labels), subset_size_test, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, sample_idx in enumerate(sample_indices):
        # 获取对应的原始测试集中的索引
        original_idx = test_indices[sample_idx]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_images[original_idx].reshape(28, 28), cmap='gray')
        plt.title(f"真: {class_names[all_targets[sample_idx]]}\n预: {class_names[all_preds[sample_idx]]}", 
                 color=("green" if all_preds[sample_idx] == all_targets[sample_idx] else "red"),
                 fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.close()

def save_model(model, filename='model.pkl'):
    """
    保存模型参数
    
    参数:
        model: 模型对象
        filename: 保存的文件名，默认'model.pkl'
    """
    # 直接保存模型的参数列表
    with open(filename, 'wb') as f:
        pickle.dump(model.parameters(), f)
    print(f"模型参数已保存为 {filename}")

def load_model(filename='model.pkl'):
    """
    加载模型参数并重建模型
    
    参数:
        filename: 保存的文件名，默认'model.pkl'
    
    返回:
        重建的模型对象
    """
    # 加载模型参数列表
    with open(filename, 'rb') as f:
        params_list = pickle.load(f)
    
    # 创建一个新的模型实例
    model = SimpleCNN(in_channels=1, num_classes=10)
    
    # 将保存的参数加载到模型中
    new_params = model.parameters()
    for i in range(len(params_list)):
        if i < len(new_params):
            new_params[i].data = params_list[i].data
    
    return model

def main():
    # 设置随机种子以便结果可复现
    np.random.seed(42)
    
    # 1. 加载Fashion MNIST数据集
    print("正在加载Fashion MNIST数据集...")
    train_dataset = FashionMNIST(train=True)
    test_dataset = FashionMNIST(train=False)
    
    # 获取数据
    train_data, train_labels = train_dataset.get_data()
    test_data, test_labels = test_dataset.get_data()
    
    # 只使用部分数据进行训练和测试，以加快速度
    subset_size_train = 5  # 只使用5000个训练样本
    subset_size_test = 1  # 只使用1000个测试样本
    
    # 随机选择子集
    train_indices = np.random.choice(len(train_data), subset_size_train, replace=False)
    test_indices = np.random.choice(len(test_data), subset_size_test, replace=False)
    
    train_data_subset = train_data[train_indices]
    train_labels_subset = train_labels[train_indices]
    test_data_subset = test_data[test_indices]
    test_labels_subset = test_labels[test_indices]
    
    print(f"原始训练集大小: {len(train_data)}, 使用子集大小: {len(train_data_subset)}")
    print(f"原始测试集大小: {len(test_data)}, 使用子集大小: {len(test_data_subset)}")
    print(f"图像形状: {train_data.shape[1:]}")
    
    # 创建数据加载器
    train_loader = DataLoader(dataset=(train_data_subset, train_labels_subset), batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=(test_data_subset, test_labels_subset), batch_size=64, shuffle=False)
    
    # 2. 创建模型、损失函数和优化器
    model = SimpleCNN(in_channels=1, num_classes=10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 3. 训练模型
    print("\n开始训练...")
    # 减少训练轮数以进一步加快速度
    train_losses, train_accs = train(model, train_loader, criterion, optimizer, epochs=3)
    
    # 4. 测试模型
    print("\n开始测试...")
    test_acc, all_preds, all_targets = test(model, test_loader)
    
    # 5. 可视化结果
    print("\n正在可视化结果...")
    visualize_results(train_losses, train_accs, test_acc, all_preds, all_targets)
    
    # 6. 保存模型
    save_model(model)
    
    print("\n所有任务已完成！")

if __name__ == "__main__":
    main()