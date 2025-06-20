from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import cnn
from cnn.core import Tensor
from cnn.layer import Linear, ReLU, Conv2d, Flatten, MaxPool2d, Softmax, BatchNorm2d
from cnn.optimizer import Adam, SGD
from cnn.loss import CrossEntropyLoss
from cnn.data import FashionMNIST, DataLoader


# ==== 评估指标函数 ====
def compute_metrics(pred, true):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(true, axis=1)
    pred_np = pred_labels
    true_np = true_labels

    tp = np.sum((pred_np == 1) & (true_np == 1))
    fp = np.sum((pred_np == 1) & (true_np == 0))
    fn = np.sum((pred_np == 0) & (true_np == 1))
    accuracy = np.mean(pred_np == true_np)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def compute_multiclass_metrics(pred: Tensor, true: Tensor, num_classes=10):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(true, axis=1)

    precision_list = []
    recall_list = []
    f1_list = []

    for cls in range(num_classes):
        tp = np.sum((pred_labels == cls) & (true_labels == cls))
        fp = np.sum((pred_labels == cls) & (true_labels != cls))
        fn = np.sum((pred_labels != cls) & (true_labels == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    accuracy = np.mean(pred_labels == true_labels)
    return accuracy, np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

# ==== 主程序 ====
if __name__ == "__main__":

    # 数据
    train_dataset = FashionMNIST(root='./data', train=True)
    train_dataset.to_one_hot()
    train_loader = DataLoader(train_dataset.get_data(), batch_size=64, shuffle=True, seed=42)

    test_dataset = FashionMNIST(root='./data', train=False)
    test_dataset.to_one_hot()
    test_loader = DataLoader(test_dataset.get_data(), batch_size=1000, shuffle=False, seed=42)

    # 模型
    model = cnn.Model()
    model.sequential(
        Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(channels=16),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(channels=64),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Flatten(),

        Linear(in_features=64*7*7, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=10),
        Softmax()
    )

    model.compile(
        loss=CrossEntropyLoss(lambda2=0.000),
        optimizer = Adam(lr=1e-4, beta1=0.9, beta2=0.999)
    )

    # 训练
    metrics = {
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    for epoch in range(10):
        model.train()
        print(f"\nEpoch {epoch + 1}/{10}")
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100)

        for batch_idx, (X, y) in train_iterator:
            pred = model.forward(X) 
            loss = model.loss(pred, y)
            model.backward(remove_graph=True)
            model.step()
            model.zero_grad()

            # 显示当前 batch loss
            train_iterator.set_postfix({"Batch Loss": loss.data})

        model.eval() # 必须执行，保证参数正确不参与计算图构建
        # 记录训练和测试的预测结果
        train_preds, train_trues = [], []
        test_preds, test_trues = [], []
        train_loss, test_loss = [], []

        # 计算训练集的指标
        for X, y in train_loader:
            pred = model.forward(X)
            loss = model.loss(pred, y)
            train_loss.append(loss.data)
            train_preds.append(pred)
            train_trues.append(y)

        # 拼接训练数据的所有预测和真实标签
        train_preds = np.concatenate([p.data for p in train_preds], axis=0)
        train_trues = np.concatenate([t.data for t in train_trues], axis=0)

        # 计算训练集指标
        train_acc, prec, rec, f1 = compute_multiclass_metrics(train_preds, train_trues)
        metrics["train_accuracy"].append(train_acc)
        metrics["train_loss"].append(np.mean(train_loss))

        # 计算测试集的指标
        for X, y in test_loader:
            pred = model.forward(X)
            loss = model.loss(pred, y)
            test_loss.append(loss.data)
            test_preds.append(pred)
            test_trues.append(y)

        # 拼接测试数据的所有预测和真实标签
        test_preds = np.concatenate([p.data for p in test_preds], axis=0)
        test_trues = np.concatenate([t.data for t in test_trues], axis=0)

        # 计算测试集指标
        test_acc, prec, rec, f1 = compute_multiclass_metrics(test_preds, test_trues)
        metrics["test_accuracy"].append(test_acc)
        metrics["test_loss"].append(np.mean(test_loss))
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)

        print(f"Epoch {epoch + 1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # 作图
    plt.figure(figsize=(12, 6))
    for key in ['train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1']:
        plt.plot(metrics[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
