import numpy as np
import urllib.request
import gzip
import os
import struct
from cnn.core import Tensor
class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, seed=42):
        """
        数据加载器
        
        参数:
            dataset: 包含数据和标签的数据集
            batch_size: 批量大小，默认32
            shuffle: 是否在每个epoch开始时打乱数据，默认True
        """
        if seed:
            np.random.seed(seed=seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = dataset[0]
        self.labels = dataset[1]
        self.n_samples = len(self.data)
        self.idx = 0
        self.indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.idx >= self.n_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.idx:min(self.idx + self.batch_size, self.n_samples)]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        self.idx += self.batch_size
        
        # 转换为Tensor
        return Tensor(batch_data), Tensor(batch_labels)
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class FashionMNIST:
    """Fashion MNIST数据集加载器"""
    
    def __init__(self, root='./data', train=True, transform=None):
        """
        参数:
            root: 数据保存的根目录，默认'./data'
            train: 是否使用训练集，默认True
            transform: 数据转换函数，默认None
        """
        self.root = root
        self.train = train
        self.transform = transform
        
        if not os.path.exists(root):
            os.makedirs(root)
        
        if train:
            self.images_file = f"{root}/train-images-idx3-ubyte.gz"
            self.labels_file = f"{root}/train-labels-idx1-ubyte.gz"
            if not os.path.exists(self.images_file):
                self._download_mnist()
            self.data, self.targets = self._load_mnist()
        else:
            self.images_file = f"{root}/t10k-images-idx3-ubyte.gz"
            self.labels_file = f"{root}/t10k-labels-idx1-ubyte.gz"
            if not os.path.exists(self.images_file):
                self._download_mnist()
            self.data, self.targets = self._load_mnist()
    
    def _download_mnist(self):
        """下载Fashion MNIST数据集"""
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        
        if self.train:
            images = "train-images-idx3-ubyte.gz"
            labels = "train-labels-idx1-ubyte.gz"
        else:
            images = "t10k-images-idx3-ubyte.gz"
            labels = "t10k-labels-idx1-ubyte.gz"
        
        print(f"正在下载 {images}...")
        urllib.request.urlretrieve(base_url + images, self.images_file)
        print(f"正在下载 {labels}...")
        urllib.request.urlretrieve(base_url + labels, self.labels_file)
        print("下载完成！")
    
    def _load_mnist(self):
        """加载Fashion MNIST数据集"""
        # 读取图像
        with gzip.open(self.images_file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
        
        # 读取标签
        with gzip.open(self.labels_file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 数据预处理：归一化到[0,1]
        images = images.astype(np.float32) / 255.0
        
        return images, labels
    
    def __getitem__(self, index):
        """获取指定索引的样本"""
        img, target = self.data[index], self.targets[index]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def get_data(self):
        """返回整个数据集"""
        return self.data, self.targets
    
    def to_one_hot(self, num_classes=10):
        """将标签转换为one-hot编码"""
        one_hot_targets = np.zeros((len(self.targets), num_classes))
        for i, target in enumerate(self.targets):
            one_hot_targets[i, target] = 1
        self.targets = one_hot_targets
        return self