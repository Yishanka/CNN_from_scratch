import matplotlib.pyplot as plt
import time
import threading
import keyboard 
from cnn.core import Tensor

class LossMonitor:
    def __init__(self, stop_key: str):
        self.losses = [] 
        self.is_training = True  # 控制是否继续训练
        self.stop_key = stop_key  # 停止训练的按键

        # 初始化画布
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        plt.ion()  # 开启交互模式

        # 启动键盘检测线程
        stop_thread = threading.Thread(target=self.check_stop_key)
        stop_thread.daemon = True
        stop_thread.start()

    def append_loss(self, loss: Tensor):
        '''添加损失值'''
        loss = loss if isinstance(loss, Tensor) else Tensor(loss)
        self.losses.append(loss.data)

    def update_plots(self):
        '''更新图表'''
        self.ax.clear()
        self.ax.plot(self.losses, label="Loss", color="blue")
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        plt.pause(0.1)

    def check_stop_key(self):
        '''检测按键以停止训练'''
        while self.is_training:
            if keyboard.is_pressed(self.stop_key):
                self.stop_training()
                print("Training stopped by user.")
                time.sleep(2)
            time.sleep(0.1) # 每 0.1 检测一次

    def stop_training(self):
        '''停止训练'''
        self.is_training = False

class MetricMonitor:
    def __init__(self, type: str):
        self.metric = [] 
        self.type = type

        # 初始化画布
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        plt.ion()  # 开启交互模式

    def append_metric(self, value):
        '''添加标准'''
        self.metric.append(value)

    def update_plots(self):
        '''更新图表'''
        self.ax.clear()
        self.ax.plot(self.metric, label=self.type, color="blue")
        self.ax.set_title(f"Test {self.type}")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel(self.type)
        self.ax.legend()
        plt.pause(0.1)