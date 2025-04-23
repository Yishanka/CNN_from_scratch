from cnn.core import Tensor

class Loss:
    def __call__(self, pred, true):
        ''' 调用时直接执行 forward，保存 loss: Tensor 实例 '''
        self.loss = self.forward(pred, true)
        return self.loss

    def forward(self, pred, true) -> Tensor:
        raise NotImplementedError("forward 方法未实现")

    def backward(self, retain_graph=False):
        ''' 基于保存的 self.loss 进行反向传播 '''
        assert isinstance(self.loss, Tensor), "loss 尚未计算，不能反向传播"
        assert self.loss.data.size == 1, "只能对标量调用 backward"

        # 初始化 loss 的导数为 1
        self.loss.one_grad()

        # 拓扑排序
        topo = []
        visited = set()
        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self.loss)

        # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
        for node in reversed(topo):
            node._backward()
            if not retain_graph:
                node._children.clear()  # 使用_children而不是_prev
                node._backward = lambda: None