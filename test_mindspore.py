import mindspore
from mindspore import Tensor, ops
from mindspore import context
from mindspore.ops.operations import Add

class Net(mindspore.nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = Add()

    def construct(self, x, y):
        out = self.add(x, y)
        out = self.add(out, x)
        out = self.add(out, y)
        return out

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    x = Tensor([[1, 2], [3, 4]], mindspore.float32)
    y = Tensor([[5, 6], [7, 8]], mindspore.float32)
    graph = mindspore.ops.export_graph(net, x, y)
    for node in graph.nodes:
        print(node.name)

