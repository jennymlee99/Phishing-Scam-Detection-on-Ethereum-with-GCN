import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    """
    定义GraphConvolution类的相关属性
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features      # 输入特征
        self.out_features = out_features    # 输出特征
        # 由于weight是可以训练的，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 由于bias是可以训练的，因此使用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    """
    为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子。
    """
    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数; size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    """
    此处主要定义的是本层的前向传播，通常采用的是 A ∗ X ∗ W A * X * WA∗X∗W的计算方法。由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
    """
    def forward(self, input, adj):
        # torch.mm(a, b)是矩阵a和b矩阵相乘
        support = torch.mm(input, self.weight)
        # torch.spmm(a,b)是稀疏矩阵相乘
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    """
    __repr__()方法是类的实例化对象用来做“自我介绍”的方法，默认情况下，它会返回当前对象的“类名+object at+内存地址”， 而如果对该方法进行重写，可以为其制作自定义的自我描述信息。
    """
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
