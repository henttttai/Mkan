import torch
from torch import nn
from torch.nn.functional import silu
import math
import numpy as np


class MLP_KAN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, bias_need=True, activation_fun=silu, **kwargs):
        super(MLP_KAN_Layer, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.activation_fun = activation_fun

        self.bias = 0
        if bias_need:
            self.bias = nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, X):
        X = X.unsqueeze(1)
        X = X.repeat(1, self.out_dim, 1)
        mid_var = self.activation_fun((torch.mul(self.weight, X)) + self.bias)
        y = mid_var.sum(-1).squeeze(-1)

        return y


if __name__ == '__main__':
    net = MLP_KAN_Layer(3, 5)
    x = torch.tensor([[1.0,2.0,3.0],
                      [4.0,5.0,6.0],
                      [7.,8.,9.]])
    tmp = net(x)
    test = 1
