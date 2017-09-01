import torch
from torch.autograd import Function


def sign(input):
    func = Sign()
    return func(input)


class Sign(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, input):
        prob = input.new(input.size()).uniform_()
        x = input.clone()
        x[(1 - input) / 2 <= prob] = 1
        x[(1 - input) / 2 > prob] = -1
        return x

    def backward(self, grad_output):
        return grad_output
