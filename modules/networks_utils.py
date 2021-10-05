from __future__ import absolute_import, division, print_function
import numpy as np
from torch import nn

def conv_weights_init_ones(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') == 0 or classname.find('Conv2d') == 0:
        m.weight.data.fill_(1 / np.prod(m.kernel_size))
        m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        try:
            m.weight.data.normal_(1.0, 0.02)
        except:
            m.module.weight.data.normal_(1.0, 0.02)

        try:
            m.bias.data.fill_(0)
        except:
            m.module.bias.data.fill_(0)

def get_activation(act):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "lrelu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=True)
    }
    return activations[act]


class ConvBlock2D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock2D, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                          stride=stride, padding=padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        if act is not None:
            self.add_module(act, get_activation(act))

