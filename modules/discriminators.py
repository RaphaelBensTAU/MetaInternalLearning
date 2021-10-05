from __future__ import absolute_import, division, print_function
from modules.networks_utils import *
from modules.backbone import MultiHeadLinearProjection, Backbone
import torch.nn.functional as F
import numpy as np
import math


class ClassicDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ClassicDiscriminator, self).__init__()
        self.opt = opt
        self.N = int(opt.nfc_d)
        self.head = ConvBlock2D(opt.nc_im, self.N, opt.ker_size, opt.ker_size // 2, stride=1, bn=False, act='lrelu')
        self.body = nn.Sequential()
        for i in range(opt.num_layers_d):
            block = ConvBlock2D(self.N, self.N, opt.ker_size, opt.ker_size // 2, stride=1, bn=False, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv2d(self.N, 1, kernel_size=opt.ker_size, padding=1, stride=1)

    def forward(self, x, original_images = None):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out

    def init_next_stage(self):
        pass

class DiscriminatorTemplate(nn.Module):
    def __init__(self, num_layers, padding, stride, N, head_layer_shape, body_layer_shape, tail_layer_shape):
        super(DiscriminatorTemplate, self).__init__()
        self.N = N
        self.padding = padding
        self.stride = stride
        self.num_layers = num_layers

        # out_dim x in_dim x kernel_size x kernel_size
        self.head_layer_shape = head_layer_shape
        self.body_layer_shape = body_layer_shape
        self.tail_layer_shape = tail_layer_shape

        self.leaky_relu_slope = 0.02
        self.leaky_relu_gain = math.sqrt(2.0 / (1 + math.pow(self.leaky_relu_slope, 2)))

    def norm_weight(self, weight, bias, gain=1.):
        out_c, in_c, k1, k2 = weight.shape
        w = weight / math.sqrt(in_c * k1 * k2) * gain
        b = bias / math.sqrt(in_c)
        return w, b

    def forward(self, x, hyper_weights):
        batch_size = x.shape[0]
        output = x.reshape(1, -1, *x.shape[2:])

        # Head
        head_w = hyper_weights[0].reshape((batch_size * self.head_layer_shape[0], *self.head_layer_shape[1:]))
        head_b = hyper_weights[1].flatten()
        head_w, head_b = self.norm_weight(head_w, head_b, gain=self.leaky_relu_gain)
        output = F.conv2d(output, head_w, head_b, padding=self.padding, stride=self.stride, groups=batch_size)
        output = F.leaky_relu(output, self.leaky_relu_slope)

        # Body
        for i in range(1, self.num_layers + 1):
            body_w = hyper_weights[2 * i].reshape((batch_size * self.body_layer_shape[0], *self.body_layer_shape[1:]))
            body_b = hyper_weights[2 * i + 1].flatten()
            body_w, body_b = self.norm_weight(body_w, body_b, gain=self.leaky_relu_gain)
            output = F.conv2d(output, body_w, body_b, padding=self.padding, stride=self.stride, groups=batch_size)
            output = F.leaky_relu(output, self.leaky_relu_slope)

        # Tail
        tail_w = hyper_weights[-2].reshape((batch_size * self.tail_layer_shape[0], *self.tail_layer_shape[1:]))
        tail_b = hyper_weights[-1].flatten()
        tail_w, tail_b = self.norm_weight(tail_w, tail_b)
        output = F.conv2d(output, tail_w, tail_b, padding=self.padding, stride=self.stride, groups=batch_size)
        output = output.reshape(batch_size, 1, output.shape[2], output.shape[3])
        return output


class HyperDiscriminator(nn.Module):
    def __init__(self, opt):
        super(HyperDiscriminator, self).__init__()
        self.opt = opt
        self.N = self.opt.nfc_d

        self.head_layer_shape = (self.N, opt.nc_im, opt.ker_size, opt.ker_size)
        self.body_layer_shape = (self.N, self.N, opt.ker_size, opt.ker_size)
        self.tail_layer_shape = (1, self.N, opt.ker_size, opt.ker_size)

        self.output_shapes = [np.prod(self.head_layer_shape), self.N] + \
                             [np.prod(self.body_layer_shape), self.N] * self.opt.num_layers_d + \
                             [np.prod(self.tail_layer_shape), 1]

        self.feature_extractor = Backbone(opt.backbone_d, pretrained=opt.pt_d, opt = opt)

        self.proj = MultiHeadLinearProjection(self.output_shapes,
                                              in_dim=self.feature_extractor.num_channels,
                                              nlayers=opt.proj_nlayers_d)

        self.disc_template = DiscriminatorTemplate(opt.num_layers_d,
                                                   padding=self.opt.padd_size, stride=1,
                                                   N=self.N,
                                                   head_layer_shape=self.head_layer_shape,
                                                   body_layer_shape=self.body_layer_shape,
                                                   tail_layer_shape=self.tail_layer_shape)

    def forward(self, x, features=None, original_images=None):
        if features is None:
            features = self.feature_extractor(original_images)
        proj_weights = self.proj(features)
        return self.disc_template(x, proj_weights)

    def init_next_stage(self):
        pass