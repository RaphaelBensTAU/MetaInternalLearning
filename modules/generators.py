from __future__ import absolute_import, division, print_function
import torch
import utils
from modules.backbone import MultiHeadLinearProjection, Backbone
from modules.networks_utils import *
import torch.nn.functional as F
import copy
import math
from string import ascii_lowercase

class GeneratorTemplate(nn.Module):
    def __init__(self, num_layers, padding, stride, N, head_layer_shape, body_layer_shape, tail_layer_shape):
        super(GeneratorTemplate, self).__init__()
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
        self.tanh_gain = 5.0 / 3

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
        # plot_histogram(head_w.flatten(), 1)
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
        tail_w, tail_b = self.norm_weight(tail_w, tail_b, gain=self.tanh_gain)
        output = F.conv2d(output, tail_w, tail_b, padding=self.padding, stride=self.stride, groups=batch_size)
        output = torch.tanh(output)

        output = output.reshape(batch_size, 3, output.shape[2], output.shape[3])

        return output


class MultiScaleHyperGenerator(nn.Module):
    def __init__(self, opt):
        super(MultiScaleHyperGenerator, self).__init__()
        self.opt = opt
        self.N = int(opt.nfc_g)
        self.p2d_once = (1, 1,
                         1, 1)
        self.p2d = (self.opt.num_layers_g + 2, self.opt.num_layers_g + 2,
                    self.opt.num_layers_g + 2, self.opt.num_layers_g + 2)
        self.mutual_until_index = opt.mutual_until_index

        self.head_layer_shape = (self.N, opt.nc_im, opt.ker_size, opt.ker_size)
        self.body_layer_shape = (self.N, self.N, opt.ker_size, opt.ker_size)
        self.tail_layer_shape = (opt.nc_im, self.N, opt.ker_size, opt.ker_size)

        self.output_shapes = [np.prod(self.head_layer_shape), self.N] + \
                             [np.prod(self.body_layer_shape), self.N] * self.opt.num_layers_g + \
                             [np.prod(self.tail_layer_shape), opt.nc_im]

        self.feature_extractor = Backbone(opt.backbone_g, pretrained=opt.pt_g, opt = opt)

        self.proj = nn.ModuleList()
        self.proj.append(
            MultiHeadLinearProjection(self.output_shapes,
                                      in_dim=self.feature_extractor.num_channels,
                                      nlayers=opt.proj_nlayers_g))

        self.gen_template = GeneratorTemplate(opt.num_layers_g,
                                              padding=self.opt.padd_size, stride=1,
                                              N=self.N,
                                              head_layer_shape=self.head_layer_shape,
                                              body_layer_shape=self.body_layer_shape,
                                              tail_layer_shape=self.tail_layer_shape)

    def init_next_stage(self, _=None):
        if _ is None:
            _ = self.opt.scale_idx
        if _ >= self.mutual_until_index:
            self.proj.append(copy.deepcopy(self.proj[-1]))

    def forward(self, noise_init, noise_amp, mode='rand', features=None, original_images=None):
        results = []
        if features == None:
            features = self.feature_extractor(original_images)
        proj_weights = self.proj[0](features)
        noise_init_2 = F.pad(noise_init, self.p2d)
        x_prev_out = self.gen_template(noise_init_2 , proj_weights)
        results.append(x_prev_out)
        for idx in range(1, self.opt.scale_idx + 1):
            body_to_apply = 0 if idx < self.mutual_until_index else idx - self.mutual_until_index + 1
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)
            proj_weights = self.proj[body_to_apply](features)
            if mode == 'rand':
                x_prev_out_up_2 = F.pad(x_prev_out_up, self.p2d)
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = self.gen_template(x_prev_out_up_2 + noise * noise_amp[idx], proj_weights)
            else:
                x_prev_out_up_2 = F.pad(x_prev_out_up, self.p2d)
                x_prev = self.gen_template(x_prev_out_up_2, proj_weights)
            x_prev_out = x_prev + x_prev_out_up
            results.append(x_prev_out)
        return results

    def forward_interpolate_features(self, noise_init, noise_amp, original_images, idx_to_inject, alpha):
        results = []
        features = self.feature_extractor(original_images)
        features_inter = torch.lerp(features[0], features[1], alpha)
        features_to_apply = features[0] if idx_to_inject > 0 else features_inter
        proj_weights = self.proj[0](features_to_apply)
        x_prev_out = self.gen_template(F.pad(noise_init[0], self.p2d), proj_weights)
        results.append(x_prev_out)
        for idx in range(1, self.opt.scale_idx + 1):
            body_to_apply = 0 if idx < self.mutual_until_index else idx - self.mutual_until_index + 1
            features_to_apply = features[0] if idx_to_inject > idx else features_inter
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)
            proj_weights = self.proj[body_to_apply](features_to_apply)
            x_prev_out_up_2 = F.pad(x_prev_out_up, self.p2d)
            x_prev = self.gen_template(x_prev_out_up_2 + noise_init[idx] * noise_amp[idx], proj_weights)
            x_prev_out = x_prev + x_prev_out_up
            results.append(x_prev_out)
        return results

    def forward_interpolate_weight(self, noise_init, noise_amp, original_images, idx_to_inject, alpha):
        results = []
        features = self.feature_extractor(original_images)
        proj_weights = self.proj[0](features)
        weights_to_apply = [proj[0] for proj in proj_weights] if idx_to_inject > 0 else [
            torch.lerp(proj[0], proj[1], alpha) for proj in proj_weights]
        x_prev_out = self.gen_template(F.pad(noise_init, self.p2d), weights_to_apply)
        results.append(x_prev_out)
        for idx in range(1, self.opt.scale_idx + 1):
            body_to_apply = 0 if idx < self.mutual_until_index else idx - self.mutual_until_index + 1
            # In-domain
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)
            proj_weights = self.proj[body_to_apply](features)
            weights_to_apply = [proj[0] for proj in proj_weights] if idx_to_inject > idx else [
                torch.lerp(proj[0], proj[1], alpha) for proj in proj_weights]
            # In-domain
            x_prev_out_up_2 = F.pad(x_prev_out_up, self.p2d)
            noise = utils.generate_noise(ref=x_prev_out_up_2)
            x_prev = self.gen_template(x_prev_out_up_2 + noise * noise_amp[idx], weights_to_apply)
            # In-domain
            x_prev_out = x_prev + x_prev_out_up
            results.append(x_prev_out)
        return results

    def forward_inject_image(self, noise_init, index, original_images, img_to_inject=None):
        results = []
        features = self.feature_extractor(original_images)
        proj_weights = self.proj[0](features)
        x_prev_out = self.gen_template(F.pad(noise_init, self.p2d), proj_weights)
        results.append(x_prev_out)
        for idx in range(1, self.opt.scale_idx + 1):
            body_to_apply = 0 if idx < self.mutual_until_index else idx - self.mutual_until_index + 1
            if idx == index:
                x_prev_out_up = utils.upscale(img_to_inject, idx, self.opt)
            else:
                x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)

            proj_weights = self.proj[body_to_apply](features)
            x_prev = self.gen_template(F.pad(x_prev_out_up, self.p2d), proj_weights)

            x_prev_out = x_prev + x_prev_out_up
            results.append(x_prev_out)
        return results

    def forward_diff_size(self, noise_init, noise_amp, h_w_ratio, img_size, mode='rand', original_images=None):
        features = self.feature_extractor(original_images)
        proj_weights = self.proj[0](features)
        noise_init_2 = F.pad(noise_init, self.p2d)
        x_prev_out = self.gen_template(noise_init_2 , proj_weights)
        for idx in range(1, self.opt.scale_idx + 1):
            body_to_apply = 0 if idx < self.mutual_until_index else idx - self.mutual_until_index + 1
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)
            proj_weights = self.proj[body_to_apply](features)
            if mode == 'rand':
                x_prev_out_up_2 = F.pad(x_prev_out_up, self.p2d)
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = self.gen_template(x_prev_out_up_2 + noise * noise_amp[idx], proj_weights)
            x_prev_out = x_prev + x_prev_out_up
        return x_prev_out
