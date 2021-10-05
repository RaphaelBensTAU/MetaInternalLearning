import torch
import torch.nn.functional as F
import math
import numpy as np

def interpolate(input, size=None, scale_factor=None, interpolation='bilinear'):
    if input.dim() == 5:
        b, c, t, h0, w0 = input.shape
        img = input.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B+T)CHW
        scaled = F.interpolate(img, size=size, scale_factor=scale_factor, mode=interpolation, align_corners=True)
        _, _, h1, w1 = scaled.shape
        scaled = scaled.reshape(b, t, c, h1, w1).permute(0, 2, 1, 3, 4)
    else:
        scaled = F.interpolate(input, size=size, scale_factor=scale_factor, mode=interpolation, align_corners=True)

    return scaled


# def adjust_scales2image(size, opt):
#     opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
#     scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
#     opt.stop_scale = opt.num_scales - scale2stop
#     opt.scale1 = min(opt.max_size / size, 1)
#     opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
#     scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
#     opt.stop_scale = opt.num_scales - scale2stop
def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.img_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.img_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.img_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop


def generate_noise(ref=None, size=None, type='normal', emb_size=None, device=None):
    # Initiate noise without batch size
    if ref is not None:
        noise = torch.zeros_like(ref)
    elif size is not None:
        noise = torch.zeros(*size).to(device)
    else:
        raise Exception("ref or size must be applied")

    if type == 'normal':
        return noise.normal_(0, 1)
    elif type == 'benoulli':
        return noise.bernoulli_(0.5)

    if type == 'int':
        assert (emb_size is not None) and (size is not None) and (device is not None)
        return torch.randint(0, emb_size, size=size, device=device)

    return noise.uniform_(0, 1)  # Default == Uniform

def get_scales_by_index(index, scale_factor, stop_scale, img_size):
    scale = math.pow(scale_factor, stop_scale - index)
    s_size = math.ceil(scale * img_size)
    return s_size

def get_scales_by_index2(i, scale_factor, stop_scale, img_size):
    if i != stop_scale:
        scale = math.pow(scale_factor, ((stop_scale - 1) / math.log(stop_scale)) * math.log(stop_scale - i) + 1)
    else:
        scale = 1
    s_size = math.ceil(scale * img_size)
    return s_size

# def get_scales_by_index3(i, scale_factor, stop_scale, img_size):
#     if i == 0:
#         return 29
#     return int(np.linspace(45, 256, 6)[i-1])

def upscale(image, index, opt):
    assert index > 0
    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, opt.img_size)
    next_shape = [int(next_shape * opt.ar), next_shape]
    # Video interpolation
    # print(next_shape)
    img_up = interpolate(image, size=next_shape)
    return img_up

def upscale_any(image, index, opt, w_h_ratio, img_size):
    assert index > 0
    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, img_size)
    next_shape = [int(next_shape * w_h_ratio), next_shape]
    # Video interpolation
    img_up = interpolate(image, size=next_shape)
    return img_up
