import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t, range=None):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def add_scalar(self, *args):
        self.writer.add_scalar(*args)

    def visualize_image(self, opt, global_step, images, name, batch_lim=3, nrow=3):
        grid_image = make_grid(images[:batch_lim, :, :, :].clone().cpu().data, nrow, normalize=True, scale_each=True)
        self.writer.add_image('Image/Scale {}/{}'.format(opt.scale_idx, name), grid_image, global_step)

    def visualize_generated(self, opt, global_step, images, name, img_by_rows):
        grid_image = make_grid(images[:, :, :, :].clone().cpu().data, img_by_rows, normalize=True, scale_each=True)
        self.writer.add_image('Image/Scale {}/{}'.format(opt.scale_idx, name), grid_image, global_step)
