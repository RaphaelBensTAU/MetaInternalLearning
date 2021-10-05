from torch.utils.data import Dataset
import imageio
import kornia as K
import utils
import cv2
import logging
import os
from glob import glob
from modules.imresize import *

class MultipleImageDataset(Dataset):
    def __init__(self, opt):
        super(MultipleImageDataset, self).__init__()
        self.opt = opt
        self.image_dir_path = opt.image_path
        if not os.path.exists(opt.image_path):
            logging.error("invalid path")
            exit(0)

        pyramid_base_sizes = [utils.get_scales_by_index(i, self.opt.scale_factor, self.opt.stop_scale,
                                                        self.opt.img_size) for i in range(self.opt.num_scales + 1)]
        self.image_dir = sorted(glob(os.path.join(self.image_dir_path, '*.{}'.format(opt.file_suffix))))

        if len(self.image_dir) == 1:
            img = imageio.imread(self.image_dir[0])[:,:,:3]
            opt.ar = img.shape[0]/ img.shape[1]
        self.original = None

        self.scaled_sizes = [[int(base_size * self.opt.ar), base_size] for base_size in pyramid_base_sizes]

    def __len__(self):
        return len(self.image_dir) * self.opt.data_rep

    def __getitem__(self, idx):
        idx = idx % len(self.image_dir)
        image = imageio.imread(self.image_dir[idx])[:, :, :3]

        scaled_size = [128, 128]
        training_image = cv2.resize(image, tuple(scaled_size[::-1]))
        training_image = K.image_to_tensor(training_image).float()
        training_image = training_image / 255  # Set range [0, 1]
        training_image = K.normalize(training_image, 0.5, 0.5)

        tensor_img = K.image_to_tensor(image).float()
        tensor_img /= 255
        tensor_img = [K.normalize(imresize(tensor_img, scaled_size, self.opt).squeeze(0), 0.5, 0.5) for scaled_size in
                      self.scaled_sizes[:self.opt.scale_idx + 1]]
        return tensor_img, training_image
