import os
import torch
import glob
from pathlib import Path

class ImageSaver(object):
    def __init__(self, opt, run_id=None):
        self.opt = opt
        clip_name = '.'.join(opt.image_path.split('/')[-1].split('.')[:-1])
        self.directory = os.path.join('run', clip_name, opt.checkname)
        if run_id is None:
            self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')), key=os.path.getctime)
            run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        # if not os.path.exists(self.experiment_dir):
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.eval_dir = os.path.join(self.experiment_dir, "eval")
        # if not os.path.exists(self.eval_dir):
        os.makedirs(self.eval_dir, exist_ok=True)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        filename = os.path.join(self.experiment_dir, filename)
        return torch.load(filename)
