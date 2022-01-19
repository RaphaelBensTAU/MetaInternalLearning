import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

class Backbone(nn.Module):
    def __init__(self, name: str,
                 pretrained: bool = False, opt = None):
        super(Backbone, self).__init__()
        self.backbone = getattr(torchvision.models, name)(pretrained=pretrained)
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        self.body = IntermediateLayerGetter(self.backbone, return_layers={"avgpool": 0})
        self.opt = opt
        if not self.opt.positive_embedding:
            self.linear = nn.Sequential(
                nn.Linear(self.num_channels, self.num_channels)
            )

    def forward(self, x):
        pooled = self.body(x)[0]
        out = torch.flatten(pooled, 1)
        if not self.opt.positive_embedding:
            out = self.linear(out)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=1):
        super(ProjectionHead, self).__init__()

        self.head = nn.Sequential()
        for i in range(nlayers - 1):
            self.head.add_module(f"linear_{i}", nn.Linear(in_dim, in_dim))
            self.head.add_module(f"relu_{i}", nn.ReLU())
        self.head.add_module(f"linear_final", nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.head(x)

class MultiHeadLinearProjection(nn.Module):
    def __init__(self, output_size, in_dim, nlayers=1):
        super(MultiHeadLinearProjection, self).__init__()
        self.linears = nn.ModuleList()
        self.output_size = output_size
        for i in output_size:
            self.linears.append(ProjectionHead(in_dim, i, nlayers))

    def forward(self, features):
        out = []
        # can be optimized by learning one large matrix and slicing appropriate areas instead of single standalone matrix projections. 
        for head in self.linears:
            out += [head(features)]
        return out
