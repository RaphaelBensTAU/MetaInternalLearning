import torch
from torch import autograd
from skimage import color, morphology, filters
import numpy as np
import matplotlib.pyplot as plt

from torch import distributed as dist


def plot_histogram(values, num):
    plt.ioff()
    values = values.detach().cpu().numpy()
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=values, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Activation layer {}'.format(num))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.savefig(opt.directory+"/here.png")
    plt.show()
    plt.close()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    # if not(opt.not_cuda):
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor)
    x = norm(x)
    return x

def convert_image_np(inp, to_denorm=True):
    if inp.shape[1]==3:
        if to_denorm:
            inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        if to_denorm:
            inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    inp = np.clip(inp,0,1)
    return inp

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*x
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def dilate_mask(mask, opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)

    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask

def calc_gradient_penalty(netD, features, original_images, real_data, fake_data, LAMBDA, device, discriminator_type):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    if "Classic" in discriminator_type:
        disc_interpolates = netD(interpolates)
    else:
        if features == None:
            if hasattr(netD, "module"):
                features = netD.module.feature_extractor(original_images)
            else:
                features = netD.feature_extractor(original_images)
        disc_interpolates = netD(interpolates, features)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
