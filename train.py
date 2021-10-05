from modules.utils import *
import utils
import torch
import os
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
import argparse
import random
from utils import logger, tools
import logging
import colorama
from torch.utils.data import DataLoader
import torch.optim as optim
from modules import generators as generators
from modules import discriminators as discriminators
from datasets.dataset import MultipleImageDataset
from modules.utils import calc_gradient_penalty
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def train(opt, netG, netD, data_loader, saver=None, summary=None, single_Z_init=None):
    # Current optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    if hasattr(netG, "module"):
        for proj in netG.module.proj[:-opt.train_depth]:
            for param in proj.parameters():
                param.requires_grad = False
    else:
        for proj in netG.proj[:-opt.train_depth]:
            for param in proj.parameters():
                param.requires_grad = False

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,
                                                      milestones=[int(opt.current_niter * opt.decay_iter)],
                                                      gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,
                                                      milestones=[int(opt.current_niter * opt.decay_iter)],
                                                      gamma=0.1)
    if is_master():
        progressbar_args = {
            "iterable": range(opt.current_niter),
            "desc": "Scale [{}/{}], Iteration [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1,
                                                              0, opt.current_niter),
            "train": True,
            "offset": 0,
            "logging_on_update": False,
            "logging_on_close": False,  # change to True
            "postfix": True
        }
        data_iterator = tools.create_progressbar(**progressbar_args)
    else:
        data_iterator = range(opt.current_niter)

    iterator = iter(data_loader)

    for iteration in data_iterator:
        try:
            real_pyramid, train_images = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            real_pyramid, train_images = next(iterator)

        batch_size = train_images.shape[0]
        real_pyramid = [real.to(opt.device) for real in real_pyramid]
        train_images = train_images.to(opt.device)

        Z_init = single_Z_init.repeat(train_images.shape[0], 1, 1, 1).to(opt.device)
        noise_init = utils.generate_noise(ref=Z_init, device=opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            if opt.const_amp:
                opt.Noise_Amps.append(1)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        opt.Noise_Amps.append(opt.noise_amp)
                    else:
                        z_reconstruction = netG(Z_init, opt.Noise_Amps, mode="rec", original_images=train_images)
                        MSE = F.mse_loss(real_pyramid[-1], z_reconstruction[-1])
                        noise_amp = MSE ** 0.5

                        # Use only values from main process
                        if dist.is_initialized():
                            dist.broadcast(noise_amp, src=0)

                        opt.noise_amp = noise_amp.item()
                        opt.Noise_Amps.append(opt.noise_amp)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        # train with real
        #################
        netD.zero_grad()
        output_real = netD(real_pyramid[-1], original_images=train_images)
        # train with fake
        #################
        fake = netG(noise_init, opt.Noise_Amps, mode="rand", original_images=train_images)
        output_fake = netD(fake[-1].detach(), original_images=train_images)
        # discriminator wgan-gp loss
        ############################
        errD_real = -output_real.mean()
        errD_fake = output_fake.mean()
        gradient_penalty = calc_gradient_penalty(netD, None, train_images, real_pyramid[-1], fake[-1],
                                                 opt.lambda_grad, opt.device,
                                                 opt.discriminator)
        errD_total = errD_real + errD_fake + gradient_penalty
        errD_total = errD_total.mean()
        errD_total.backward()
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(netD.parameters(), opt.grad_clip)
        optimizerD.step()
        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        # reconstruction loss
        #####################
        generated = netG(Z_init, opt.Noise_Amps, mode="rec", original_images=train_images)
        if not opt.accumulate_rec_loss:
            rec_loss = F.mse_loss(generated[-1], real_pyramid[-1])
        else:
            rec_loss = 0
            for i in range(len(generated)):
                rec_loss += F.mse_loss(generated[i], real_pyramid[i])
            rec_loss /= len(generated)
        if opt.scale_idx >= 1:
            noise_amp = rec_loss ** 0.5
            if dist.is_initialized():
                dist.all_reduce(noise_amp, dist.reduce_op.SUM)
                size = float(dist.get_world_size())
                noise_amp = noise_amp / size

            opt.noise_amp = noise_amp.item()
        opt.Noise_Amps[-1] = opt.noise_amp
        # train with Discriminator
        ##########################
        output = netD(fake[-1], original_images=train_images)
        errG = -output.mean()  * opt.disc_loss_weight
        errG_total = errG + opt.rec_weight * rec_loss
        errG_total = errG_total.mean()
        netG.zero_grad()
        errG_total.backward()
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.grad_clip)
        optimizerG.step()

        schedulerG.step()
        schedulerD.step()
        if is_master():
            data_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
                opt.scale_idx + 1, opt.stop_scale + 1,
                iteration + 1, opt.current_niter,
            ))

        if opt.visualize:
            if not opt.no_tb and is_master():
                summary.add_scalar('Image/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
                summary.add_scalar('Image/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
                summary.add_scalar('Image/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                summary.add_scalar('Image/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.mean(), iteration)
                summary.add_scalar('Image/Scale {}/errD_real'.format(opt.scale_idx), errD_real.mean(), iteration)
            num_to_generate = 10
            if iteration == len(data_iterator) - 1 or iteration % opt.print_interval == 0:
            # if iteration == len(data_iterator) - 1:
                with torch.no_grad():
                    fake_var = []
                    for _ in range(num_to_generate):
                        noise_init = utils.generate_noise(ref=noise_init)
                        fake = netG(noise_init, opt.Noise_Amps, mode="rand", original_images=train_images)
                        fake_var.append(fake[-1])
                        if opt.SAVE_IMGS:
                            for i, img in enumerate(fake[-1]):
                                plt.imsave('{}/{}_{}.png'.format(saver.experiment_dir, i, _),
                                           convert_image_np(fake[-1][i].unsqueeze(0).detach()), vmin=0, vmax=1)
                            for i, img in enumerate(generated[-1]):
                                plt.imsave('{}/rec_{}.png'.format(saver.experiment_dir, i),
                                           convert_image_np(generated[-1][i].unsqueeze(0).detach()), vmin=0, vmax=1)

                    fake_var = torch.cat(fake_var, dim=0).cpu()
                    fake_test_var = []

                    if not opt.no_tb and is_master():
                        summary.visualize_image(opt, iteration, real_pyramid[-1], 'Real',
                                                batch_lim=batch_size, nrow=batch_size)
                        summary.visualize_image(opt, iteration, generated[-1], 'Reconstructed',
                                                batch_lim=batch_size, nrow=batch_size)
                        summary.visualize_image(opt, iteration, fake_var, 'Generated',
                                                batch_lim=batch_size * num_to_generate, nrow=batch_size)
                        summary.visualize_image(opt, iteration, train_images, 'Original images',
                                                batch_lim=batch_size, nrow=batch_size)

    if is_master():
        data_iterator.close()
        if opt.SAVE_MODEL:
            g_state_dict = netG.module.state_dict() if hasattr(netG, "module") else netG.state_dict()
            saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
            saver.save_checkpoint({
                'scale': opt.scale_idx,
                'state_dict': g_state_dict,
                'optimizer': optimizerG.state_dict(),
                'noise_amps': opt.Noise_Amps,
                'current_niter': opt.current_niter,
            }, 'netG.pth')
            if opt.SAVE_MODEL_D:
                d_state_dict = netD.module.state_dict() if hasattr(netD, "module") else netD.state_dict()
                saver.save_checkpoint({
                    'scale': opt.scale_idx,
                    'state_dict': d_state_dict,
                    'optimizer': optimizerD.state_dict(),
                }, 'netD.pth')


def main(opt, init_distributed=False):
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.device_id)
        torch.cuda.init()
        opt.device = torch.device("cuda")

    if init_distributed:
        dist.init_process_group(
            backend=opt.backend,
            init_method=opt.init_method,
            world_size=opt.world_size,
            rank=opt.rank,
        )
        dist.all_reduce(torch.zeros(1).cuda())
        opt.device = torch.device("cuda", opt.rank)

    # saver = None
    summary = None
    opt.run_id = opt.run_id if opt.run_id >= 0 else None
    if is_master():
        # Define Saver
        saver = utils.ImageSaver(opt, run_id=opt.run_id)

        # Define Tensorboard Summary
        summary = utils.TensorboardSummary(saver.experiment_dir)
        logger.configure_logging(os.path.abspath(os.path.join(saver.experiment_dir, 'logbook.txt')))

        # Save args
        with open(os.path.join(saver.experiment_dir, 'args.txt'), 'w') as args_file:
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list, bool):
                    args_file.write('{}: {}\n'.format(argument, value))

        # Print args
        with logger.LoggingBlock("Commandline Arguments", emph=True):
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list):
                    logging.info('{}: {}'.format(argument, value))

        with logger.LoggingBlock("Experiment Summary", emph=True):
            video_file_name, checkname, experiment = saver.experiment_dir.split('/')[-3:]
            logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
            logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

            with logger.LoggingBlock("Commandline Summary", emph=True):
                logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
                logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
                logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))

    else:
        # Define Saver
        saver = utils.ImageSaver(opt, run_id=opt.run_id)

    # Data
    dataset = MultipleImageDataset(opt)



    # Current networks
    assert hasattr(generators, opt.generator)
    assert hasattr(discriminators, opt.discriminator)
    netG = getattr(generators, opt.generator)(opt).to(opt.device)
    netD = getattr(discriminators, opt.discriminator)(opt).to(opt.device)
    #globalD = getattr(global_discr, "GlobalDiscriminator")(opt).to(opt.device)

    if opt.resume and opt.run_id is not None:
        opt.netG = os.path.join(saver.experiment_dir, "netG.pth")
        if not os.path.isfile(opt.netG):
            opt.netG = ''

    if opt.netG != '':
        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = torch.load(opt.netG)
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
        opt.current_niter = max(opt.niter_min, int(checkpoint['current_niter'] * opt.niter_decay))
        for _ in range(1, opt.scale_idx + 1):
            netG.init_next_stage(_)
        netG.load_state_dict(checkpoint['state_dict'])
        opt.scale_idx += 1

        # NoiseAmp
        opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
        single_Z_init = torch.load(os.path.join(opt.resume_dir, 'single_Z_init.pth'))['data']

    else:
        opt.current_niter = opt.niter
        single_Z_init = None
        opt.resumed_idx = -1

    if opt.netD != '':
        if not os.path.isfile(opt.netD):
            raise RuntimeError("=> no <D> checkpoint found at '{}'".format(opt.netG))
        checkpoint_D = torch.load(opt.netD)
        netD.load_state_dict(checkpoint_D['state_dict'])


    if single_Z_init is None:
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        if dist.is_initialized():
            single_Z_init = utils.generate_noise(size=[1, 3, *initial_size])
            if dist.get_rank() == 0:
                saver.save_checkpoint({'data': single_Z_init}, 'single_Z_init.pth')
            single_Z_init = single_Z_init.to(opt.device)
            dist.broadcast(single_Z_init, src=0)
        else:
            single_Z_init = utils.generate_noise(size=[1, 3, *initial_size])
            saver.save_checkpoint({'data': single_Z_init}, 'single_Z_init.pth')
            single_Z_init = single_Z_init.to(opt.device)

    # Parallel
    if torch.cuda.is_available():
        if opt.world_size > 1:
            if opt.sync_bn:
                netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
                netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD)
            netG = DDP(
                netG,
                device_ids=[opt.rank],
                output_device=opt.rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            netD = DDP(
                netD,
                device_ids=[opt.rank],
                output_device=opt.rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        else:
            netG = netG.to(opt.device)
            netD = netD.to(opt.device)

    start_time = time.time()

    curr_batch_size = opt.batch_size
    while opt.scale_idx < opt.stop_scale + 1:
        if opt.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=opt.no_shuffle,
            )
            data_loader = DataLoader(dataset,
                                     shuffle=False,
                                     drop_last=True,
                                     batch_size=curr_batch_size,
                                     num_workers=4,
                                     sampler=train_sampler,
                                     pin_memory=True)
        else:
            data_loader = DataLoader(dataset,
                                     shuffle=opt.no_shuffle,
                                     drop_last=True,
                                     batch_size=curr_batch_size,
                                     num_workers=4,
                                     pin_memory=True)

        if dist.is_initialized():
            train_sampler.set_epoch(opt.scale_idx)
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            # if isinstance(globalD, DDP):
            #     globalD.module.init_next_stage()
            #     globalD.cuda()
            # else:
            #     globalD.init_next_stage()
            #     globalD.cuda()

            if isinstance(netG, DDP):
                netG.module.init_next_stage()
                netG.cuda()
            else:
                netG.init_next_stage()
                netG.cuda()

        train(opt, netG, netD, data_loader,saver=saver, summary=summary, single_Z_init=single_Z_init)
        opt.scale_idx += 1
        opt.current_niter = max(opt.niter_min, int(opt.current_niter * opt.niter_decay))
        if "cuda" in str(opt.device):
            torch.cuda.empty_cache()
    with open("{}/runtime.txt".format(saver.experiment_dir), 'w+') as f:
        f.write('{}'.format((time.time() - start_time) // 60))


def distributed_main(device_id, opt):
    opt.device_id = device_id

    if opt.rank is None:
        opt.rank = opt.start_rank + device_id

    main(opt, init_distributed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc-im', type=int, default=3, help='# channels')
    parser.add_argument('--nfc-g', type=int, default=64, help='generator basic # channels')
    parser.add_argument('--nfc-d', type=int, default=64, help='discriminator basic # channels')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=0, help='net pad size')
    parser.add_argument('--num-layers-g', type=int, default=3, help='number of layers for generator')
    parser.add_argument('--num-layers-d', type=int, default=3, help='number of layers for discriminator')
    parser.add_argument('--generator', type=str, default='MultiScaleHyperGenerator', help='generator model')
    parser.add_argument('--discriminator', type=str, default='HyperDiscriminator', help='discriminator model')
    parser.add_argument('--backbone-g', type=str, default='resnet34', help='generator backbone')
    parser.add_argument('--backbone-d', type=str, default='resnet34', help='discriminator backbone')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='sync batchnorm')
    parser.add_argument('--pt-g', action='store_true', default=False, help='sync batchnorm')
    parser.add_argument('--pt-d', action='store_true', default=False, help='sync batchnorm')
    parser.add_argument('--proj-nlayers-g', type=int, default=1, help='# channels')
    parser.add_argument('--proj-nlayers-d', type=int, default=1, help='# channels')
    parser.add_argument('--qdim', type=int, default=2, help='# channels')
    parser.add_argument('--learn-noise', action='store_true', default=False, help='sync batchnorm')
    parser.add_argument('--freeze-noise', action='store_true', default=False, help='sync batchnorm')
    parser.add_argument('--scale-factor', type=float, default=0.60, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=25, help='image minimal size at the coarser scale')
    parser.add_argument('--train-img-size', type=float, default=128, help='pyramid scale factor')
    parser.add_argument('--niter', type=int, default=4000, help='number of iterations to train per scale')
    parser.add_argument('--niter-min', type=int, default=10, help='number of iterations to train per scale')
    parser.add_argument('--niter-decay', type=int, default=1, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.00005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.00005, help='learning rate, default=0.0005')
    parser.add_argument('--accumulate-rec-loss', action='store_true', default=False,
                        help='to accumulate reconstruction losses between scales')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=50., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--norm-g', type=str, default='none', help='spectral-norm for generator')
    parser.add_argument('--norm-d', type=str, default='none', help='spectral-norm for discriminator')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--mutual-until-index', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--mutual-from-index', type=int, default=10, help='how many layers are trained if growing')
    parser.add_argument('--decay-iter', type=float, default=0.8, help='gradient clip')
    parser.add_argument('--grad-clip', type=float, default=1, help='gradient clip')
    parser.add_argument('--const-amp', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--positive-embedding', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--SAVE-MODEL', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--SAVE-MODEL-D', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--no-shuffle', action='store_false', default=True, help='shuffle')
    parser.add_argument('--img-size', type=int, default=250)
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')
    parser.add_argument('--ar', type=float, default=0.75)
    parser.add_argument('--pth-result', type=str, default='DEBUG', help='check name')
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')
    parser.add_argument('--no-tb', action='store_true', default=False, help='no tensorboard')
    parser.add_argument('--run-id', type=int, default=-1, help='experiment id')
    parser.add_argument('--resume', action='store_true', default=False, help='resume (only with --single-exp)')
    parser.add_argument('--local_rank', type=int, default=1, help='?????')
    parser.add_argument('--SAVE-IMGS', action='store_true', default=False, help='save imgs (for single and batch)')
    parser.add_argument('--file-suffix', type=str, default='jpg', help='')



    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    # Initial config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    utils.adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    # Distributed
    opt.world_size = torch.cuda.device_count()

    if opt.world_size == 1:
        opt.device_id = 0
        main(opt)
    if opt.world_size > 1:
        opt.batch_size = int(opt.batch_size / opt.world_size)
        port = random.randint(10000, 20000)
        opt.init_method = f"tcp://localhost:{port}"
        # opt.init_method = "env://"
        opt.rank = None
        opt.start_rank = 0
        opt.backend = 'nccl'
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(opt,),
            nprocs=opt.world_size,
        )
    else:
        opt.device = torch.device("cpu")
        main(opt)
