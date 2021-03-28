"""
@author: Bossun Wang
@date: 20210321
@contact: vvmodouco@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

import glob
import random
import os
from PIL import Image
import math
import shutil
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import *
from models import *
from utils import *


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_dataset = ImageDataset(args.content_dir, args.style_dir, mode='train')
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size, shuffle=True, num_workers=4)

    # models
    in_nc = 3
    out_nc = 3
    generator = Generator(in_nc, out_nc, args.n_styles, args.ngf)
    discriminator = Discriminator(in_nc, args.n_styles, args.ndf)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=args.lr, betas=(0.5, 0.999))

    # init losses
    if args.use_lsgan:
        criterion_GAN = nn.MSELoss()
    else:
        criterion_GAN = nn.BCELoss()

    criterion_ACGAN = nn.CrossEntropyLoss().to(device)
    criterion_Rec = nn.L1Loss().to(device)
    criterion_TV = TVLoss(TVLoss_weight=args.tv_strength).to(device)

    ori_epoch = 0
    if args.resume:
        checkpoint = torch.load(conf.pretrain_model, map_location=conf.device)
        assert "epoch" in checkpoint \
               and "generator" in checkpoint \
               and "discriminator" in checkpoint \
               and "optimizerG" in checkpoint \
               and "optimizerD" in checkpoint
        ori_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    # set DataParallel for models
    generator = torch.nn.DataParallel(generator).to(device)
    discriminator = torch.nn.DataParallel(discriminator).to(device)

    # LR schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(args.epochs
                                                                          , ori_epoch
                                                                          , args.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(args.epochs
                                                                          , ori_epoch
                                                                          , args.decay_epoch).step)

    # init Weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Set vars for training
    batch = next(iter(dataloader))

    input_A = torch.zeros((args.batch_size, in_nc, args.fineSize, args.fineSize)).to(device)
    input_B = torch.zeros((args.batch_size, out_nc, args.fineSize, args.fineSize)).to(device)
    target_real = Variable(torch.tensor(args.batch_size).fill_(1.0), requires_grad=False).to(device)
    target_fake = Variable(torch.tensor(args.batch_size).fill_(0.0), requires_grad=False).to(device)

    D_A_size = discriminator(input_A.copy_(batch['style']))[0].size()
    D_AC_size = discriminator(input_B.copy_(batch['style']))[1].size()

    print('D_A_size:', D_A_size)
    print('D_AC_size:', D_AC_size)

    class_label_B = torch.zeros(D_AC_size[0], D_AC_size[1], D_AC_size[2]).long().to(device)

    autoflag_OHE = torch.zeros(1, args.n_styles + 1).long().to(device)
    # assigned last one to content label
    autoflag_OHE[0][-1] = 1

    fake_label = torch.zeros(D_A_size).fill_(0.0).to(device)
    real_label = torch.zeros(D_A_size).fill_(0.99).to(device)
    fake_buffer = ReplayBuffer()

    generator_loss_meter = AverageMeter()
    reconstruction_loss_meter = AverageMeter()
    loss_g_gan_meter = AverageMeter()
    loss_g_ac_meter = AverageMeter()
    discriminator_gan_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()
    discriminator_class_loss_meter = AverageMeter()

    # TRAIN LOOP
    for epoch in range(ori_epoch, args.epochs):
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            global_batch_idx = epoch * len(dataloader) + batch_idx

            # Unpack minibatch
            # source content
            real_content = batch['content'].to(device)
            # target style
            real_style = batch['style'].to(device)
            # style label
            style_label = batch['style_label'].to(device)
            # one-hot encoded style
            style_OHE = F.one_hot(style_label, args.n_styles).long().to(device)
            # style Label mapped over 1x19x19 tensor for patch discriminator
            class_label = class_label_B.copy_(label2tensor(style_label, class_label_B)).long().to(device)

            # Update Discriminator
            optimizer_D.zero_grad()

            # Generate style-transfered image
            genfake = generator({
                'content': real_content,
                'style_label': style_OHE})

            # Add generated image to image pool and randomly sample pool
            fake = fake_buffer.push_and_pop(genfake)
            # Discriminator forward pass with sampled fake
            out_gan, out_class = discriminator(fake)

            # Discriminator Fake loss (correctly identify generated images)
            errD_fake = criterion_GAN(out_gan, fake_label)
            # Backward pass and parameter optimization
            errD_fake.backward()
            optimizer_D.step()

            optimizer_D.zero_grad()
            # Discriminator forward pass with target style
            out_gan, out_class = discriminator(real_style)
            # Discriminator Style Classification loss
            errD_real_class = criterion_ACGAN(out_class.transpose(1, 3), class_label) * args.lambda_A
            # Discriminator Real loss (correctly identify real style images)
            errD_real = criterion_GAN(out_gan, real_label)
            errD_real_total = errD_real + errD_real_class
            # Backward pass and parameter optimization
            errD_real_total.backward()
            optimizer_D.step()

            errD = (errD_real + errD_fake) / 2.0

            # Generator Update
            # Style Transfer Loss
            optimizer_G.zero_grad()

            # Discriminator forward pass with generated style transfer
            out_gan, out_class = discriminator(genfake)

            # Generator gan (real/fake) loss
            err_gan = criterion_GAN(out_gan, real_label)
            # Generator style class loss
            err_class = criterion_ACGAN(out_class.transpose(1, 3), class_label) * args.lambda_A
            # Total Variation loss
            err_TV = criterion_TV(genfake)

            errG_tot = err_gan + err_class + err_TV

            errG_tot.backward()
            optimizer_G.step()

            # Auto-Encoder (Recreation) Loss
            optimizer_G.zero_grad()
            identity = generator({
                'content': real_content,
                'style_label': autoflag_OHE,
            })
            err_ae = criterion_Rec(identity, real_content) * args.autoencoder_constrain
            err_ae.backward()
            optimizer_G.step()

            generator_loss_meter.update(errG_tot.item(), args.batch_size)
            reconstruction_loss_meter.update(err_ae.item(), args.batch_size)
            loss_g_gan_meter.update(err_gan.item(), args.batch_size)
            loss_g_ac_meter.update(err_class.item(), args.batch_size)
            discriminator_gan_loss_meter.update(errD.item(), args.batch_size)
            tv_loss_meter.update(err_TV.item(), args.batch_size)
            discriminator_class_loss_meter.update(errD_real_class.item(), args.batch_size)

            if batch_idx % args.print_freq == 0:
                generator_loss_val = generator_loss_meter.avg
                reconstruction_loss_val = reconstruction_loss_meter.avg
                loss_g_gan_val = loss_g_gan_meter.avg
                loss_g_ac_val = loss_g_ac_meter.avg
                discriminator_gan_loss_val = discriminator_gan_loss_meter.avg
                tv_loss_val = tv_loss_meter.avg
                discriminator_class_loss_val = discriminator_class_loss_meter.avg

                lr_G = get_lr(optimizer_G)
                lr_D = get_lr(optimizer_D)

                print('Epoch %d, iter %d / %d, lr G %f, lr D %f'
                      ', Generator Loss %f'
                      ', Reconstruction Loss %f'
                      ', loss_G_GAN %f'
                      ', loss_G_AC %f'
                      ', Discriminator GAN Loss %f'
                      ', tv_loss %f'
                      ', Discriminator Class Loss %f' %
                      (epoch, batch_idx, len(dataloader), lr_G, lr_D
                       , generator_loss_val
                       , reconstruction_loss_val
                       , loss_g_gan_val
                       , loss_g_ac_val
                       , discriminator_gan_loss_val
                       , tv_loss_val
                       , discriminator_class_loss_val))

                plt.imsave(os.path.join(args.log_dir, 'images', 'content_%d.png' % epoch)
                           , (real_content[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'style_%d.png' % epoch)
                           , (real_style[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'transfer_%d.png' % epoch)
                           , (genfake[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'auto-reconstruction_%d.png' % epoch)
                           , (identity[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)

                # record to tensorboard
                args.writer.add_scalar('lr_G', lr_G, global_batch_idx)
                args.writer.add_scalar('lr_D', lr_G, global_batch_idx)
                args.writer.add_scalar('Generator Loss', generator_loss_val, global_batch_idx)
                args.writer.add_scalar('Reconstruction Loss', reconstruction_loss_val, global_batch_idx)
                args.writer.add_scalar('loss_G_GAN', loss_g_gan_val, global_batch_idx)
                args.writer.add_scalar('loss_G_AC', loss_g_ac_val, global_batch_idx)
                args.writer.add_scalar('Discriminator GAN Loss', discriminator_gan_loss_val, global_batch_idx)
                args.writer.add_scalar('tv_loss', tv_loss_val, global_batch_idx)
                args.writer.add_scalar('Discriminator Class Loss', discriminator_class_loss_val, global_batch_idx)

                generator_loss_meter.reset()
                reconstruction_loss_meter.reset()
                loss_g_gan_meter.reset()
                loss_g_ac_meter.reset()
                discriminator_gan_loss_meter.reset()
                tv_loss_meter.reset()
                discriminator_class_loss_meter.reset()

        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            # Save model
            saved_name = 'GatedGAN_Epoch_%d.pt' % epoch
            state = {
                'generator': generator.module.state_dict()
                , 'discriminator': discriminator.state_dict()
                , 'optimizer_G': optimizer_G.state_dict()
                , 'optimizer_D': optimizer_D.state_dict()
                , 'epoch': epoch
            }

            torch.save(state, os.path.join(args.out_dir, saved_name))
            print('save checkpoint %s to disk...' % saved_name)


if __name__ == '__main__':
    # TRAIN OPTIONS FROM GATED GAN

    conf = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    conf.add_argument("--content_dir", type=str,
                      help="The content root folder of training set.")
    conf.add_argument("--style_dir", type=str,
                      help="The style root folder of training set.")
    conf.add_argument('--lr', type=float, default=2e-4,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str, default='saved_models',
                      help=" The folder to save models.")
    conf.add_argument('--epochs', type=int, default=200,
                      help='The training epoches.')
    conf.add_argument('--decay_epoch', type=int, default=100,
                      help='Step for lr.')
    conf.add_argument('--loadSize', type=int, default=143,
                      help='The load image size.')
    conf.add_argument('--fineSize', type=int, default=128,
                      help='The cropped image size.')
    conf.add_argument('--ngf', type=int, default=32,
                      help='The number of filter for generator.')
    conf.add_argument('--ndf', type=int, default=32,
                      help='The number of filter for discriminator.')
    conf.add_argument('--n_styles', type=int, default=4,
                      help='The number of styles.')
    conf.add_argument('--use_lsgan', type=bool, default=True,
                      help='whether to use lsgan loss.')
    conf.add_argument('--lambda_A', type=float, default=1.,
                      help='The weight for discriminator style classification loss.')
    conf.add_argument('--autoencoder_constrain', type=float, default=10.,
                      help='The weight for reconstruct loss.')
    conf.add_argument('--tv_strength', type=float, default=1e-6,
                      help='The weight for tv loss.')
    conf.add_argument('--print_freq', type=int, default=10,
                      help='The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=1,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=1,
                      help='batch size over all gpus.')
    conf.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for sgd.')
    conf.add_argument('--log_dir', type=str, default='log',
                      help='The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type=str,
                      help='The directory to save tensorboardx logs')
    conf.add_argument('--resume', '-r', action='store_true', default=False,
                      help='Resume from checkpoint or not.')
    conf.add_argument("--pretrain_model", type=str, default='',
                      help="where the checkpoint saved")

    args = conf.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        os.makedirs(os.path.join(args.log_dir, 'images'))
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer

    train(args)
