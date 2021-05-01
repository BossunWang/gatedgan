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
from scipy.linalg import block_diag
import numpy as np

from data import *
from models import *
from utils import *


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_block_diagonal_mask(n_group, n_member):
    G = n_group
    ones = np.ones((n_member, n_member)).tolist()
    mask = block_diag(ones, ones)
    for i in range(G - 2):
        mask = block_diag(mask, ones)
    return torch.from_numpy(mask).float()


def l1_criterion(input, target):
    return torch.mean(torch.abs(input - target))


def reg(x_arr, device):
    # whitening_reg: G,C//G,C//G
    I = torch.eye(x_arr[0][0].size(1)).unsqueeze(0).to(device)  # 1,C//G,C//G
    loss = torch.FloatTensor([0]).to(device)
    for x in x_arr:
        x = torch.cat(x, dim=0)  # G*(# of style),C//G,C//G
        loss = loss + torch.mean(torch.abs(x - I))
    return loss / len(x_arr)

def l1_criterion(input, target):
    return torch.mean(torch.abs(input - target))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_dataset = ImageDataset(args.content_dir, args.style_dir, mode='train')
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size, shuffle=True, num_workers=4)

    # The number of blocks in the coloring matrix: G, The number of elements for each block: n_members^2
    n_mem = args.content_dim // args.n_group
    # This is used in generators to make the coloring matrix the block diagonal form.
    mask = get_block_diagonal_mask(args.n_group, n_mem).to(device)
    # models
    generatorA = Generator(1, args.conv_dim, args.res_blocks_num, mask, args.n_group
                           , args.mlp_dim, args.bias_dim, args.content_dim, device)
    generatorB = Generator(args.n_styles, args.conv_dim, args.res_blocks_num, mask, args.n_group
                           , args.mlp_dim, args.bias_dim, args.content_dim, device)

    discriminatorA = Discriminator()
    discriminatorB = Discriminator()

    # Optimizers
    G_params = list(generatorA.parameters()) + list(generatorB.parameters())
    optimizer_G = torch.optim.Adam([p for p in G_params if p.requires_grad],
                                   lr=args.lr, betas=(0.5, 0.999))
    D_params = list(discriminatorA.parameters()) + list(discriminatorB.parameters())
    optimizer_D = torch.optim.Adam([p for p in D_params if p.requires_grad],
                                   lr=args.lr, betas=(0.5, 0.999))

    generatorA.apply(weights_init('kaiming'))
    generatorB.apply(weights_init('kaiming'))
    discriminatorA.apply(weights_init('gaussian'))
    discriminatorB.apply(weights_init('gaussian'))

    # init losses
    criterion_TV = TVLoss(TVLoss_weight=args.tv_strength).to(device)

    ori_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.pretrain_model, map_location=device)
        assert "epoch" in checkpoint \
               and "generatorA" in checkpoint \
               and "generatorB" in checkpoint \
               and "discriminatorA" in checkpoint \
               and "discriminatorB" in checkpoint \
               and "optimizer_G" in checkpoint \
               and "optimizer_D" in checkpoint
        ori_epoch = checkpoint['epoch'] + 1
        generatorA.load_state_dict(checkpoint['generatorA'])
        generatorB.load_state_dict(checkpoint['generatorB'])
        discriminatorA.load_state_dict(checkpoint['discriminatorA'])
        discriminatorB.load_state_dict(checkpoint['discriminatorB'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    # LR schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(args.epochs
                                                                          , ori_epoch
                                                                          , args.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(args.epochs
                                                                          , ori_epoch
                                                                          , args.decay_epoch).step)

    # set DataParallel for models
    generatorA = torch.nn.DataParallel(generatorA).to(device)
    generatorB = torch.nn.DataParallel(generatorB).to(device)
    discriminatorA = torch.nn.DataParallel(discriminatorA).to(device)
    discriminatorB = torch.nn.DataParallel(discriminatorB).to(device)

    autoflag_OHE = torch.ones(1, 1).long().to(device)

    generatorA.train()
    generatorB.train()
    discriminatorA.train()
    discriminatorB.train()

    generator_loss_meter = AverageMeter()
    loss_g_gan_meter = AverageMeter()
    reconstruction_loss_meter = AverageMeter()
    loss_cyc_meter = AverageMeter()
    loss_feature_s_meter = AverageMeter()
    loss_feature_c_meter = AverageMeter()
    loss_whitening_reg_meter = AverageMeter()
    loss_coloring_reg_meter = AverageMeter()
    discriminator_gan_loss_meter = AverageMeter()

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
            style_dict = {'style': real_style, 'style_label': style_OHE}
            identity_dict = {'style': real_content, 'style_label': autoflag_OHE}

            # Update Discriminator
            # Generate style-transferred image
            x_BA, _, _, _, _ = generatorA(real_style, identity_dict)
            x_AB, _, _, _, _ = generatorB(real_content, style_dict)

            # D loss
            d_loss_a = discriminatorA.module.calc_dis_loss(x_BA.detach(), real_content)
            d_loss_b = discriminatorB.module.calc_dis_loss(x_AB.detach(), real_style)

            d_loss = d_loss_a + d_loss_b

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Update Generator

            # Generate style-transferred image
            # 1st stage
            x_BA, whitening_reg_BA, coloring_reg_BA, c_B, s_A = generatorA(real_style, identity_dict)
            x_AB, whitening_reg_AB, coloring_reg_AB, c_A, s_B = generatorB(real_content, style_dict)

            # 2st stage
            style_dict2 = {'style': x_AB, 'style_label': style_OHE}
            identity_dict2 = {'style': x_BA, 'style_label': autoflag_OHE}
            # from AB to A
            x_ABA, whitening_reg_ABA, coloring_reg_ABA, c_BA, s_BA = generatorA(x_BA, identity_dict2)
            # from BA to B
            x_BAB, whitening_reg_BAB, coloring_reg_BAB, c_AB, s_AB = generatorB(x_AB, style_dict2)
            # from A to A
            x_AA, _, _, _, _ = generatorA(real_content, identity_dict)
            # from B to B
            x_BB, _, _, _, _ = generatorB(real_style, style_dict)

            # G losses
            g_loss_fake = discriminatorA.module.calc_gen_loss(x_BA) + discriminatorB.module.calc_gen_loss(x_AB)
            loss_cross_rec = l1_criterion(x_ABA, real_content) + l1_criterion(x_BAB, real_style)
            loss_ae_rec = l1_criterion(x_AA, real_content) + l1_criterion(x_BB, real_style)
            loss_cross_s = l1_criterion(s_AB, s_B) + l1_criterion(s_BA, s_A)
            loss_cross_c = l1_criterion(c_AB, c_A) + l1_criterion(c_BA, c_B)

            loss_whitening_reg = reg([whitening_reg_AB, whitening_reg_BA, whitening_reg_ABA, whitening_reg_BAB], device)
            loss_coloring_reg = reg([coloring_reg_AB, coloring_reg_BA, coloring_reg_ABA, coloring_reg_BAB], device)

            g_loss = g_loss_fake \
                     + args.lambda_x_rec * loss_ae_rec \
                     + args.lambda_x_cyc * loss_cross_rec \
                     + args.lambda_s * loss_cross_s \
                     + args.lambda_c * loss_cross_c \
                     + args.lambda_w_reg * loss_whitening_reg \
                     + args.lambda_c_reg * loss_coloring_reg

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            generator_loss_meter.update(g_loss.item(), args.batch_size)

            loss_g_gan_meter.update(g_loss_fake.item(), args.batch_size)
            reconstruction_loss_meter.update(loss_ae_rec.item(), args.batch_size)
            loss_cyc_meter.update(loss_cross_rec.item(), args.batch_size)

            loss_feature_s_meter.update(loss_cross_s.item(), args.batch_size)
            loss_feature_c_meter.update(loss_cross_c.item(), args.batch_size)

            loss_whitening_reg_meter.update(loss_whitening_reg.item(), args.batch_size)
            loss_coloring_reg_meter.update(loss_coloring_reg.item(), args.batch_size)

            discriminator_gan_loss_meter.update(d_loss.item(), args.batch_size)

            if batch_idx % args.print_freq == 0:
                generator_loss_val = generator_loss_meter.avg

                loss_g_gan_val = loss_g_gan_meter.avg
                reconstruction_loss_val = reconstruction_loss_meter.avg
                loss_cyc_val = loss_cyc_meter.avg

                loss_feature_s_val = loss_feature_s_meter.avg
                loss_feature_c_val = loss_feature_c_meter.avg

                loss_whitening_reg_val = loss_whitening_reg_meter.avg
                loss_coloring_reg_val = loss_coloring_reg_meter.avg

                discriminator_gan_loss_val = discriminator_gan_loss_meter.avg

                lr_G = get_lr(optimizer_G)
                lr_D = get_lr(optimizer_D)

                print('Epoch %d, iter %d / %d, lr G %f, lr D %f'
                      ', Generator Loss %f'
                      ', loss_G_GAN %f'
                      ', Reconstruction Loss %f'
                      ', loss_cyc_val %f'
                      ', loss_feature_s_val %f'
                      ', loss_feature_c_val %f'
                      ', whitening_reg %f'
                      ', coloring_reg %f'
                      ', Discriminator GAN Loss %f' %
                      (epoch, batch_idx, len(dataloader), lr_G, lr_D
                       , generator_loss_val
                       , loss_g_gan_val
                       , reconstruction_loss_val
                       , loss_cyc_val
                       , loss_feature_s_val
                       , loss_feature_c_val
                       , loss_whitening_reg_val
                       , loss_coloring_reg_val
                       , discriminator_gan_loss_val))

                plt.imsave(os.path.join(args.log_dir, 'images', 'content_%d.png' % epoch)
                           , (real_content[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'style_%d.png' % epoch)
                           , (real_style[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'transfer_%d.png' % epoch)
                           , (x_AB[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(os.path.join(args.log_dir, 'images', 'auto-reconstruction_%d.png' % epoch)
                           , (x_AA[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)

                # record to tensorboard
                args.writer.add_scalar('lr_G', lr_G, global_batch_idx)
                args.writer.add_scalar('lr_D', lr_G, global_batch_idx)
                args.writer.add_scalar('Generator Loss', generator_loss_val, global_batch_idx)
                args.writer.add_scalar('loss_G_GAN', loss_g_gan_val, global_batch_idx)
                args.writer.add_scalar('Reconstruction Loss', reconstruction_loss_val, global_batch_idx)
                args.writer.add_scalar('loss_cyc', loss_cyc_val, global_batch_idx)
                args.writer.add_scalar('loss_feature_s', loss_feature_s_val, global_batch_idx)
                args.writer.add_scalar('loss_feature_c', loss_feature_c_val, global_batch_idx)
                args.writer.add_scalar('loss_whitening_reg', loss_whitening_reg_val, global_batch_idx)
                args.writer.add_scalar('loss_coloring_reg', loss_coloring_reg_val, global_batch_idx)
                args.writer.add_scalar('Discriminator GAN Loss', discriminator_gan_loss_val, global_batch_idx)

                generator_loss_meter.reset()
                loss_g_gan_meter.reset()
                reconstruction_loss_meter.reset()
                loss_cyc_meter.reset()
                loss_feature_s_meter.reset()
                loss_feature_c_meter.reset()
                loss_whitening_reg_meter.reset()
                loss_coloring_reg_meter.reset()
                discriminator_gan_loss_meter.reset()

        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            # Save model
            saved_name = 'GatedGAN_Epoch_%d.pt' % epoch
            state = {
                'generatorA': generatorA.module.state_dict()
                , 'generatorB': generatorB.module.state_dict()
                , 'discriminatorA': discriminatorA.module.state_dict()
                , 'discriminatorB': discriminatorB.module.state_dict()
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
    conf.add_argument('--epochs', type=int, default=20,
                      help='The training epoches.')
    conf.add_argument('--decay_epoch', type=int, default=10,
                      help='Step for lr.')
    conf.add_argument('--loadSize', type=int, default=143,
                      help='The load image size.')
    conf.add_argument('--fineSize', type=int, default=128,
                      help='The cropped image size.')
    conf.add_argument('--conv_dim', type=int, default=64,
                      help='The number of filter for encoder.')
    conf.add_argument('--content_dim', type=int, default=256,
                      help='The number of filter for discriminator.')
    conf.add_argument('--mlp_dim', type=int, default=256,
                      help='The number of layer for mlp_CT.')
    conf.add_argument('--bias_dim', type=int, default=256,
                      help='The number of layer for mlp_mu.')
    conf.add_argument('--res_blocks_num', type=int, default=8,
                      help='The number of residual block for generator.')
    conf.add_argument('--n_group', type=int, default=8,
                      help='The number of group for generator.')
    conf.add_argument('--n_styles', type=int, default=4,
                      help='The number of styles.')
    conf.add_argument('--lambda_x_rec', type=float, default=10.,
                      help='The weight for auto encoder.')
    conf.add_argument('--lambda_x_cyc', type=float, default=10.,
                      help='The weight for cycle consistence.')
    conf.add_argument('--lambda_s', type=float, default=1.,
                      help='The weight for content feature loss.')
    conf.add_argument('--lambda_c', type=float, default=1.,
                      help='The weight for style feature loss.')
    conf.add_argument('--lambda_w_reg', type=float, default=1e-3,
                      help='The weight for whitening regularization.')
    conf.add_argument('--lambda_c_reg', type=float, default=10.,
                      help='The weight for coloring regularization.')
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
    conf.add_argument("--vgg_model", type=str, default='',
                      help="where the vgg encoder saved")

    args = conf.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(os.path.join(args.log_dir, 'images')):
        os.makedirs(os.path.join(args.log_dir, 'images'))

    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer

    train(args)
