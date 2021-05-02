"""
@author: Bossun Wang
@date: 20210328
@contact: vvmodouco@gmail.com
"""

import torch
import torch.nn.functional as F

import glob
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.linalg import block_diag
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import *
from utils import *
from data import *


def get_block_diagonal_mask(n_group, n_member):
    G = n_group
    ones = np.ones((n_member, n_member)).tolist()
    mask = block_diag(ones, ones)
    for i in range(G - 2):
        mask = block_diag(mask, ones)
    return torch.from_numpy(mask).float()


def toTensor(img, mean, std):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).float()
    return image


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The number of blocks in the coloring matrix: G, The number of elements for each block: n_members^2
    n_mem = args.content_dim // args.n_group
    # This is used in generators to make the coloring matrix the block diagonal form.
    mask = get_block_diagonal_mask(args.n_group, n_mem).to(device)

    # models
    generator = Generator(args.n_styles, args.conv_dim, args.res_blocks_num, mask, args.n_group
                          , args.mlp_dim, args.bias_dim, args.content_dim, is_training=False, device=device).to(device)
    # generator = torch.nn.DataParallel(generator).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)

    assert "generatorB" in checkpoint
    generator.load_state_dict(checkpoint['generatorB'])

    generator.eval()

    mean_array = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std_array = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)

    #
    # train_dataset = ImageDataset(args.content_dir, args.style_dir, mode='train')
    # dataloader = DataLoader(train_dataset,
    #                         batch_size=1, shuffle=False, num_workers=4)
    #

    # for batch_idx, batch in tqdm(enumerate(dataloader)):
    #     # Unpack minibatch
    #     # source content
    #     real_content = batch['content'].to(device)
    #     # target style
    #     real_style = batch['style'].to(device)
    #     # style label
    #     style_label = batch['style_label'].to(device)
    #     # one-hot encoded style
    #     style_OHE = F.one_hot(style_label, args.n_styles).long().to(device)
    #     style_dict = {'style': real_style, 'style_label': style_OHE}
    #
    #     clamping_alpha(generator)
    #
    #     with torch.no_grad():
    #         transform_output, _, _, _, _ = generator(real_content, style_dict)
    #
    #     plt.imshow((transform_output[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     plt.show()
    #
    #     # generate_img = transform_output.squeeze(0).cpu().data.numpy()
    #     # generate_img = np.clip((generate_img.transpose(1, 2, 0) * std_array) + mean_array, 0., 1.)
    #     # generate_img = 255 * generate_img
    #     # generate_img = generate_img.astype(np.uint8)
    #     # generate_img = cv2.cvtColor(generate_img, cv2.COLOR_BGR2RGB)
    #     #
    #     # cv2.imshow("test", generate_img)
    #     # cv2.waitKey(0)

    # create label dir
    style_sources = sorted(glob.glob(os.path.join(args.style_dir, '*')))

    style_labels = []
    for style_label in style_sources:
        style_path = os.path.join(args.out_dir, style_label.split('/')[-1])
        style_labels.append(style_label.split('/')[-1])
        if not os.path.exists(style_path):
            os.makedirs(style_path)

    assert len(style_labels) == args.n_styles

    print(style_labels)

    for file_path, dirs, file_names in os.walk(args.content_dir):
        for file_name in tqdm(sorted(file_names)):
            org_p = os.path.join(file_path, file_name)
            org_image = cv2.imread(org_p)
            if org_image is None:
                continue

            # w = org_image.shape[0]
            # h = org_image.shape[1]

            # offest = h if w > h else w

            # crop_img = org_image[0:offest, 0:offest]

            w = int(org_image.shape[1] / args.img_scale)
            h = int(org_image.shape[0] / args.img_scale)

            w = (w // 100) * 100
            h = (h // 100) * 100
            # print(w, h)

            # content_img = cv2.resize(crop_img, (400, 400), interpolation=cv2.INTER_CUBIC)
            content_img = cv2.resize(org_image, (w, h), interpolation=cv2.INTER_CUBIC)
            content_tensor = toTensor(content_img, mean_array, std_array).to(device)

            style_dict_list = []

            for style_index in range(args.n_styles):
                style_path = os.path.join(args.style_dir, style_labels[style_index].split('/')[-1])
                style_files = os.listdir(style_path)

                style_image = cv2.imread(os.path.join(style_path, style_files[0]))
                style_image = cv2.resize(style_image, (400, 400), interpolation=cv2.INTER_CUBIC)

                style_tensor = toTensor(style_image, mean_array, std_array).to(device)
                style_index = torch.tensor(style_index)

                style_OHE = F.one_hot(style_index, args.n_styles).unsqueeze(0).float().to(device)
                style_dict = {'style': style_tensor, 'style_label': style_OHE}

                with torch.no_grad():
                    transform_output, _, _, _, _ = generator([content_tensor], [style_dict])

                generate_img = transform_output.squeeze(0).cpu().data.numpy()
                generate_img = np.clip((generate_img.transpose(1, 2, 0) * std_array) + mean_array, 0., 1.)
                generate_img = 255 * generate_img
                generate_img = generate_img.astype(np.uint8)
                generate_img = cv2.cvtColor(generate_img, cv2.COLOR_BGR2RGB)

                output_name = os.path.join(args.out_dir, style_labels[style_index], file_name)
                cv2.imwrite(output_name, generate_img)

                style_dict_list.append(style_dict)

            # mix style
            mix_count = 4
            mix_raate = 1. / mix_count
            content_tensor_list = []
            for i, style_dict in enumerate(style_dict_list):
                content_tensor_list.append(content_tensor)
                style_dict_list[i]['style_label'] *= mix_raate

            # mix first two style
            with torch.no_grad():
                transform_output, _, _, _, _ = generator(content_tensor_list[:mix_count], style_dict_list[:mix_count])

            generate_img = transform_output.squeeze(0).cpu().data.numpy()
            generate_img = np.clip((generate_img.transpose(1, 2, 0) * std_array) + mean_array, 0., 1.)
            generate_img = 255 * generate_img
            generate_img = generate_img.astype(np.uint8)
            generate_img = cv2.cvtColor(generate_img, cv2.COLOR_BGR2RGB)

            mixed_file_name = ''
            for style_label in style_labels:
                mixed_file_name += style_label + '_'

            mixed_file_name += file_name
            output_mixed_name = os.path.join(args.out_dir, mixed_file_name)
            cv2.imwrite(output_mixed_name, generate_img)


if __name__ == '__main__':
    # TRAIN OPTIONS FROM GATED GAN

    conf = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    conf.add_argument("--content_dir", type=str,
                      help="The content root folder of testing set.")
    conf.add_argument("--style_dir", type=str,
                      help="The style root folder of training set.")
    conf.add_argument("--out_dir", type=str, default='transfer_images',
                      help=" The folder to save models.")
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
    conf.add_argument('--img_scale', type=int, default=1)
    conf.add_argument('--n_styles', type=int, default=4,
                      help='The number of styles.')
    conf.add_argument('--model_path', type=str, default='',
                        help='where the checkpoint saved')
    args = conf.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    test(args)
