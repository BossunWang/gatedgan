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

from models import *
from utils import *


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

    # models
    in_nc = 3
    out_nc = 3
    generator = Generator(in_nc, out_nc, args.n_styles, args.ngf).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    assert "generator" in checkpoint
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    mean_array = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std_array = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)

    # create label dir
    style_sources = sorted(glob.glob(os.path.join(args.style_dir, 'train', '*')))
    style_labels = []
    for style_label in style_sources:
        style_path = os.path.join(args.out_dir, style_label.split('/')[-1])
        style_labels.append(style_label.split('/')[-1])
        if not os.path.exists(style_path):
            os.makedirs(style_path)

    assert len(style_labels) == args.n_styles

    for file_path, dirs, file_names in os.walk(args.content_dir):
        for file_name in tqdm(sorted(file_names)):
            org_p = os.path.join(file_path, file_name)
            org_image = cv2.imread(org_p)
            if org_image is None:
                continue

            w = int(org_image.shape[0] / args.img_scale)
            h = int(org_image.shape[1] / args.img_scale)

            w = (w // 100) * 100
            h = (h // 100) * 100
            # print(w, h)

            content_img = cv2.resize(org_image, (h, w), interpolation=cv2.INTER_CUBIC)
            content_tensor = toTensor(content_img, mean_array, std_array).to(device)

            for style_index in range(args.n_styles):
                style_index = torch.tensor(style_index)
                style_OHE = F.one_hot(style_index, args.n_styles).unsqueeze(0).long().to(device)
                input_dict = {'content': content_tensor, 'style_label': style_OHE}
                transform_output = generator(input_dict)

                generate_img = transform_output.squeeze(0).cpu().data.numpy()
                generate_img = np.clip((generate_img.transpose(1, 2, 0) * std_array) + mean_array, 0., 1.)
                generate_img = 255 * generate_img
                generate_img = generate_img.astype(np.uint8)
                generate_img = cv2.cvtColor(generate_img, cv2.COLOR_BGR2RGB)

                output_name = os.path.join(args.out_dir, style_labels[style_index], file_name)
                cv2.imwrite(output_name, generate_img)


if __name__ == '__main__':
    # TRAIN OPTIONS FROM GATED GAN

    conf = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    conf.add_argument("--content_dir", type=str,
                      help="The content root folder of testing set.")
    conf.add_argument("--style_dir", type=str,
                      help="The style root folder of training set.")
    conf.add_argument("--out_dir", type=str, default='transfer_images',
                      help=" The folder to save models.")
    conf.add_argument('--ngf', type=int, default=32,
                      help='The number of filter for generator.')
    conf.add_argument('--ndf', type=int, default=32,
                      help='The number of filter for discriminator.')
    conf.add_argument('--img_scale', type=int, default=1)
    conf.add_argument('--n_styles', type=int, default=4,
                      help='The number of styles.')
    conf.add_argument('--model_path', type=str, default='',
                        help='where the checkpoint saved')
    args = conf.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    test(args)
