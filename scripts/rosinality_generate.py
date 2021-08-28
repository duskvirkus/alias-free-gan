# Based on Rosinality generate.py
# https://github.com/rosinality/alias-free-gan-pytorch/commit/9dfd1255823dab98608edb7d25e7f81cab05b6ce

import os
import sys
from argparse import ArgumentParser
import inspect
import random

import torch
from torchvision import utils

import pytorch_lightning as pl

import numpy as np

import cv2

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

import utils.easings as easings
import interpolation.methods as methods

def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    parser = ArgumentParser()
    parser = AliasFreeGAN.add_generate_specific_args(parser)

    script_parser = parser.add_argument_group("Rosinality Generate Script")

    script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
    script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
    script_parser.add_argument('--batch', default=8, help='Not implymented yet! Number of images to generate each batch. default: %(default)s)') #TODO currently does nothing

    script_parser.add_argument(
        "--n_img", type=int, default=16, help="number of images to be generated (default: %(default)s)"
    )
    script_parser.add_argument(
        "--n_row", type=int, default=4, help="number of samples per row (default: %(default)s)"
    )
    script_parser.add_argument(
        "--truncation", type=float, default=0.5, help="truncation ratio (default: %(default)s)"
    )
    script_parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation (default: %(default)s)",
    )
    script_parser.add_argument("--n_frame", help="(default: %(default)s)", type=int, default=120)
    script_parser.add_argument("--radius", help="(default: %(default)s)", type=float, default=30)
    script_parser.add_argument("--video", help="Export a video from frames. (default: %(default)s)", type=bool, default=True)
    script_parser.add_argument("--frames", help="Save frames in frames subfolder. (default: %(default)s)", type=bool, default=False)
    script_parser.add_argument(
        "ckpt", metavar="CKPT", type=str, help="path to the model checkpoint"
    )

    args = parser.parse_args(args)

    trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
    model = AliasFreeGAN(args.model_arch, args.ckpt, args.outdir, None, **vars(args))
    trainer.fit(model, get_fake_dataloader(args.size))

    custom_checkpoint = args.load_model.endswith('.pt')

    if custom_chepoint:
        print(f'Loading Custom Model from: {args.ckpt}')
        model.load_checkpoint(args.ckpt)
    else:
        print(f'Attempting to load pretrained model...')
        pretrained = get_pretrained_model_from_name(args.ckpt)

        if pretrained.model_size != args.size:
            raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

        if args.model_arch != pretrained.model_architecture:
            raise Exception(f'Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {args.model_arch}.')

        print(f'Loading pretrained model from: {pretrained.model_path}')
        model.load_checkpoint(pretrained.model_path)

        print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

    model.generator.eval()

    mean_latent = model.generator.mean_latent(args.truncation_mean)
    x = torch.randn(args.n_img, model.generator.style_dim, device=model.device)

    theta = np.radians(np.linspace(0, 360, args.n_frame))
    x_2 = np.cos(theta) * args.radius
    y_2 = np.sin(theta) * args.radius

    trans_x = x_2.tolist()
    trans_y = y_2.tolist()

    images = []

    transform_p = model.generator.get_transform(
        x, truncation=args.truncation, truncation_latent=mean_latent
    )

    with torch.no_grad():
      for i, (t_x, t_y) in enumerate(tqdm(zip(trans_x, trans_y), total=args.n_frame)):
          transform_p[:, 2] = t_y
          transform_p[:, 3] = t_x

          img = model.generator(
              x,
              truncation=args.truncation,
              truncation_latent=mean_latent,
              transform=transform_p,
          )
          images.append(
              utils.make_grid(
                  img.cpu(), normalize=True, nrow=args.n_row, value_range=(-1, 1)
              )
              .mul(255)
              .permute(1, 2, 0)
              .numpy()
              .astype(np.uint8)
          )
    videodims = (images[0].shape[1], images[0].shape[0])

    base_name = 'ros-gen-' + args.ckpt.split('/')[-1]

    os.makedirs(args.outdir, exist_ok=True)

    if args.frames:

        os.makedirs(os.path.join(args.outdir, 'frames'), exist_ok=True)

        for i in range(len(images)):
            im = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)

            new_file = base_name + '-' + str(i).zfill(9) + ".png"
            cv2.imwrite(os.path.join(args.outdir, 'frames', new_file), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if args.video:

        vid_name = base_name + '.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vid = cv2.VideoWriter(os.path.join(args.outdir, vid_name), fourcc, 24, videodims)

        for i in tqdm(images):
            vid.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

        vid.release()

if __name__ == "__main__":
    cli_main()