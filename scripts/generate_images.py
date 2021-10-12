import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN

from utils.model_for_generation import get_model_for_generation
from utils.get_image_producer import add_image_producer_specific_args, get_image_producer

def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    parser = ArgumentParser()

    script_parser = parser.add_argument_group("Generate Script")
    script_parser.add_argument("--load_model", help='Load a model checkpoint to use for generating content.', type=str, required=True)
    script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
    script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
    script_parser.add_argument('--seed_start', type=int, help='Start range for seed values. (default: %(default)s)', default=0)
    script_parser.add_argument('--seed_stop', type=int, help='Stop range for seed values. Is inclusive. (default: %(default)s)', default=99)
    script_parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.75)
    script_parser.add_argument('--batch', default=8, help='Number of images to generate each batch. default: %(default)s)')
    parser = AliasFreeGAN.add_generate_specific_args(parser)
    parser = add_image_producer_specific_args(parser)

    args = parser.parse_args(args)

    model = get_model_for_generation(**vars(args))

    image_producer = get_image_producer(model, **vars(args))

    latents = []
    for i in range(args.seed_start, args.seed_stop, 1):
        image_producer.generate(np.random.RandomState(i).randn(1, model.generator.style_dim), args.trunc, args.outdir, 'seed-' + str(i).zfill(6))

if __name__ == "__main__":
    cli_main()