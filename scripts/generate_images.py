import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

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

    args = parser.parse_args(args)

    if not os.path.isfile(args.load_model) or args.load_model.split('.')[-1] != 'pt':
        print('Invalid path %s is not a .pt model file.' % args.load_model)
        exit(1)

    trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
    model = AliasFreeGAN(args.model_arch, args.load_model, args.outdir, **vars(args))
    trainer.fit(model, get_fake_dataloader(args.size))

    print(f'Loading Model from: %s\n' % args.load_model)
    model.load_checkpoint(args.load_model)

    seeds = []
    for i in range(args.seed_start, args.seed_stop, 1):
        seeds.append(i)

    model.generate_images(seeds, args.outdir, args.trunc)

if __name__ == "__main__":
    cli_main()