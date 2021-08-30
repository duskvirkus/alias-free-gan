import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

from utils.get_pretrained import get_pretrained_model_from_name

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

    trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
    model = AliasFreeGAN(args.model_arch, args.load_model, args.outdir, None, **vars(args))
    trainer.fit(model, get_fake_dataloader(args.size))

    custom_checkpoint = args.load_model.endswith('.pt')

    if custom_checkpoint:
        print(f'Loading Custom Model from: {args.load_model}')
        model.load_checkpoint(args.load_model)
    else:
        print(f'Attempting to load pretrained model...')
        pretrained = get_pretrained_model_from_name(args.load_model)

        if pretrained.model_size != args.size:
            raise Exception(f'{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {args.size} that was specified in arguments.')

        if args.model_arch != pretrained.model_architecture:
            raise Exception(f'Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {args.model_arch}.')

        print(f'Loading pretrained model from: {pretrained.model_path}')
        model.load_checkpoint(pretrained.model_path)

        print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

    seeds = []
    for i in range(args.seed_start, args.seed_stop, 1):
        seeds.append(i)

    model.generate_images(seeds, args.outdir, args.trunc)

if __name__ == "__main__":
    cli_main()