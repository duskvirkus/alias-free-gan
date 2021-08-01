import os
import sys
from argparse import ArgumentParser
import inspect
import random

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

import utils.easings as easings
import interpolation.methods as methods


def get_function_options(a):
    functions = inspect.getmembers(a, inspect.isfunction)
    func_options = '['
    for i in range(len(functions)):
        func_options += "'" + functions[i][0] + "'"
        if i < len(functions) - 1:
            func_options += ', '
    func_options += ']'
    return func_options


def cli_main(args=None):

    print('Using Alias-Free GAN version: %s' % __version__)

    parser = ArgumentParser()
    parser = AliasFreeGAN.add_generate_specific_args(parser)

    script_parser = parser.add_argument_group("Generate Script")
    script_parser.add_argument("--load_model", help='Load a model checkpoint to use for generating content.', type=str, required=True)
    script_parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True)
    script_parser.add_argument('--frames', type=int, required=True, help='Total number of frames to generate.')
    script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
    script_parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.75)
    script_parser.add_argument('--batch', default=8, help='Number of images to generate each batch. default: %(default)s)')

    default_method = 'interpolate'
    script_parser.add_argument('--method', type=str, help='Select a method for interpolation. Options: %s default: %s' % (get_function_options(methods), default_method), default=default_method)

    script_parser.add_argument('--seeds', nargs='+', help="Add a seed value to a interpolation walk. First seed value will be used as the seed for a circular or noise walk. If none are provided random ones will be generated. For methods: 'interpolate', 'circular', 'noise'")

    default_easing = 'linear'
    script_parser.add_argument('--easing', type=str, help="How to ease between seeds. For method: 'interpolate' Options: %s default: %s" % (get_function_options(easings), default_easing), default=default_easing)

    script_parser.add_argument('--diameter', type=float, help="Defines the diameter of the circular path. For method: 'circular' default: %(default)s", default=500.0)

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
    if args.seeds is not None:
        for s in args.seeds:
            seeds.append(int(s))

    z_vectors = None
    if args.method == 'interpolate':

        s = seeds
        while len(s) < 2:
            s.append(random.randint(0, 100000))

        easing_func = getattr(easings, args.easing)

        z_vectors = methods.interpolate(model.generator.style_dim, s, args.frames, easing_func)

    elif args.method == 'circular':

        s = seeds
        while len(s) < 1:
            s.append(random.randint(0, 100000))

        z_vectors = methods.circular(s[0], args.frames, args.diameter)

    elif args.method == 'simplex_noise':

        s = seeds
        while len(s) < 1:
            s.append(random.randint(0, 100000))

        z_vectors = methods.simplex_noise(s[0], args.frames)

    else:
        print('%s method not recognized!' % args.method)
        exit(2)

    model.generate_from_vectors(z_vectors, args.outdir, args.trunc)

if __name__ == "__main__":
    cli_main()