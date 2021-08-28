import os
import sys
from argparse import ArgumentParser
import inspect
import random

import torch

import pytorch_lightning as pl

from scipy.interpolate import interp1d

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

import utils.easings as easings
import interpolation.methods as methods
from utils.get_pretrained import get_pretrained_model_from_name

def create_diameter_list(style_dim: int, seed: int, diameters: list) -> np.array:
    if len(diameters) > 1:
        m = interp1d([0, 1], [diameters[0], diameters[1]])

        np.random.seed(seed)
        ret = np.random.randn(style_dim)
        for i in range(len(ret)):
            ret[i] = m(ret[i])

        return ret

    else:
        ret = np.zeros(style_dim)
        ret.fill(diameters[0])
        return ret



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
    script_parser.add_argument('--model_arch', help='The model architecture of the model to be loaded. (default: %(default)s)', type=str, default='alias-free-rosinality-v1')
    script_parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.75)
    script_parser.add_argument('--batch', default=8, help='Number of images to generate each batch. default: %(default)s)')
    script_parser.add_argument('--save_z_vectors', type=bool, help='Save the z vectors used to interpolate. default: %(default)s', default=True)
    script_parser.add_argument('--log_args', type=bool, help='Saves the arguments to a text file for later reference. default: %(default)s', default=True)

    default_method = 'interpolate'
    script_parser.add_argument('--method', type=str, help='Select a method for interpolation. Options: %s default: %s' % (get_function_options(methods), default_method), default=default_method)

    script_parser.add_argument('--path_to_z_vectors', type=str, help="Path to saved z vectors to load. For method: 'load_z_vectors'")

    script_parser.add_argument('--frames', type=int, help="Total number of frames to generate. For methods: 'interpolate', 'circular', 'simplex_noise'")

    script_parser.add_argument('--seeds', nargs='+', help="Add a seed value to a interpolation walk. First seed value will be used as the seed for a circular or noise walk. If none are provided random ones will be generated. For methods: 'interpolate', 'circular', 'simplex_noise'")

    default_easing = 'linear'
    script_parser.add_argument('--easing', type=str, help="How to ease between seeds. For method: 'interpolate' Options: %s default: %s" % (get_function_options(easings), default_easing), default=default_easing)

    script_parser.add_argument('--diameter', nargs='+', help="Defines the diameter of the circular or noise path. If two arguments are passed they will be used as a min and max a range for random diameters. For method: 'circular', 'simplex_noise'")

    args = parser.parse_args(args)

    if not os.path.isfile(args.load_model) or args.load_model.split('.')[-1] != 'pt':
        print('Invalid path %s is not a .pt model file.' % args.load_model)
        exit(1)

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
    if args.seeds is not None:
        for s in args.seeds:
            seeds.append(int(s))

    if args.method == 'interpolate' or args.method == 'circular' or args.method == 'simplex_noise':

        if args.frames is None:
            print('--frames is a required argument for method interpolate, circular, or simplex_noise!')
            exit(3)

    if args.method == 'circular' or args.method == 'simplex_noise':

        if args.diameter is None:
            print('--diameter is a required argument for method circular or simplex_noise!')
            exit(3)

    z_vectors = None
    if args.method == 'load_z_vectors':

        if args.path_to_z_vectors is None:
            print('--path_to_z_vectors is a required argument for method load_z_vectors!')
            exit(3)

        z_vectors = methods.load_z_vectors(args.path_to_z_vectors)

    elif args.method == 'interpolate':

        s = seeds
        while len(s) < 2:
            s.append(random.randint(0, 100000))

        easing_func = getattr(easings, args.easing)

        z_vectors = methods.interpolate(model.generator.style_dim, s, args.frames, easing_func)

    elif args.method == 'circular':

        s = seeds
        while len(s) < 1:
            s.append(random.randint(0, 100000))

        diameter_list = create_diameter_list(model.generator.style_dim, s, args.diameter)

        z_vectors = methods.circular(model.generator.style_dim, s[0], args.frames, diameter_list)

    elif args.method == 'simplex_noise':

        s = seeds
        while len(s) < 1:
            s.append(random.randint(0, 100000))

        diameter_list = create_diameter_list(model.generator.style_dim, s, args.diameter)

        z_vectors = methods.simplex_noise(model.generator.style_dim, s[0], args.frames, diameter_list)

    else:
        print('%s method not recognized!' % args.method)
        exit(2)

    os.makedirs(args.outdir, exist_ok=True)

    if args.log_args:
        args_log_save_path = os.path.join(args.outdir, 'args_log.txt')
        print('Saving log of arguments to %s' % args_log_save_path)
        f = open(args_log_save_path, 'w')
        print(args, file = f)
        f.close()

    if args.save_z_vectors and args.method != 'load_z_vectors':
        z_vector_save_path = os.path.join(args.outdir, 'z_vectors.pt')
        print('Saving z_vectors to %s' % z_vector_save_path)
        torch.save(
            {
                "z_vectors": z_vectors,
            },
            z_vector_save_path
        )
    elif args.save_z_vectors and args.method == 'load_z_vectors':
        print('Skipping save z_vectors because load_z_vectors is the selected method.')

    model.generate_from_vectors(z_vectors, args.outdir, args.trunc)

if __name__ == "__main__":
    cli_main()