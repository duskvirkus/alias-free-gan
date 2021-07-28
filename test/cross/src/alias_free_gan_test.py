import sys
import os
from argparse import ArgumentParser
import copy

import pytest

import torch

from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from src.alias_free_gan import AliasFreeGAN
from src.stylegan2.dataset import MultiResolutionDataset

# @pytest.fixture
# def alias_free(size):
#     parser = ArgumentParser()
#     parser = AliasFreeGAN.add_model_specific_args(parser)
#     args = parser.parse_args(['--size', '256'])
#     return AliasFreeGAN(**vars(args))


def tensors_equal(tensor1, tensor2):
    assert(isinstance(tensor1, torch.Tensor))
    assert(isinstance(tensor2, torch.Tensor))

    i = tensor1
    j = tensor2

    while True:
        try:
            i = torch.sum(i)
            j = torch.sum(j)
            val = (i == j)
            return val
        except RuntimeError:
            if not (err == 'Boolean value of Tensor with more than one value is ambiguous'):
                throw(err)



def assert_ordered_dict_equal(dict1, dict2):
    for i, j in zip(dict1.items(), dict2.items()):
        if isinstance(i, tuple) and isinstance(j, tuple):
            i = list(i)
            j = list(j)

            for k, m in zip(i, j):
                if isinstance(k, torch.Tensor) and isinstance(m, torch.Tensor):
                    assert(tensors_equal(k, m))
                elif isinstance(k, torch.Tensor) or isinstance(m, torch.Tensor):
                    assert False # Expected both i and j to be torch.Tensor but only one was
                else:
                    assert k == m

        elif isinstance(i, torch.Tensor) and isinstance(j, torch.Tensor):
            assert(tensors_equal(i, j))
        elif isinstance(i, torch.Tensor) or isinstance(j, torch.Tensor):
            assert False # Expected both i and j to be torch.Tensor but only one was
        else:
            assert i == j


def test_default_arguments():
    parser = ArgumentParser()
    parser = AliasFreeGAN.add_model_specific_args(parser)
    args = parser.parse_args(['--size', '256'])

    assert args.ada_every == 256
    assert args.ada_length == 500000
    assert args.ada_target == 0.6
    assert args.argument_p == 0.0
    assert args.augment == False
    assert args.batch == 16
    assert args.d_reg_every == 16
    assert args.lr_d == 0.002
    assert args.lr_g == 0.002
    assert args.n_samples == 8
    assert args.r1 == 10.0
    assert args.size == 256

def test_save_checkpoint_and_load_checkpoint():
    parser = ArgumentParser()
    parser = AliasFreeGAN.add_model_specific_args(parser)
    args = parser.parse_args(['--size', '256'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset('./ci/flowers-test-dataset-32-256', transform=transform, resolution=args.size)
    train_loader = data.DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True)

    model = AliasFreeGAN('alias-free-rosinality-v1', '', '/dev/null', **vars(args))
    trainer = None
    if 'TPU_IP_ADDRESS' in os.environ:
        trainer = pl.Trainer(max_epochs=0, tpu_cores=8)
    else:
        trainer = pl.Trainer(max_epochs=0, gpus=1)
    trainer.fit(model, train_loader)

    # copy current state
    generator_copy = model.generator.state_dict()
    discriminator_copy = model.discriminator.state_dict()
    g_ema_copy = model.g_ema.state_dict()
    opt_0_copy = model.optimizers()[0].state_dict()
    opt_1_copy = model.optimizers()[1].state_dict()

    os.makedirs('./ci/temp', exist_ok=True)

    model.save_checkpoint('./ci/temp/test-checkpoint.pt')

    assert os.path.isfile('./ci/temp/test-checkpoint.pt')

    model.load_checkpoint('./ci/temp/test-checkpoint.pt')

    assert_ordered_dict_equal(model.generator.state_dict(), generator_copy)
    assert_ordered_dict_equal(model.discriminator.state_dict(), discriminator_copy)
    assert_ordered_dict_equal(model.g_ema.state_dict(), g_ema_copy)
    assert_ordered_dict_equal(model.optimizers()[0].state_dict(), opt_0_copy)
    assert_ordered_dict_equal(model.optimizers()[1].state_dict(), opt_1_copy)

