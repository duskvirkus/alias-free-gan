import sys
import os
from argparse import ArgumentParser

import pytest

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
    assert args.n_samples == 9
    assert args.r1 == 10.0
    assert args.size == 256

def test_save_checkpoint():
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

    alias_free = AliasFreeGAN(**vars(args))
    trainer = None
    if 'TPU_IP_ADDRESS' in os.environ:
        trainer = pl.Trainer(max_epochs=0, tpu_cores=8)
    else:
        trainer = pl.Trainer(max_epochs=0, gpus=1)
    trainer.fit(alias_free, train_loader)

    os.makedirs('./ci/temp', exist_ok=True)
    alias_free.save_checkpoint('./ci/temp/test-checkpoint.pt')

    assert os.path.isfile('./ci/temp/test-checkpoint.pt')
