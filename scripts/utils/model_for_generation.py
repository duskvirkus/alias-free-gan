import sys
import os
from typing import Any

import pytorch_lightning as pl

from utils.get_pretrained import get_pretrained_model_from_name

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src import __version__
from src.alias_free_gan import AliasFreeGAN
from src.fake_dataloader import get_fake_dataloader

def get_model_for_generation(**kwargs: Any):

    trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
    model = AliasFreeGAN(kwargs['model_arch'], kwargs['load_model'], kwargs['outdir'], None, **kwargs)
    trainer.fit(model, get_fake_dataloader(kwargs['size']))

    custom_checkpoint = kwargs['load_model'].endswith('.pt')

    if custom_checkpoint:
        print(f"Loading Custom Model from: {kwargs['load_model']}")
        model.load_checkpoint(kwargs['load_model'])
    else:
        print(f'Attempting to load pretrained model...')
        pretrained = get_pretrained_model_from_name(kwargs['load_model'])

        if pretrained.model_size != kwargs['size']:
            raise Exception(f"{pretrained.model_name} size of {pretrained.model_size} is not the same as size of {kwargs['size']} that was specified in arguments.")

        if kwargs.model_arch != pretrained.model_architecture:
            raise Exception(f"Pretrained model_architecture of {pretrained.model_architecture} does not match --model_arch value of {kwargs['model_arch']}.")

        print(f'Loading pretrained model from: {pretrained.model_path}')
        model.load_checkpoint(pretrained.model_path)

        print(f'\n\n{pretrained.model_name} information:\n{pretrained.description}\n\n')

    return model