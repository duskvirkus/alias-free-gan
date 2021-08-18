import os
from argparse import ArgumentParser
from typing import Any

import torch

import pytorch_lightning as pl

def bool_printout_count(num):
    if num == False:
        return 'F'
    return 'T'

class KimgSaverCallback(pl.Callback):
    """
    Saves checkpoints on intervals of kimgs instead of pytorch-lighting epochs.
    """

    def __init__(
        self,
        save_dir: str,
        kimg_start_from_resume = None,
        model_name: str = 'AliasFreeGAN_model',
        **kwargs: Any,
    ):
        self.save_sample_frequency = kwargs['save_sample_every_kimgs']
        self.save_checkpoint_frequency = kwargs['save_checkpoint_every_kimgs']

        if kwargs['start_kimg_count']:
            self.kimg_start = kwargs['start_kimg_count']
        elif kimg_start_from_resume is not None:
            self.kimg_start = kimg_start_from_resume
        else:
            self.kimg_start = 0
        
        self.stop_training_at = kwargs['stop_training_at_kimgs']

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_name = model_name

        sample_grid_data = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/sample_grids', kwargs['sample_grid'] + '.pt'))
        self.sample_grid_vectors = sample_grid_data['sample_grid_vectors']
        self.sample_grid_rows = sample_grid_data['sample_grid_rows']
        self.sample_grid_cols = sample_grid_data['sample_grid_cols']

        self.img_count = self.kimg_start * 1000

    def update_progress_items(self, items):
        items['kimgs'] = '{:.3f}'.format(self.img_count / 1000)
        return items

    def on_batch_end(self, trainer, model):

        global_step = trainer.global_step
        self.img_count = (self.kimg_start * 1000) + (trainer.global_step + 1) * model.batch

        save_sample_num = False
        save_checkpoint_num = False
        for i in range(model.batch):
            if (self.img_count - i) % (self.save_sample_frequency * 1000) == 0:
                save_sample_num = (self.img_count - i) / 1000
            if (self.img_count - i) % (self.save_checkpoint_frequency * 1000) == 0:
                save_checkpoint_num = (self.img_count - i) / 1000

        if save_sample_num:
            self.save_sample(model, str(int(save_sample_num)).zfill(9) + '-kimg-' + self.model_name + '-samples.jpg')

        if save_checkpoint_num:
            self.save_checkpoint(model, str(int(save_checkpoint_num)).zfill(9) + '-kimg-' + self.model_name + '-checkpoint.pt')

        if self.img_count >= self.stop_training_at * 1000:
            print('Training stopping because image count reached stop training at count.')
            exit(0)

    def save_checkpoint(self, model, checkpoint_name):
        model.save_checkpoint(os.path.join(self.save_dir, checkpoint_name))

    def save_sample(self, model, sample_name):
        model.save_samples(os.path.join(self.save_dir, sample_name), self.sample_grid_rows, self.sample_grid_cols, self.sample_grid_vectors)

    @staticmethod
    def add_callback_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("kimg Saver Callback")
        parser.add_argument("--save_sample_every_kimgs", help='Sets the frequency of saving samples in kimgs (thousands of image). (default: %(default)s)', type=int, default=1)
        parser.add_argument("--save_checkpoint_every_kimgs", help='Sets the frequency of saving model checkpoints in kimgs (thousands of image). (default: %(default)s)', type=int, default=4)
        parser.add_argument('--start_kimg_count', help='Manually override the start count for kimgs. If not set the count will be inferred from checkpoint name. If count can not be inferred it will default to 0.', type=int)
        parser.add_argument('--stop_training_at_kimgs', help='Automatically stop training at this number of kimgs. (default: %(default)s)', type=int, default=12800)
        parser.add_argument('--sample_grid', help='Sample grid to use for samples. Saved under assets/sample_grids. (default: %(default)s)', type=str, default='default_5x3_sample_grid')
        return parent_parser
