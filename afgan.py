import pytorch_lightning as pl
from torch import optim
from argparse import ArgumentParser
from typing import Any
from model import Generator, filter_parameters
from stylegan2.model import Discriminator

class AFGAN(pl.LightningModule):

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()

        # kwargs['style_dim'] = kwargs['size']
        # kwargs['n_mlp'] = 2
        # kwargs['kernel_size'] = 3
        # kwargs['n_taps']: 6
        # kwargs['filter_parameters'] = {
        #     '__target': 'model.filter_parameters',
        #     'n_layer': 14,
        #     'n_critical': 2,
        #     'sr_max': kwargs['size'],
        #     'cutoff_0': 2,
        #     'cutoff_n': kwargs['size'] / 2,
        #     'stopband_0': pow(2, 2.1),
        #     'stopband_n': (kwargs['size'] / 2) * pow(2, 0.3),
        #     'channel_max': 512,
        #     'channel_base': pow(2, 14)
        # }

        #         style_dim: 512,
        # n_mlp: 2,
        # kernel_size: 3,
        # n_taps: 6,
        # filter_parameters: {
        #     __target: 'model.filter_parameters',
        #     n_layer: 14,
        #     n_critical: 2,
        #     sr_max: $.training.size,
        #     cutoff_0: 2,
        #     cutoff_n: self.sr_max / 2,
        #     stopband_0: std.pow(2, 2.1),
        #     stopband_n: self.cutoff_n * std.pow(2, 0.3),
        #     channel_max: 512,
        #     channel_base: std.pow(2, 14)
        # },
        # margin: 10,
        # lr_mlp: 0.01

        self.lr_g = kwargs['lr_g']
        self.lr_d = kwargs['lr_d']
        self.d_reg_ratio = kwargs['d_reg_every'] / (kwargs['d_reg_every'] + 1)

        self.generator = Generator(
            style_dim=kwargs['size'],
            n_mlp=2,
            kernel_size=3,
            n_taps=6,
            filter_parameters=filter_parameters(
                n_layer=14,
                n_critical=2,
                sr_max=kwargs['size'],
                cutoff_0=2,
                cutoff_n=kwargs['size'] / 2,
                stopband_0=pow(2, 2.1),
                stopband_n=(kwargs['size'] / 2) * pow(2, 0.3),
                channel_max=512,
                channel_base=pow(2, 14)
            ),
            **kwargs
        )

        self.discriminator = Discriminator(
            size=kwargs['size'],
            channel_multiplier=2
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def configure_optimizers(self):
        g_optim = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0, 0.99))
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d * self.d_reg_ratio,
            betas=(0 ** self.d_reg_ratio, 0.99 ** self.d_reg_ratio),
        )
        return [g_optim, d_optim]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("AFGAN Model")
        parser.add_argument("--lr_g", help='Generator learning rate. (default: %(default)s)', default=2e-3, type=float)
        parser.add_argument("--lr_d", help='Discriminator learning rate. (default: %(default)s)', default=2e-3, type=float)
        parser.add_argument("--d_reg_every", help='Regularize discriminator ever _ iters. (default: %(default)s)', default=16, type=int)
        parser.add_argument("--r1", help='R1 regularization weights. (default: %(default)s)', default=10., type=float)
        parser.add_argument("--augment", help='Use augmentations. (default: %(default)s)', default=False, type=bool)
        parser.add_argument("--argument_p", help='(default: %(default)s)', default=0., type=float)
        parser.add_argument("--ada_target", help='(default: %(default)s)', default=0.6, type=float)
        parser.add_argument("--ada_length", help='(default: %(default)s)', default=(500 * 1000), type=int)
        parser.add_argument("--ada_every", help='(default: %(default)s)', default=256, type=int)
        return parent_parser