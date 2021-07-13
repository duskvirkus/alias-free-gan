import pytorch_lightning as pl
import torch
from torch import optim, Tensor
from argparse import ArgumentParser
from typing import Any
from model import Generator, filter_parameters
from stylegan2.model import Discriminator
from stylegan2.non_leaking import augment
from stylegan2.op import conv2d_gradfix
from torch.nn import functional as F

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

        self.batch = kwargs['batch']
        self.augment = kwargs['augment']

        self.lr_g = kwargs['lr_g']
        self.lr_d = kwargs['lr_d']
        self.d_reg_ratio = kwargs['d_reg_every'] / (kwargs['d_reg_every'] + 1)


        generator_args = {
            'style_dim':kwargs['size'],
            'n_mlp':2,
            'kernel_size':3,
            'n_taps':6,
            'filter_parameters':filter_parameters(
                n_layer=14,
                n_critical=2,
                sr_max=kwargs['size'],
                cutoff_0=2,
                cutoff_n=kwargs['size'] / 2,
                stopband_0=pow(2, 2.1),
                stopband_n=(kwargs['size'] / 2) * pow(2, 0.3),
                channel_max=512,
                channel_base=pow(2, 14)
            )
        }

        # self.generator = Generator(
        #     style_dim=kwargs['size'],
        #     n_mlp=2,
        #     kernel_size=3,
        #     n_taps=6,
        #     filter_parameters=filter_parameters(
        #         n_layer=14,
        #         n_critical=2,
        #         sr_max=kwargs['size'],
        #         cutoff_0=2,
        #         cutoff_n=kwargs['size'] / 2,
        #         stopband_0=pow(2, 2.1),
        #         stopband_n=(kwargs['size'] / 2) * pow(2, 0.3),
        #         channel_max=512,
        #         channel_base=pow(2, 14)
        #     ),
        #     **kwargs
        # )

        self.generator = Generator(
            **generator_args,
            **kwargs
        )

        self.g_ema = Generator(
            **generator_args,
            **kwargs
        )

        self.discriminator = Discriminator(
            size=kwargs['size'],
            channel_multiplier=2
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        fake_predict = self._get_fake_predict()
        
        # real_score = real_predict.mean()
        # fake_score = fake_predict.mean()
        # r1_loss = self._d_r1_loss(real_predict, real)

        # Train generator
        if optimizer_idx == 0:
            g_loss = self._g_nonsaturating_loss(fake_predict)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            real_predict = self._get_real_predict(real)
            d_loss = self._d_logistic_loss(real_predict, fake_predict)
            return d_loss

    def _get_real_predict(self, real: Tensor) -> Tensor:
        real_img_aug = self._get_real_img_aug(real)
        return self.discriminator(real_img_aug)

    def _get_fake_predict(self):
        fake_img = self._get_fake_img()
        return self.discriminator(fake_img)

    def _make_noise(self, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(self.batch, latent_dim)
        return torch.randn(n_noise, self.batch, latent_dim)

    def _d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def _get_fake_img(self):
        noise = self._make_noise(self.generator.get_style_dim(), 1)
        return self.generator(noise)

    def _get_real_img_aug(self, real: Tensor):
        if self.augment:
            real_img_aug, _ = augment(real, ada_aug_p)
            return real_img_aug
        return real
    
    def _d_r1_loss(self, real_predict, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_predict.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def _g_nonsaturating_loss(self, fake_predict):
        loss = F.softplus(-fake_predict).mean()

        return loss

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