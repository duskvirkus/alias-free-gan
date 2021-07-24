from argparse import ArgumentParser
from typing import Any
import os

import torch
from torch import optim, Tensor
from torch.nn import functional as F
from torchvision import utils

import pytorch_lightning as pl

from src.model import Generator, filter_parameters
from src.stylegan2.model import Discriminator
from src.stylegan2.non_leaking import augment

if 'USE_CPU_OP' in os.environ:
    from src.op import conv2d_gradfix
else:
    from src.stylegan2.op import conv2d_gradfix

class AliasFreeGAN(pl.LightningModule):

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.batch = kwargs['batch']
        self.augment = kwargs['augment']
        self.n_samples = kwargs['n_samples']
        self.size = kwargs['size']

        self.lr_g = kwargs['lr_g']
        self.lr_d = kwargs['lr_d']
        self.d_reg_ratio = kwargs['d_reg_every'] / (kwargs['d_reg_every'] + 1)


        generator_args = {
            'style_dim':self.size,
            'n_mlp':2,
            'kernel_size':3,
            'n_taps':6,
            'filter_parameters':filter_parameters(
                n_layer=14,
                n_critical=2,
                sr_max=self.size,
                cutoff_0=2,
                cutoff_n=self.size / 2,
                stopband_0=pow(2, 2.1),
                stopband_n=(self.size / 2) * pow(2, 0.3),
                channel_max=512,
                channel_base=pow(2, 14)
            ),
        }

        self.generator = Generator(
            **generator_args,
            **kwargs
        )

        self.g_ema = Generator(
            **generator_args,
            **kwargs
        )

        self.discriminator = Discriminator(
            size=self.size,
            channel_multiplier=2
        )

        
        self.sample_z = torch.randn(
            self.n_samples, self.size
        )

    def on_train_start(self):
        print(f'\nAlignFreeGAN device: %s\n' % self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch

        loss = None

        # Train generator
        if optimizer_idx == 0:
            AliasFreeGAN._requires_grad(self.generator, True)
            AliasFreeGAN._requires_grad(self.discriminator, False)
            self.generator.eval()
            fake_predict = self._get_fake_predict()
            g_loss = self._g_nonsaturating_loss(fake_predict)
            # self.generator.zero_grad()
            # return g_loss
            loss = g_loss
            accum = 0.5 ** (32 / (10 * 1000))
            self._accumulate(self.g_ema, self.generator, accum)

        # Train discriminator
        if optimizer_idx == 1:
            AliasFreeGAN._requires_grad(self.generator, False)
            AliasFreeGAN._requires_grad(self.discriminator, True)
            self.generator.eval()
            fake_predict = self._get_fake_predict()
            real_predict = self._get_real_predict(real)
            d_loss = self._d_logistic_loss(real_predict, fake_predict)
            # self.discriminator.zero_grad()
            # return d_loss
            loss = d_loss

        return loss

    def _get_real_predict(self, real: Tensor) -> Tensor:
        real_img_aug = self._get_real_img_aug(real)
        return self.discriminator(real_img_aug)

    def _get_fake_predict(self):
        fake_img = self._get_fake_img()
        return self.discriminator(fake_img)

    def _make_noise(self, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(self.batch, latent_dim, device=self.device)
        return torch.randn(n_noise, self.batch, latent_dim, device=self.device)

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

    @staticmethod
    def _requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def _accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data = par1[k].data.to(self.device)
            par2[k].data = par2[k].data.to(self.device)
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

        buf1 = dict(model1.named_buffers())
        buf2 = dict(model2.named_buffers())

        for k in buf1.keys():
            if "ema_var" not in k:
                continue

            buf1[k].data.mul_(0).add_(buf2[k].data, alpha=1)

    def configure_optimizers(self):
        g_optim = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0, 0.99))
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d * self.d_reg_ratio,
            betas=(0 ** self.d_reg_ratio, 0.99 ** self.d_reg_ratio),
        )
        return [g_optim, d_optim]

    def training_epoch_end(self, training_step_outputs):
        self.g_ema.eval()
        self.sample_z = self.sample_z.to(self.device)
        sample = self.g_ema(self.sample_z)
        utils.save_image(
            sample,
            f"sample/{str(self.current_epoch).zfill(6)}.png",
            nrow=int(self.n_samples ** 0.5),
            normalize=True,
            value_range=(-1, 1),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("AliasFreeGAN Model")
        parser.add_argument("--size", help='Pixel dimension of model. Must be 256, 512, or 1024. Required!', type=int, required=True)
        parser.add_argument("--batch", help='Batch size. Will be overridden if --auto_scale_batch_size is used. (default: %(default)s)', default=16, type=int) # TODO add support for --auto_scale_batch_size
        parser.add_argument("--n_samples", help='Number of samples to generate in training process. (default: %(default)s)', default=9, type=int)
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
