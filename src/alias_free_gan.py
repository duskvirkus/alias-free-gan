from argparse import ArgumentParser
from typing import Any
import sys
import os
import gc

import torch
from torch import optim, Tensor
from torch.nn import functional as F
from torchvision import utils

import numpy as np

import PIL.Image

import pytorch_lightning as pl

from src.model import Generator, filter_parameters
from src.stylegan2.model import Discriminator
from src.stylegan2.non_leaking import augment, AdaptiveAugment

from src.supported_arch import SUPPORTED_ARCHITECTURE

if 'USE_CPU_OP' in os.environ:
    from src.op import conv2d_gradfix
else:
    from src.stylegan2.op import conv2d_gradfix

from src.utils import print_gpu_memory_stats


class AliasFreeGAN(pl.LightningModule):

    def __init__(
        self,
        model_architecture,
        resume_path,
        results_dir,
        kimg_callback,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters()

        if model_architecture not in SUPPORTED_ARCHITECTURE:
            print('%s is not a supported model architecture for this version of Alias-Free-GAN! Please check the Alias-Free-GAN version number you are using and refer to documentation to insure you\'re model version is compatable.' % model_architecture)
            exit(2)

        self.model_architecture = model_architecture
        self.results_dir = results_dir
        self.kimg_callback = kimg_callback

        self.resume_path = resume_path
        self.stylegan2_discriminator = None
        if 'stylegan2_discriminator' in kwargs and kwargs['stylegan2_discriminator'] is not None:
            self.stylegan2_discriminator = kwargs['stylegan2_discriminator']

        self.batch = int(kwargs['batch'])
        self.size = kwargs['size']

        self.augment = None
        if 'augment' in kwargs:
            self.augment = kwargs['augment']

        self.lr_g = 2e-3
        if 'lr_g' in kwargs:
            self.lr_g = kwargs['lr_g']

        self.lr_d = 2e-3
        if 'lr_d' in kwargs:
            self.lr_d = kwargs['lr_d']

        self.ada_aug_p = 0.0
        if 'augment_p' in kwargs and kwargs['augment_p'] > 0:
            self.ada_aug_p = kwargs['augment_p']

        self.augment = False
        if 'augment' in kwargs:
            self.augment = kwargs['augment']

        self.use_ada_augment = self.augment and self.ada_aug_p == 0

        if self.use_ada_augment:
            self.ada_augment = AdaptiveAugment(kwargs['ada_target'], kwargs['ada_length'], kwargs['ada_every'], self.device)

        self.r_t_stat = None

        generator_args = {
            'style_dim':512,
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

    def on_train_start(self):
        print('\n')
        if self.resume_path is not None and self.resume_path != '':
            print(f'Resuming from: %s\n' % self.resume_path)
            self.load_checkpoint(self.resume_path)

        if self.stylegan2_discriminator is not None and self.self.stylegan2_discriminator != '':
            print('Loading StyleGAN2 discriminator from %s' % self.stylegan2_discriminator)
            self.load_stylegan2_discriminator(self.stylegan2_discriminator)


        print('AlignFreeGAN device: %s' % self.device)
        print('\n')

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
            if self.use_ada_augment:
                self.ada_aug_p = self.ada_augment.tune(real_predict)
                self.r_t_stat = self.ada_augment.r_t_stat
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
        if self.augment is not None and self.augment:
            real_img_aug, _ = augment(real, self.ada_aug_p)
            real_img_aug = real_img_aug.contiguous()
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
        d_optim = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0, 0.99))
        # d_optim = optim.Adam(
        #     self.discriminator.parameters(),
        #     lr=self.lr_d * self.d_reg_ratio,
        #     betas=(0 ** self.d_reg_ratio, 0.99 ** self.d_reg_ratio),
        # )
        return [g_optim, d_optim]

    def save_checkpoint(self, save_path):
        optimizers = self.optimizers()
        conf = self.hparams
        conf['kimg_callback'] = None
        torch.save(
            {
                "g": self.generator.state_dict(),
                "d": self.discriminator.state_dict(),
                "g_ema": self.g_ema.state_dict(),
                "g_optim": self.optimizers()[0].state_dict(),
                "d_optim": self.optimizers()[1].state_dict(),
                "conf": conf,
                "ada_aug_p": self.ada_aug_p,
            },
            save_path,
            # f"checkpoint/{str(i).zfill(6)}.pt",
        )

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)

        self.generator.load_state_dict(checkpoint["g"])
        self.discriminator.load_state_dict(checkpoint["d"])
        self.g_ema.load_state_dict(checkpoint["g_ema"])

        self.optimizers()[0].load_state_dict(checkpoint["g_optim"])
        self.optimizers()[1].load_state_dict(checkpoint["d_optim"])

        # TODO add support for loading hparams

    def load_stylegan2_discriminator(self, load_path):
        checkpoint = torch.load(load_path)
        self.discriminator.load_state_dict(checkpoint["d"])
        self.optimizers()[1].load_state_dict(checkpoint["d_optim"])

    def generate_images(
        self,
        seeds: list,
        save_dir: str,
        trunc,
    ):
        # TODO get batch generation working
        # remaining_seeds = seeds

        # while remaining_seeds is not None and len(remaining_seeds) > 0:
        #     current_seeds = None
        #     if len(remaining_seeds) > self.batch:
        #         current_seeds = remaining_seeds[:self.batch]
        #         remaining_seeds = remaining_seeds[self.batch:]
        #     else:
        #         current_seeds = remaining_seeds
        #         remaining_seeds = None
        #     print('Generating images for seeds: ', current_seeds)

        #     np_rand_array = []
        #     for seed in current_seeds:
        #         np_rand_array.append(np.random.RandomState(seed).randn(self.generator.style_dim))

        #     np_rand_stack = np.stack(np_rand_array)
        #     # print(np_rand_stack.shape)
        #     z = torch.from_numpy(np_rand_stack)
        #     z = z.float()
        #     print(z.shape)
        #     z = z.to(self.device)
        #     images = self.g_ema(z) #, trunc, self.generator.mean_latent(4096))
        #     print(images.shape)

        self.g_ema.eval()

        os.makedirs(save_dir, exist_ok=True)

        for seed in seeds:

            print('Generating image for seed %d at truncation %f and saving to %s' % (seed, trunc, f'{save_dir}/seed{seed:04d}.png'))

            img = self.g_ema(torch.from_numpy(np.random.RandomState(seed).randn(1, self.generator.style_dim)).float().to(self.device), trunc, self.generator.mean_latent(4096))

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{save_dir}/seed{seed:04d}.png')

    def save_samples(
        self,
        save_location: str,
        sample_grid_rows,
        sample_grid_cols,
        sample_grid_vectors,
        grid_cell_dim: int = 256,
    ):
        self.g_ema.eval()

        # results = self.generate_from_vectors(sample_grid_vectors, return_results=True, print_progress=False)

        grid = PIL.Image.new('RGB', (sample_grid_cols * grid_cell_dim, sample_grid_rows * grid_cell_dim))

        for i in range(len(sample_grid_vectors)):
            x_loc = int(i / sample_grid_rows) * grid_cell_dim
            y_loc = int(i % sample_grid_rows) * grid_cell_dim

            img = self.g_ema(torch.from_numpy(sample_grid_vectors[i]).float().to(self.device))
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            img = img.resize((grid_cell_dim, grid_cell_dim))

            grid.paste(img, (x_loc, y_loc))

        grid.save(save_location)

    def generate_from_vectors(
        self,
        z_vectors: np.array,
        save_dir: str,
        trunc: float,
        sub_dir: str = 'frames',
    ) -> None:
        self.g_ema.eval()
        actual_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(actual_save_dir, exist_ok=True)

        z_count = 1
        for z in z_vectors:
            print('Generate from vectors progress: %d/%d' % (z_count, len(z_vectors)))

            img = self.g_ema(torch.from_numpy(z).float().to(self.device), trunc, self.generator.mean_latent(4096))

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{actual_save_dir}/frame-{z_count:09d}.png')
            z_count += 1

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        
        items.pop("v_num", None)
        items.pop("loss", None)

        if self.kimg_callback is not None:
            items = self.kimg_callback.update_progress_items(items)

        if self.r_t_stat is not None:
            items['r_t_stat'] = '{:.3f}'.format(self.r_t_stat)

        if self.ada_aug_p:
            items['ada_aug_p'] = '{:.6f}'.format(self.ada_aug_p)
        
        return items

    @staticmethod
    def _requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("AliasFreeGAN Model")
        parser.add_argument("--size", help='Pixel dimension of model. Must be 256, 512, or 1024. Required!', type=int, required=True)
        parser.add_argument("--batch", help='Batch size. Will be overridden if --auto_scale_batch_size is used. (default: %(default)s)', default=16, type=int) # TODO add support for --auto_scale_batch_size
        parser.add_argument("--lr_g", help='Generator learning rate. (default: %(default)s)', default=2e-3, type=float)
        parser.add_argument("--lr_d", help='Discriminator learning rate. (default: %(default)s)', default=2e-3, type=float)
        parser.add_argument("--r1", help='R1 regularization weights. (default: %(default)s)', default=10., type=float)
        parser.add_argument("--augment", help='Use augmentations. (default: %(default)s)', default=False, type=bool)
        parser.add_argument("--augment_p", help='Augment probability, the probability that augmentation is applied. 0.0 is 0 percent and 1.0 is 100. If set to 0.0 and augment is enabled AdaptiveAugmentation will be used. (default: %(default)s)', default=0., type=float)
        parser.add_argument("--ada_target", help='Target for AdaptiveAugmentation. (default: %(default)s)', default=0.6, type=float)
        parser.add_argument("--ada_length", help='(default: %(default)s)', default=(500 * 1000), type=int)
        parser.add_argument("--ada_every", help='How often to update augmentation probabilities when using AdaptiveAugmentation. (default: %(default)s)', default=8, type=int)
        parser.add_argument("--stylegan2_discriminator", help='Provide path to a rosinality stylegan2 checkpoint to load the discriminator from it. Will load second so if you load another model first it will override that discriminator.', type=str)
        return parent_parser


    @staticmethod
    def add_generate_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("AliasFreeGenerator")
        parser.add_argument("--size", help='Pixel dimension of model. Must be 256, 512, or 1024. Required!', type=int, required=True)
        return parent_parser
