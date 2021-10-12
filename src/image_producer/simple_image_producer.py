import os

import torch

import numpy as np

import PIL.Image

from src.image_producer.abstract_image_producer import AbstractImageProducer, get_extension_str, ExtensionType
from src.alias_free_gan import AliasFreeGAN

class SimpleImageProducer(AbstractImageProducer):

    def __init__(self, model: AliasFreeGAN, extension: ExtensionType):
        super().__init__(model, extension)

    def generate(self, latent: np.array, truncation: float, save_dir: str, save_name: str) -> None:
        super().generate(latent, truncation, save_dir, save_name)

        save_path = os.path.join(save_dir, save_name + get_extension_str(self.extension))

        img = self.model.g_ema(torch.from_numpy(latent).float().to(self.model.device), truncation, self.model.g_ema.mean_latent(4096))

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path)

