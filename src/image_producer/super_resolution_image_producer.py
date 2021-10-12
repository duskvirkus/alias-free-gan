import os
import math

import cv2

import numpy as np
from numpy.core.shape_base import stack

import torch

from src.image_producer.abstract_image_producer import AbstractImageProducer, get_extension_str, get_extension_cv2_compression, ExtensionType
from src.alias_free_gan import AliasFreeGAN

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Return a sharpened version of the image, using an unsharp mask.
    
    source: https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

class SuperResolutionImageProducer(AbstractImageProducer):

    def __init__(self, model: AliasFreeGAN, extension: ExtensionType, upscale_factor: float, samples: int, translate_range: float, margin_crop: bool):
        super().__init__(model, extension)

        self.upscale_factor = upscale_factor
        self.samples = samples
        self.translate_range = translate_range
        self.margin_crop = margin_crop

    def generate(self, latent: np.array, truncation: float, save_dir: str, save_name: str) -> None:
        super().generate(latent, truncation, save_dir, save_name)

        latent = torch.from_numpy(latent).float().to(self.model.device)
        mean_latent = self.model.generator.mean_latent(4096)

        transform_p = self.model.generator.get_transform(
            latent, truncation=truncation, truncation_latent=mean_latent
        )

        stacked_image = None

        with torch.no_grad():
            for i in range(self.samples):

                t_x = 0.0
                t_y = 0.0

                if i != 0:
                    t_x = (np.random.random_sample() * (self.translate_range * 2)) - self.translate_range
                    t_y = (np.random.random_sample() * (self.translate_range * 2)) - self.translate_range

                transform_p[:, 2] = t_y
                transform_p[:, 3] = t_x

                print(transform_p)

                img = self.model.g_ema(latent, truncation, mean_latent, transform_p)

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = img[0].cpu().numpy()

                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                if i == 0:

                    cv2.imwrite(os.path.join(save_dir, save_name + '-raw' + get_extension_str(self.extension)), img, get_extension_cv2_compression(self.extension))


                img = img.astype(np.float32) / 255

                img_shape = img.shape
                new_width = int(img_shape[1] * self.upscale_factor)
                new_height = int(img_shape[0] * self.upscale_factor)

                img = cv2.resize(img, (new_width, new_height), cv2.INTER_NEAREST)

                if i == 0:

                    stacked_image = img

                else:

                    print(t_x, t_y)

                    # translate and add to stack
                    img = cv2.warpAffine(
                        img,
                        np.float32([
                            [1, 0, -t_x],
                            [0, 1, -t_y]
                        ]),
                        (
                            img.shape[1],
                            img.shape[0]
                        )
                    )

                    stacked_image += img

                # TODO
                print(f'{i + 1}/{self.samples}')

            stacked_image /= float(self.samples)
            stacked_image = (stacked_image*255).astype(np.uint8)

            # if self.margin_crop:
            #     img_shape = stacked_image.shape
            #     margin = math.ceil(self.translate_range)
            #     stacked_image = stacked_image[margin:img_shape[0] - margin, margin:img_shape[1] - margin]

            cv2.imwrite(os.path.join(save_dir, save_name + '-super-res' + get_extension_str(self.extension)), stacked_image, get_extension_cv2_compression(self.extension))

            sharpened = unsharp_mask(stacked_image, sigma=1.2, amount=1.5)
            cv2.imwrite(os.path.join(save_dir, save_name + '-super-res-sharpened' + get_extension_str(self.extension)), stacked_image, get_extension_cv2_compression(self.extension))
