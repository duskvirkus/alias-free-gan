import abc
from enum import Enum
import os
from typing import List

import cv2

import numpy as np

from src.alias_free_gan import AliasFreeGAN

class ExtensionType(Enum):
    PNG = 0
    JPEG = 1

def get_extension_str(ext_type: ExtensionType):
    """
    Converts ExtensionType enum to string.

    Args:
        ext_type: ExtensionType enum.

    Returns:
        String with period followed by extension type.
    """
    if ext_type == ExtensionType.PNG:
        return '.png'
    elif ext_type == ExtensionType.JPEG:
        return '.jpeg'
    else:
        raise Exception(f"Invalid extension type passed to get_extension_str! Expected PNG or JPEG but got {ext_type}")

def get_extension_cv2_compression(ext_type: ExtensionType):
    """
    Converts ExtensionType enum to the expected arg for cv2.im_write

    Args:
        ext_type: ExtensionType enum.

    Returns:
        cv2.imwrite argument for given filetype.
    """
    if ext_type == ExtensionType.PNG:
        return [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext_type == ExtensionType.JPEG:
        return [cv2.IMWRITE_JPEG_QUALITY, 90]
    else:
        raise Exception(f"Invalid extension type passed to get_extension_cv2_compression! Expected PNG or JPEG but got {ext_type}")

class AbstractImageProducer(metaclass=abc.ABCMeta):

    def __init__(self, model: AliasFreeGAN, extension: ExtensionType):
        super().__init__()
        self.model = model
        self.extension = extension

    def set_to_eval(self):
        self.model.g_ema.eval()

    def generate_multiple(self, latents: List[np.array], truncation: float, save_directory: str, prefix: str = "frame-") -> None:
        """
        Generates multiple output images from model's generator using the specified creator's generate method.

        Args:
            latents: Latent space vectors in the shape of an array of [1, style_dim] (style_dim is typically 512)
            save_directory: Path to the directory where generated images are saved.
            prefix: Prefix for file naming used in save path for images.

        Returns:
            None
        """

        os.makedirs(save_directory, exist_ok=True)

        for index, latent in enumerate(latents):
            self.generate(latent, truncation, save_directory, prefix + str(index).zfill(9))

    @abc.abstractmethod
    def generate(self, latent: np.array, truncation: float, save_dir: str, save_name: str) -> None:
        """
        This is an abstract method that must be implymented by the sub classes.

        Args:
            latent: A latent space vector in shape of [1, style_dim] (style_dim is typically 512)
            truncation: Truncation to save image at.
            save_path: The path for where to save the generated image.

        Returns:
            None

        Raises:
            Exception: If latent shape is incorrect.
        """

        if latent.shape[0] != 1 or latent.shape[1] != self.model.generator.style_dim:
            raise Exception(f"Invalid latent passed to generate method in {self.__class__.__name__}. Expected shape [1, {self.model.generator.style_dim}] but got {latent.shape}.")

        os.makedirs(save_dir, exist_ok=True)
