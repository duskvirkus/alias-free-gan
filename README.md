# Alias-Free GAN Pytorch Lightning

This is a fork of Kim Seonghyeon's (rosinality) implementation of Alias-Free Generative Adversarial Networks (https://arxiv.org/abs/2106.12423). This version has been adapted to pytorch lightning. The result can be used on a wider variety of hardware including TPUs.

⚠️ Incomplete Project ⚠️

Still todo:

- Finish conversion to pytorch lightning
- add checkpoints and resume from checkpoint test / resume from lightning log
- Convert upfindn2d and fused_act to ArrayFire to allow for non gpu usage.
- Add support for --auto_scale_batch_size
- get callback tensorboard working

## Notebooks

### Basic Training Colab Notebook for Alias Free GAN in pytorch lightning

<a href="https://colab.research.google.com/github/duskvirkus/alias-free-gan-pytorch/blob/main/notebooks/AliasFreeGAN_lightning_basic_training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A simple notebook for single gpu training on colab. Will likely change and might not be working so use with that in mind.

## Examples

coming soon

## Pre-trained Models

coming soon
