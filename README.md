# Alias-Free GAN Pytorch Lightning

[![CI pytest on GPU](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci-gpu.yml/badge.svg)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci-gpu.yml)

[![CI pytest on TPUs](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci-gpu.yml/badge.svg)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci-tpus.yml)

Based on Kim Seonghyeon's (rosinality) implementation of Alias-Free Generative Adversarial Networks (https://arxiv.org/abs/2106.12423). This version has been adapted to pytorch lightning. The result can be used on a wider variety of hardware including TPUs.

⚠️ Incomplete Project ⚠️

Still todo:

- Finish conversion to pytorch lightning
- add checkpoints and resume from checkpoint test / resume from lightning log
- Convert upfindn2d and fused_act to ArrayFire to allow for non gpu usage.
- Add support for --auto_scale_batch_size
- get callback tensorboard working

## Notebooks

coming soon

## Examples

coming soon

## Pre-trained Models

coming soon 
