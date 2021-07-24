# Alias-Free GAN Pytorch Lightning

[![CI](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml/badge.svg)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml)
[![gpu pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/main/gpu-pytest-on-gcloud)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml)
[![tpus pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/main/tpus-pytest-on-gcloud)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml)

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
