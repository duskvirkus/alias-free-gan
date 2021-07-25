# Alias-Free GAN

⚠️ Incomplete project ⚠️

An unofficial version of Alias-Free Generative Adversarial Networks (https://arxiv.org/abs/2106.12423). This repository was heavily based on [Kim Seonghyeon's (rosinality) implementation](https://github.com/rosinality/alias-free-gan-pytorch). The goal of this version is to be maintainable, easy to use, and expand the features of existing implementations. This is built using pytorch and pytorch lightning (a framework that abstracts away much of the hardware specific code).

See open issues unsupported features, planned features, and current bugs.

| Branch | All CI | GPU pytest | TPUs pytest |
|-|-|-|-|
| `devel` | [![CI](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml/badge.svg?branch=devel)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | [![gpu pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/devel/gpu-pytest-on-gcloud?label="GPU devel")](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | [![tpus pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/devel/tpus-pytest-on-gcloud?label="TPUs devel")](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) |
| `stable` | [![CI](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml/badge.svg?branch=stable)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | [![gpu pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/stable/gpu-pytest-on-gcloud?label="GPU stable")](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | [![tpus pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan-pytorch-lightning/stable/tpus-pytest-on-gcloud?label="TPUs stable")](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) |

## About Branches

`devel` is the primary branch and it includes the latest features but may have breaking changes. 

`stable` branch is the latest stable release of the repository the only updates it receives between versions are to the available pre-trained models and any bug fixes.

## Consider Donating

Please consider donating to help maintain this repository. While most continous integration products are free for open source projects they do not offer GPU or TPU runtimes so this project uses Google Cloud instances for CI testing and they aren't free. Any donations using the following link will help cover the cost of running these test or go to other open source machine learning projects.

tbd

## Notebooks

- GPU Colab Training Notebook - [stable Branch]() | [devel Branch]()

- GPU Colab Inference Notebook - [stable Branch]() | [devel Branch]()

- TPU Colab Training Notebook - [stable Branch]() | [devel Branch]()

### Note about TPU training

TPU training exists currently exist simply as a proof of concept however little optimization has been done yet so it does not appear to be significantly faster than GPU training on google colab.

## Pre-trained Models

| `--resume_from` Value | Size | Description |
|-|-|-|
| `rosinality-ffhq-280000` | 1024 | Incomplete ffhq model at 280000 steps by rosinality (source: https://github.com/rosinality/alias-free-gan-pytorch/issues/3) |

