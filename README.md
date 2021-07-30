# Alias-Free GAN

⚠️ Incomplete project ⚠️

An unofficial version of Alias-Free Generative Adversarial Networks (https://arxiv.org/abs/2106.12423). This repository was heavily based on [Kim Seonghyeon's (rosinality) implementation](https://github.com/rosinality/alias-free-gan-pytorch). The goal of this version is to be maintainable, easy to use, and expand the features of existing implementations. This is built using pytorch and pytorch lightning (a framework that abstracts away much of the hardware specific code).

See open issues unsupported features, planned features, and current bugs.

## Licence and Compensation Information

This project is officially licensed as an MIT project. However, it's requested that you use this repository with the intention of actively elevating historically marginalized communities. Avoid using this project for creating anything that inflicts physical or psychological violence on individuals, groups, or animals.

If you have the financial means please consider contributing to this project or when using a pretrained model.

This project takes money to run because while many continuos integration (ci) tools are free to open source projects, they do not offer the necessary hardware to run tests on GPUs or TPUs. Because of this the ci testing needs to be run using Google Cloud Platform which charges for GPU and TPU instances. Any financial contributions to this project will first go to covering those costs first.

Licenses and contributions for pretrained models are designed to be flexible so please review the information for a particular model before using it.

## About Branches

`devel` is the default branch and it includes the latest features but may have breaking changes. 

`stable` branch is the latest stable release of the repository the only updates it receives between versions are to the available pretrained models, additional notebooks, and any bug fixes.

| Branch | All CI | GPU pytest | TPUs pytest |
|-|-|-|-|
| `devel` | [![CI](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml/badge.svg?branch=devel)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | ![gpu pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan/devel/gpu-pytest-on-gcloud?label=GPU%20devel) | ![tpus pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan/devel/tpus-pytest-on-gcloud?label=TPUs%20devel) |
| `stabel` | [![CI](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml/badge.svg?branch=stabel)](https://github.com/duskvirkus/alias-free-gan-pytorch-lightning/actions/workflows/ci.yml) | ![gpu pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan/stabel/gpu-pytest-on-gcloud?label=GPU%20stabel) | ![tpus pytest on gcloud](https://badgen.net/github/checks/duskvirkus/alias-free-gan/stabel/tpus-pytest-on-gcloud?label=TPUs%20stabel) |

## Supported Model Architectures

Below is a list of supported model architecture. The hope is to support NVlabs code when it comes out.

| `model-architecture` | Base Repository | Repository Version Limit \[start,end\) | Description |
|-|-|-|-|
| `alias-free-rosinality-v1` | https://github.com/rosinality/alias-free-gan-pytorch | \[[`fixed model` commit on July 7th 2021](https://github.com/rosinality/alias-free-gan-pytorch/tree/755a22bc60dca5bd0a8caafd29a40f1412d6b754), _\) | Based on rosinality implementation after some model fixes. |

*Notes:* 
- _ means to present unless something breaks and this table is not broken.
- If bracket syntax is confusing you see: https://en.wikipedia.org/wiki/Bracket_(mathematics)#Intervals

## Notebooks

### GPU Colab Training Notebook

**`devel` Branch**

[View Notebook](https://github.com/duskvirkus/alias-free-gan/blob/devel/notebooks/GPU_Training-Alias-Free_GAN.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duskvirkus/alias-free-gan/blob/devel/notebooks/GPU_Training-Alias-Free_GAN.ipynb)

### GPU Colab Inference Notebook

coming soon

### TPU Colab Training Notebook

coming soon

### Note about TPU training

TPU training exists currently exist simply as a proof of concept however little optimization has been done yet so it does not appear to be significantly faster than GPU training on google colab.

## Pre-trained Models

!!!Auto generate table here!!!


## Contributing

### Contribute Pretrained Models

Contributing your trained models for others to transfer learn off of or use in their projects. This helps reduce the training time and resources required to use.

You can do so by creating a pull request to the `stable` branch. Add information about your model to the to `pretrained_models.json` using the template below.

```json
        {
            "model_name": "model-name-no-spaces-no-file-extension",
            "creator": "github username example:'githubusername'",
            "model_architecture": "see model architecture section",
            "description": "Describe your project.",
            "model_size": 512,
            "gdown": "Please include a link to download your model in you're pull request and I will update this", 
            "sha1": "If you know how to make a sha1 hash then you can fill this out if not leave this blank.",
            "licence_and_compensation_information": "Please put any licencing and compensation information here you would like to include. An example could be: 'This model is under a CC-BY-NC-SA Licence. Please send $5 for transfer learning off of it on paypal to name@example.com. Contact me at name@example.com if you would like to use it in a commercial project.'"
        }
```

### Contribute Notebooks

If you make a notebook and want to share it that is welcome. If it's just a notebook you can just make a pull request to `stable`. If it requires changes to the code base please open an issue to discuss which branch it should go on.

### Other Contributions

Other contributions are welcome but open an issue to discuss what you want to change/add. Unless it's a small non breaking bug fix pull requests may or may not be accepted without discussion beforehand.
