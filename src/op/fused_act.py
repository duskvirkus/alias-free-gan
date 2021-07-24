import os

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

# TODO finish type hinting
# TODO write unit test for FusedLeakyReLU

class FusedLeakyReLU(pl.LightningModule):
    def __init__(
        self,
        channel,
        bias=True,
        negative_slope: float = 0.2,
        scale: float = 2 ** 0.5
    ):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        self.bias.data = self.bias.data.to(self.device)
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(
    in_tensor: torch.Tensor,
    bias: torch.Tensor = None,
    negative_slope: float = 0.2,
    scale: float = 2 ** 0.5
) -> torch.Tensor:
    if bias is not None:
        rest_dim = [1] * (in_tensor.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                in_tensor + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    else:
        return F.leaky_relu(in_tensor, negative_slope=0.2) * scale
