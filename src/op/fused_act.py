import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

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
