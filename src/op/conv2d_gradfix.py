import contextlib
import warnings

import torch
from torch import autograd
from torch.nn import functional as F

def conv2d(in_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups:int = 1) -> torch.Tensor:
    return F.conv2d(
        input=in_tensor,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )