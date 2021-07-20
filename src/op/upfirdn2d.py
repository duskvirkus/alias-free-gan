from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function

# import arrayfire as af

# arrayfire_setup = False

# def setup_arrayfire():

#     is_cuda = os.path.isfile('/usr/local/cuda/version.txt')

#     if is_cuda:
#         af.set_backend('cuda')
#     else:
#         af.set_backend('cpu')

#     arrayfire_setup = True

def upfirdn2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up = 1,
    down = 1,
    pad = (0, 0)
) -> torch.Tensor:
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    return upfirdn2d_native(input, kernel, up[0], up[1], down[0], down[1], pad[0], pad[1], pad[2], pad[3])


def upfirdn2d_native(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up_x: int,
    up_y: int,
    down_x: int,
    down_y: int,
    pad_x0: int,
    pad_x1: int,
    pad_y0: int,
    pad_y1: int
) -> torch.Tensor:
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)

# TODO See if you can get arrayfire working
# def upfirdn2d_af(
#     input: torch.Tensor,
#     kernel: torch.Tensor,
#     up_x: int,
#     up_y: int,
#     down_x: int,
#     down_y: int,
#     pad_x0: int,
#     pad_x1: int,
#     pad_y0: int,
#     pad_y1: int
# ) -> torch.Tensor:

#     if not arrayfire_setup:
#         setup_arrayfire()
    
#     # _, channel, in_h, in_w = input.shape
#     # input = input.reshape(-1, in_h, in_w, 1)

#     # # convert tensors to numpy arrays
    
#     # if input.is_cuda:
#     #     input = input.cpu().numpy()
#     # else:
#     #     input = input.numpy()

#     # if kernel.is_cuda:
#     #     kernel = kernel.cpu().numpy()
#     # else:
#     #     kernel = kernel.numpy()

#     # # numpy arrays to af arrays
#     # input = af.to_array(input)
#     # kernel = af.to_array(kernel)


#     _, channel, in_h, in_w = input.shape
#     input = input.reshape(-1, in_h, in_w, 1)

#     _, in_h, in_w, minor = input.shape
#     kernel_h, kernel_w = kernel.shape

#     out = input.view(-1, in_h, 1, in_w, 1, minor)
#     out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
#     out = out.view(-1, in_h * up_y, in_w * up_x, minor)

#     out = F.pad(
#         out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
#     )
#     out = out[
#         :,
#         max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
#         max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
#         :,
#     ]

#     out = out.permute(0, 3, 1, 2)
#     out = out.reshape(
#         [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
#     )
#     w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)

#     # convert to arrayfire
#     if w.is_cuda:
#         w = w.cpu().numpy()
#     else:
#         w = w.numpy()
#     w = af.to_array(w)

#     if out.is_cuda:
#         out = out.cpu().numpy()
#     else:
#         out = out.numpy()
#     out = af.to_array(out)

#     print(out.shape)
#     print(w.shape)

#     # run convolution
#     out = af.signal.convolve2(out, w)

#     # convert back to tensor
#     out = torch.from_numpy(out.to_ndarray())


#     out = out.reshape(
#         -1,
#         minor,
#         in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
#         in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
#     )
#     out = out.permute(0, 2, 3, 1)
#     out = out[:, ::down_y, ::down_x, :]

#     out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
#     out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

#     return out.view(-1, channel, out_h, out_w)

#     return torch.rand(1)
