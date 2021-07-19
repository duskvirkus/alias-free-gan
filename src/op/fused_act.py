import os

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

import arrayfire as af

# torch::Tensor fused_bias_act(const torch::Tensor &input,
#                              const torch::Tensor &bias,
#                              const torch::Tensor &refer, int act, int grad,
#                              float alpha, float scale) {
#   CHECK_INPUT(input);
#   CHECK_INPUT(bias);

#   at::DeviceGuard guard(input.device());

#   return fused_bias_act_op(input, bias, refer, act, grad, alpha, scale);
# }
# converting to python
def fused_bias_act(
    input_val: torch.Tensor,
    bias: torch.Tensor,
    refer: torch.Tensor,
    act: int,
    grad: int,
    alpha: float,
    scale: float
) -> torch.Tensor:
    '''
    Converts the torch.Tensors to af.Array and then calls fused_bias_act_op will convert the result back to torch.Tensor and return it.

    Also checks that input and bias are continuous.
    '''

    # check that input is on the right device
    # check that input is continuous

    # check that bias is on the right device
    # check that bias is continuous

    input_np = None
    if input_val.is_cuda:
        input_np = input_val.cpu().numpy()
    else:
        input_np = input_val.numpy()

    bias_np = None
    if bias is not None:
        if bias.is_cuda:
            bias_np = bias.cpu().numpy()
        else:
            bias_np = bias.numpy()

    refer_np = None
    if refer is not None:
        if refer.is_cuda:
            refer_np = refer.cpu().numpy()
        else:
            refer_np = refer.numpy()
    

    input_af = af.from_numpy(input_np)
    bias_af = af.from_numpy(bias_np)
    refer_af = af.from_numpy(refer_np)

    op_result = fused_bias_act_op(input_af, bias_af, refer_af, act, grad, alpha, scale)

    return torch.Tensor(af.to_ndarray(op_result))


# torch::Tensor fused_bias_act_op(const torch::Tensor &input,
#                                 const torch::Tensor &bias,
#                                 const torch::Tensor &refer, int act, int grad,
#                                 float alpha, float scale);
# torch::Tensor fused_bias_act_op(const torch::Tensor &input,
#                                 const torch::Tensor &bias,
#                                 const torch::Tensor &refer, int act, int grad,
#                                 float alpha, float scale) {
#   int curDevice = -1;
#   cudaGetDevice(&curDevice);
#   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#   auto x = input.contiguous();
#   auto b = bias.contiguous();
#   auto ref = refer.contiguous();

#   int use_bias = b.numel() ? 1 : 0;
#   int use_ref = ref.numel() ? 1 : 0;

#   int size_x = x.numel();
#   int size_b = b.numel();
#   int step_b = 1;

#   for (int i = 1 + 1; i < x.dim(); i++) {
#     step_b *= x.size(i);
#   }

#   int loop_x = 4;
#   int block_size = 4 * 32;
#   int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

#   auto y = torch::empty_like(x);

#   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
#       x.scalar_type(), "fused_bias_act_kernel", [&] {
#         fused_bias_act_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
#             y.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
#             b.data_ptr<scalar_t>(), ref.data_ptr<scalar_t>(), act, grad, alpha,
#             scale, loop_x, size_x, step_b, size_b, use_bias, use_ref);
#       });

#   return y;
# }
# template <typename scalar_t>
# static __global__ void
# fused_bias_act_kernel(scalar_t *out, const scalar_t *p_x, const scalar_t *p_b,
#                       const scalar_t *p_ref, int act, int grad, scalar_t alpha,
#                       scalar_t scale, int loop_x, int size_x, int step_b,
#                       int size_b, int use_bias, int use_ref) {
#   int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

#   scalar_t zero = 0.0;

#   for (int loop_idx = 0; loop_idx < loop_x && xi < size_x;
#        loop_idx++, xi += blockDim.x) {
#     scalar_t x = p_x[xi];

#     if (use_bias) {
#       x += p_b[(xi / step_b) % size_b];
#     }

#     scalar_t ref = use_ref ? p_ref[xi] : zero;

#     scalar_t y;

#     switch (act * 10 + grad) {
#     default:
#     case 10:
#       y = x;
#       break;
#     case 11:
#       y = x;
#       break;
#     case 12:
#       y = 0.0;
#       break;

#     case 30:
#       y = (x > 0.0) ? x : x * alpha;
#       break;
#     case 31:
#       y = (ref > 0.0) ? x : x * alpha;
#       break;
#     case 32:
#       y = 0.0;
#       break;
#     }

#     out[xi] = y * scale;
#   }
# }
# converting to python
def fused_bias_act_op(
    input_val: af.Array,
    bias: af.Array,
    refer: af.Array,
    act: int,
    grad: int,
    alpha: float,
    scale: float
) -> af.Array:

    out = af.zeros(input_val.shape, dtype=input_val.dtype)

    x = input_val.contiguous()
    size_x = x.numel()

    if bias is not None:

        # size_b = bias.to_ndarray().flatten().shape[0]
        # step_b = 1

        # x_np_flat = x.to_numpy().flatten()
        # for index in range(len(x_np_flat)):
        #     x[index] += bias[(index / step_b) % size_b]
        # x += bias

        # tile bias if necessary


    y = af.zeros(x.shape, dtype=input_val.dtype)
    
    switch_val = act * 10 + grad
    if switch_val == 10 || switch_val == 11:
        y = x
    elif switch_val == 12:
        y = 0.0
    elif switch_val == 30:
        if x > 0.0:
            y = x
        else:
            y = x * alpha
    elif switch_val == 31:
        if refer > 0.0:
            y = x
        else:
            y = x * alpha
    elif switch_val == 32:
        y = 0.0
    
    out[:] = y * scale

    return out

class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(
        context,
        grad_output,
        out,
        bias: bool,
        negative_slope: float,
        scale: float
    ):
        context.save_for_backward(out)
        context.negative_slope = negative_slope
        context.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias

    @staticmethod
    def backward(context, gradgrad_input, gradgrad_bias):
        out, = context.saved_tensors
        gradgrad_out = fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, context.negative_slope, context.scale
        )

        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(
        context,
        input_val,
        bias,
        negative_slope: float,
        scale: float
    ):
        empty = input_val.new_empty(0)

        context.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused_bias_act(input_val, bias, empty, 3, 0, negative_slope, scale)
        context.save_for_backward(out)
        context.negative_slope = negative_slope
        context.scale = scale

        return out

    @staticmethod
    def backward(context, grad_output):
        out, = context.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, context.bias, context.negative_slope, context.scale
        )

        if not context.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        return fused_leaky_relu(
            input_val,
            self.bias,
            self.negative_slope,
            self.scale
        )


def fused_leaky_relu(
    input_val: torch.Tensor,
    bias=None,
    negative_slope: float = 0.2,
    scale: float = 2 ** 0.5
):
    return FusedLeakyReLUFunction.apply(input_val, bias, negative_slope, scale)
