from torch.autograd import Function
import arrayfire as af

# #include <ATen/ATen.h>
# #include <torch/extension.h>

# torch::Tensor upfirdn2d_op(const torch::Tensor &input,
#                            const torch::Tensor &kernel, int up_x, int up_y,
#                            int down_x, int down_y, int pad_x0, int pad_x1,
#                            int pad_y0, int pad_y1);

# #define CHECK_CUDA(x)                                                          \
#   TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
# #define CHECK_CONTIGUOUS(x)                                                    \
#   TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
# #define CHECK_INPUT(x)                                                         \
#   CHECK_CUDA(x);                                                               \
#   CHECK_CONTIGUOUS(x)

# torch::Tensor upfirdn2d(const torch::Tensor &input, const torch::Tensor &kernel,
#                         int up_x, int up_y, int down_x, int down_y, int pad_x0,
#                         int pad_x1, int pad_y0, int pad_y1) {
#   CHECK_INPUT(input);
#   CHECK_INPUT(kernel);

#   at::DeviceGuard guard(input.device());

#   return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
#                       pad_y0, pad_y1);
# }

# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#   m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
# }

def upfirdn2d_op(in_arr: af.base.BaseArray, kernel: af.base.BaseArray, up_x: int, up_y: int, down_x: int, down_y: int, pad_x0: int, pad_x1: int, pad_y0: int, pad_y1: int) -> af.base.BaseArray:

    in_dims = in_arr.dims()
    kernel_dims = kernel.dims()

    out_h = (in_dims[1] * up_y + pad_y0 + pad_y1 - kernel_dims[0] + down_y) / down_y;
    out_w = (in_dims[2] * up_x + pad_x0 + pad_x1 - kernel_dims[1] + down_x) / down_x;

    out = af.data.constant(0, in_dims[0], out_h, out_w, in_dims[3], in_arr.dtype())

    mode = -1

    tile_out_h = -1
    tile_out_w = -1



class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None