import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import src.op as arrayfire_op
import src.op.conv2d_gradfix as arrayfire_op_conv2d_gradfix

import src.stylegan2.op as cuda_op
import src.stylegan2.op.conv2d_gradfix as cuda_op_conv2d_gradfix

class TestOp():
    '''
    This test group is designed to insure that the arrayfire op module has the same outputs as the stylegan2 op module.

    To compare the results we test the outputs of the two against each other.

    This does not guarantee that the modules aren't broken they could just be broken in the exact same way.
    '''
    
    def test_conv2d(self):

        # Actual Input
        # in_tensor shape: torch.Size([1, 6144, 36, 36])
        # weight shape: torch.Size([6144, 512, 3, 3])
        # no bias
        # stride: 1
        # padding: 1
        # dilation: 1
        # groups: 12
        # in_tensor shape: torch.Size([1, 6144, 36, 36])
        # weight shape: torch.Size([6144, 512, 3, 3])
        # no bias
        # stride: 1
        # padding: 1
        # dilation: 1
        # groups: 12
        # in_tensor shape: torch.Size([1, 6144, 52, 52])
        # weight shape: torch.Size([6144, 512, 3, 3])
        # no bias
        # stride: 1
        # padding: 1
        # dilation: 1
        # groups: 12
        # in_tensor shape: torch.Size([12, 512, 36, 36])
        # weight shape: torch.Size([512, 512, 1, 1])
        # bias shape: torch.Size([512])
        # stride: 1
        # padding: 0
        # dilation: 1
        # groups: 1

        in_tensors = [
            torch.rand(1, 6144, 36, 36),
            torch.rand(1, 6144, 36, 36),
            torch.rand(1, 6144, 52, 52),
            torch.rand(12, 512, 36, 36)
        ]
        weights = [
            torch.rand(6144, 512, 3, 3),
            torch.rand(6144, 512, 3, 3),
            torch.rand(6144, 512, 3, 3),
            torch.rand(512, 512, 1, 1)
        ]
        biases = [
            None,
            None,
            None,
            torch.rand(512)
        ]
        paddings = [1, 1, 1, 0]
        groups = [12, 12, 12, 1]

        assert(len(in_tensors) == len(weights))
        assert(len(in_tensors) == len(biases))
        assert(len(in_tensors) == len(paddings))
        assert(len(in_tensors) == len(groups))

        for i in range(len(in_tensors)):
            cuda_result = cuda_op_conv2d_gradfix.conv2d(
                in_tensors[i],
                weights[i],
                biases[i],
                1,
                paddings[i],
                1,
                groups[i]
            )
            arrayfire_result = arrayfire_op_conv2d_gradfix.conv2d(
                in_tensors[i],
                weights[i],
                biases[i],
                1,
                paddings[i],
                1,
                groups[i]
            )

            assert(cuda_result.shape == arrayfire_result.shape)
            assert(torch.allclose(cuda_result, arrayfire_result))
            assert(cuda_result.dtype == arrayfire_result.dtype)

# def test_op_api():

#     # conv2d_gradfix.py

#     assert(cuda_op.conv2d_gradfix.no_weight_gradients.__name__ == arrayfire_op.conv2d_gradfix.no_weight_gradients.__name__)

#     assert(cuda_op.conv2d_gradfix.conv2d.__name__ == arrayfire_op.conv2d_gradfix.conv2d.__name__)

#     assert(cuda_op.conv_transpose2d.__name__ == arrayfire_op.conv_transpose2d.__name__)

#     assert(cuda_op.could_use_op.__name__ == arrayfire_op.could_use_op.__name__)

#     assert(cuda_op.ensure_tuple.__name__ == arrayfire_op.ensure_tuple.__name__)
    
#     assert(cuda_op.conv2d_gradfix.__name__ == arrayfire_op.conv2d_gradfix.__name__)

#     # fused_act.py

#     assert(cuda_op.FusedLeakyReLUFunctionBackward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.__name__)
#     assert(cuda_op.FusedLeakyReLUFunctionBackward.forward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.forward.__name__)
#     assert(cuda_op.FusedLeakyReLUFunctionBackward.backward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.backward.__name__)

#     assert(cuda_op.FusedLeakyReLUFunction.__name__ == arrayfire_op.FusedLeakyReLUFunction.__name__)
#     assert(cuda_op.FusedLeakyReLUFunction.forward.__name__ == arrayfire_op.FusedLeakyReLUFunction.forward.__name__)
#     assert(cuda_op.FusedLeakyReLUFunction.backward.__name__ == arrayfire_op.FusedLeakyReLUFunction.backward.__name__)

#     assert(cuda_op.FusedLeakyReLU.__name__ == arrayfire_op.FusedLeakyReLU.__name__)
#     assert(cuda_op.FusedLeakyReLU.forward.__name__ == arrayfire_op.FusedLeakyReLU.forward.__name__)

#     assert(cuda_op.fused_leaky_relu.__name__ == arrayfire_op.fused_leaky_relu.__name__)

#     # upfindn2d.py

#     assert(cuda_op.FusedLeakyReLUFunctionBackward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.__name__)
#     assert(cuda_op.FusedLeakyReLUFunctionBackward.forward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.forward.__name__)

#     assert(cuda_op.FusedLeakyReLUFunction.__name__ == arrayfire_op.FusedLeakyReLUFunction.__name__)
#     assert(cuda_op.FusedLeakyReLUFunction.forward.__name__ == arrayfire_op.FusedLeakyReLUFunction.forward.__name__)
#     assert(cuda_op.FusedLeakyReLUFunction.backward.__name__ == arrayfire_op.FusedLeakyReLUFunction.backward.__name__)

#     assert(cuda_op.FusedLeakyReLU.__name__ == arrayfire_op.FusedLeakyReLU.__name__)
#     assert(cuda_op.FusedLeakyReLU.forward.__name__ == arrayfire_op.FusedLeakyReLU.forward.__name__)

#     assert(cuda_op.fused_leaky_relu.__name__ == arrayfire_op.fused_leaky_relu.__name__)