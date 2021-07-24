import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

import src.op as arrayfire_op
import src.op.conv2d_gradfix as arrayfire_op_conv2d_gradfix
# import src.op.upfirdn2d as arrayfire_op_upfirdn2d

import src.stylegan2.op as cuda_op
import src.stylegan2.op.conv2d_gradfix as cuda_op_conv2d_gradfix
# import src.stylegan2.op.upfirdn2d as cuda_op_upfirdn2d

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


    def test_upfindn2d(self):

        # actual input
        # input.shape=torch.Size([12, 128, 84, 336])
        # kernel.shape=torch.Size([24, 1])
        # up=(1, 4), down=1, pad=(0, 0, 13, 10)
        # input.shape=torch.Size([12, 128, 336, 336])
        # kernel.shape=torch.Size([1, 12])
        # up=1, down=(2, 1), pad=(5, 5, 0, 0)
        # input.shape=torch.Size([12, 128, 336, 168])
        # kernel.shape=torch.Size([12, 1])
        # up=1, down=(1, 2), pad=(0, 0, 5, 5)
        # input.shape=torch.Size([12, 128, 148, 148])
        # kernel.shape=torch.Size([1, 12])
        # up=(2, 1), down=1, pad=(6, 5, 0, 0)
        # input.shape=torch.Size([12, 128, 148, 296])
        # kernel.shape=torch.Size([12, 1])
        # up=(1, 2), down=1, pad=(0, 0, 6, 5)

        in_tensors = [
            torch.rand(12, 128, 84, 336),
            torch.rand(12, 128, 336, 336),
            torch.rand(12, 128, 336, 168),
            torch.rand(12, 128, 148, 148),
            torch.rand(12, 128, 148, 296)
        ]
        kernels = [
            torch.rand(24, 1),
            torch.rand(1, 12),
            torch.rand(12, 1),
            torch.rand(1, 12),
            torch.rand(12, 1)
        ]
        ups = [
            (1, 4),
            1,
            1,
            (2, 1),
            (1, 2),
        ]
        downs = [
            1,
            (2, 1),
            (1, 2),
            1,
            1,
        ]
        pads = [
            (0, 0, 13, 10),
            (5, 5, 0, 0),
            (0, 0, 5, 5),
            (6, 5, 0, 0),
            (0, 0, 6, 5)
        ]

        assert(len(in_tensors) == len(kernels))
        assert(len(in_tensors) == len(ups))
        assert(len(in_tensors) == len(downs))
        assert(len(in_tensors) == len(pads))

        for i in range(len(in_tensors)):
            cuda_result = cuda_op.upfirdn2d(
                in_tensors[i],
                kernels[i],
                ups[i],
                downs[i],
                pads[i]
            )

            arrayfire_result = arrayfire_op.upfirdn2d(
                in_tensors[i],
                kernels[i],
                ups[i],
                downs[i],
                pads[i]
            )

            assert(cuda_result.shape == arrayfire_result.shape)
            assert(torch.allclose(cuda_result, arrayfire_result))
            assert(cuda_result.dtype == arrayfire_result.dtype)
