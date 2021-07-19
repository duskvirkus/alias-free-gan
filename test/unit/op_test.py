import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import src.op as arrayfire_op
import src.stylegan2.op as cuda_op

def test_op_api():

    # conv2d_gradfix.py

    assert(cuda_op.conv2d_gradfix.no_weight_gradients.__name__ == arrayfire_op.conv2d_gradfix.no_weight_gradients.__name__)

    assert(cuda_op.conv2d_gradfix.conv2d.__name__ == arrayfire_op.conv2d_gradfix.conv2d.__name__)

    assert(cuda_op.conv_transpose2d.__name__ == arrayfire_op.conv_transpose2d.__name__)

    assert(cuda_op.could_use_op.__name__ == arrayfire_op.could_use_op.__name__)

    assert(cuda_op.ensure_tuple.__name__ == arrayfire_op.ensure_tuple.__name__)
    
    assert(cuda_op.conv2d_gradfix.__name__ == arrayfire_op.conv2d_gradfix.__name__)

    # fused_act.py

    assert(cuda_op.FusedLeakyReLUFunctionBackward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.__name__)
    assert(cuda_op.FusedLeakyReLUFunctionBackward.forward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.forward.__name__)
    assert(cuda_op.FusedLeakyReLUFunctionBackward.backward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.backward.__name__)

    assert(cuda_op.FusedLeakyReLUFunction.__name__ == arrayfire_op.FusedLeakyReLUFunction.__name__)
    assert(cuda_op.FusedLeakyReLUFunction.forward.__name__ == arrayfire_op.FusedLeakyReLUFunction.forward.__name__)
    assert(cuda_op.FusedLeakyReLUFunction.backward.__name__ == arrayfire_op.FusedLeakyReLUFunction.backward.__name__)

    assert(cuda_op.FusedLeakyReLU.__name__ == arrayfire_op.FusedLeakyReLU.__name__)
    assert(cuda_op.FusedLeakyReLU.forward.__name__ == arrayfire_op.FusedLeakyReLU.forward.__name__)

    assert(cuda_op.fused_leaky_relu.__name__ == arrayfire_op.fused_leaky_relu.__name__)

    # upfindn2d.py

    assert(cuda_op.FusedLeakyReLUFunctionBackward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.__name__)
    assert(cuda_op.FusedLeakyReLUFunctionBackward.forward.__name__ == arrayfire_op.FusedLeakyReLUFunctionBackward.forward.__name__)

    assert(cuda_op.FusedLeakyReLUFunction.__name__ == arrayfire_op.FusedLeakyReLUFunction.__name__)
    assert(cuda_op.FusedLeakyReLUFunction.forward.__name__ == arrayfire_op.FusedLeakyReLUFunction.forward.__name__)
    assert(cuda_op.FusedLeakyReLUFunction.backward.__name__ == arrayfire_op.FusedLeakyReLUFunction.backward.__name__)

    assert(cuda_op.FusedLeakyReLU.__name__ == arrayfire_op.FusedLeakyReLU.__name__)
    assert(cuda_op.FusedLeakyReLU.forward.__name__ == arrayfire_op.FusedLeakyReLU.forward.__name__)

    assert(cuda_op.fused_leaky_relu.__name__ == arrayfire_op.fused_leaky_relu.__name__)