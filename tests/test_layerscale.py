"""
    Basic test with TVM Layer Scale vs PyTorch Layer Scale
"""
import sys
sys.path.append('..')

import torch
import numpy as np
import tvm

from core.vision_encoder.pe import LayerScale
from utils import get_devices, get_tensors, print_diff
from tir_kernels.layerscale import layer_scale

def test_layerscale(n=32,s=128, width=1596):
    device = 'cpu'

    # Build tensors
    init_vals = 1e-5 * np.ones(width)
    tvm_input, pt_input = get_tensors(np.random.rand(n,s,width).astype('float32'))
    tvm_gamma, _ = get_tensors(init_vals.astype("float32"))
    tvm_out = tvm.nd.array(np.zeros((n,s,width)).astype("float32"))

    # PyTorch
    pt_layerscale = LayerScale(width, init_values=1e-5).to(device)
    pt_layerscale.init_tensors()
    
    # TVM
    tvm_layer_scale_mod = tvm.IRModule({'layer_scale': layer_scale})
    tvm_layer_scale = tvm.build(tvm_layer_scale_mod, target="llvm")

    # Test
    with torch.no_grad():
        tvm_layer_scale(tvm_input, tvm_gamma, tvm_out)
        pt_out = pt_layerscale(pt_input)

        print_diff(tvm_out.numpy(), pt_out.numpy())
    return

if __name__ == '__main__':
    test_layerscale()