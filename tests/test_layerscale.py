"""
    Basic test with TVM Layer Scale vs PyTorch Layer Scale
"""
import sys
sys.path.append('..')

import torch
import numpy as np
import tvm

from core.vision_encoder.pe import LayerScale
from pe import NNLayerScale
from compile import compile
from utils import get_devices, get_tensors, print_diff

def test_layerscale(dim=128):
    device = 'cpu'
    tvm_device, pt_device = get_devices(device)

    # Build tensors
    init_vals = 1e-5 * np.ones(dim)
    tvm_input, pt_input = get_tensors(np.random.rand(dim).astype('float32'))
    tvm_gamma_param, _ = get_tensors(init_vals.astype('float32'))

    # PyTorch
    pt_layerscale = LayerScale(dim, init_values=1e-5).to(pt_device)
    pt_layerscale.init_tensors()
    
    # TVM
    nnmod = NNLayerScale(dim = dim)
    irmod, named_params = nnmod.export_tvm(spec=nnmod.get_default_spec())
    _, vm = compile(irmod, tvm_device)

    # Test
    with torch.no_grad():
        tvm_out = vm['forward'](tvm_input, [tvm_gamma_param])
        pt_out = pt_layerscale(pt_input)

        print_diff(tvm_out.numpy(), pt_out.numpy())
    return

if __name__ == '__main__':
    test_layerscale()