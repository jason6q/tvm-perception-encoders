import sys
sys.path.append('..')

import torch
import numpy as np
from torch.nn import LayerNorm, Parameter
import tvm

from utils import get_tensors, print_diff
from tir_kernels.layernorm import layer_norm

def test_layernorm(n=1, seq=512, width=1024):
    np_x = np.random.uniform(size=(n,seq,width)).astype(np.float32)
    tvm_x, pt_x = get_tensors(np_x)
    np_gamma = np.random.uniform(size=(width,)).astype(np.float32)
    tvm_gamma, pt_gamma = get_tensors(np_gamma)
    np_beta = np.random.uniform(size=(width,)).astype(np.float32)
    tvm_beta, pt_beta = get_tensors(np_beta)
    tvm_out = tvm.nd.array(np.zeros_like(np_x).astype(np.float32))

    # PyTorch Module
    pt_ln = LayerNorm(width)
    pt_ln.weight = Parameter(pt_gamma) 
    pt_ln.bias = Parameter(pt_beta)
    with torch.no_grad():
        pt_out = pt_ln(pt_x)

    # TVM Module
    tvm_layernorm_mod = tvm.IRModule({'layer_norm': layer_norm})
    tvm_layernorm = tvm.build(tvm_layernorm_mod, target="llvm")
    tvm_layernorm['layer_norm'](pt_x, tvm_gamma, tvm_beta, tvm_out)

    print_diff(pt_out.numpy(), tvm_out.numpy())

if __name__ == '__main__':
    test_layernorm()