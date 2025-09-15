import sys
sys.path.append('..')

import numpy as np
from torch.nn import LayerNorm, Parameter
import tvm

from utils import get_tensors
from tir_kernels.layernorm import layer_norm

def test_layernorm(n=1, seq=512, width=1024):
    np_x = np.random.uniform(size=(n,seq,width)).astype(np.float32)
    tvm_x, pt_x = get_tensors(np_x)
    np_gamma = np.random.uniform(size=(width,)).astype(np.float32)
    tvm_gamma, pt_gamma = get_tensors(np_gamma)
    np_beta = np.random.uniform(size=(width,)).astype(np.float32)
    tvm_beta, pt_beta = get_tensors(np_beta)

    # PyTorch Module
    pt_ln = LayerNorm(width)
    pt_ln.weight = Parameter(pt_gamma) 
    pt_ln.bias = Parameter(pt_beta)
    pt_out = pt_ln(pt_x)
    print(pt_out.shape)

    # TVM Module
    tvm_layernorm_mod = tvm.IRModule({'layer_norm': layer_norm})
    tvm_layernorm = tvm.build(tvm_layernorm_mod, target="llvm")
    tvm_layernorm['layer_norm'](pt_x, tvm_gamma, tvm_beta)

    return

if __name__ == '__main__':
    test_layernorm()