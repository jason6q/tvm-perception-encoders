import sys
sys.path.append('..')
from typing import OrderedDict

import tvm
import numpy as np
import torch.nn as nn
import torch

from tir_kernels.mlp import mlp
from utils import print_diff, get_tensors

def test_mlp(n=1,seq=1024,width=1596, mlp_width=1024):
    # Init input/output
    np_x = np.random.uniform(size=(n,seq,width)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)

    np_out = np.zeros_like(np_x).astype("float32")
    tvm_out, _ = get_tensors(np_out)

    # Init Torch Module
    pt_mlp = nn.Sequential(
        OrderedDict(
            [
                ("c_fc", nn.Linear(width, mlp_width)),
                ("gelu", nn.GELU()),
                ("c_proj", nn.Linear(mlp_width, width)),
            ]
        )
    )

    # Grab weights from PyTorch.
    pt_c_fc_w = pt_mlp[0].weight
    pt_c_fc_b = pt_mlp[0].bias
    pt_c_proj_w = pt_mlp[2].weight
    pt_c_proj_b = pt_mlp[2].bias

    # Convert weights to TVM
    tvm_c_fc_w, _ = get_tensors(pt_c_fc_w.detach().numpy().astype("float32"))
    tvm_c_fc_b, _ = get_tensors(pt_c_fc_b.detach().numpy().astype("float32"))
    tvm_c_proj_w, _ = get_tensors(pt_c_proj_w.detach().numpy().astype("float32"))
    tvm_c_proj_b, _ = get_tensors(pt_c_proj_b.detach().numpy().astype("float32"))

    # Init TVM Module
    tvm_mlp_mod = tvm.IRModule({'mlp': mlp})
    tvm_mlp = tvm.build(tvm_mlp_mod, target="llvm")

    # Test
    with torch.no_grad():
        pt_out = pt_mlp(pt_x)
        tvm_mlp(tvm_x, tvm_c_fc_w, tvm_c_fc_b, tvm_c_proj_w, tvm_c_proj_b, tvm_out)

        print_diff(pt_out.numpy(), tvm_out.numpy())

if __name__ == '__main__':
    test_mlp()