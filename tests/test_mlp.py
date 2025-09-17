import sys
sys.path.append('..')
from typing import OrderedDict

import tvm
import numpy as np
import torch.nn as nn

from utils import print_diff, get_tensors

def test_mlp(n=1,seq=1024,width=1596, mlp_width=1024):
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

if __name__ == '__main__':
    test_mlp()