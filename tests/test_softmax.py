import sys
sys.path.append('..')
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import tvm

from utils import print_diff, get_tensors
from tir_kernels.softmax import safe_softmax

def test_softmax(n=32,h=8,seq=256,dim=1564):
    np_x = np.random.uniform(size=(n,h,seq,dim)).astype("float32")
    np_out = np.zeros((n,h,seq,dim)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)
    tvm_out, _ = get_tensors(np_out)

    tvm_softmax_mod = tvm.IRModule({'safe_softmax': safe_softmax})
    tvm_softmax = tvm.build(tvm_softmax_mod, target="llvm")

    pt_out = F.softmax(pt_x, dim=-1)

    tvm_softmax(tvm_x, tvm_out)
    print_diff(pt_out.numpy(), tvm_out.numpy())

if __name__ == '__main__':
    test_softmax()