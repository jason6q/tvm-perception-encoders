import sys
sys.path.append('..')
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import tvm

from utils import print_diff, get_tensors
from tir_kernels.softmax import safe_softmax, naive_softmax

# Softmax implementations may be different, but ultimately
# the argmax values should be the same.
# Softmax is a monotonically increasing function.

def test_softmax_arg(n=32,h=8,seq=256,dim=256,n_samples=10):
    tvm_softmax_mod = tvm.IRModule({'safe_softmax': safe_softmax})
    tvm_softmax = tvm.build(tvm_softmax_mod, target="llvm")

    tot_num_trues = 0
    tot_num_falses = 0
    tot_num_vals = 0
    for i in range(n_samples):
        np_x = np.random.uniform(size=(n,h,seq,dim)).astype("float32")
        np_out = np.zeros((n,h,seq,dim)).astype("float32")
        tvm_x, pt_x = get_tensors(np_x)
        tvm_out, _ = get_tensors(np_out)

        pt_out = F.softmax(pt_x, dim=-1)

        tvm_softmax(tvm_x, tvm_out)
        tvm_amaxes = np.argmax(tvm_out.numpy(), axis=-1)
        pt_amaxes = torch.argmax(pt_out, dim=-1)

        num_true = (tvm_amaxes == pt_amaxes).sum()
        num_false = (~(tvm_amaxes == pt_amaxes)).sum()
        tot_num_trues += num_true
        tot_num_falses += num_false
        tot_num_vals += n*h*seq
    print(f"Average number of argmax softmax values that are the same: {tot_num_trues / tot_num_vals}")
    print(f"Average number of argmax softmax values that are different: {tot_num_falses / tot_num_vals}")

def test_softmax(n=32,h=8,seq=256,dim=256):
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
    test_softmax_arg()