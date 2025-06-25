import sys
sys.path.append('..')

import torch
import numpy as np
import tvm
import torch.nn.functional as F

from tir_kernels.gelu import gelu

from utils import print_diff

def test_gelu():
    np_x = np.random.uniform(size=(1,10,100)).astype("float32")
    tvm_x, pt_x = tvm.nd.array(np_x), torch.from_numpy(np_x)
    tvm_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))

    mod = tvm.IRModule({"gelu": gelu})
    mod = tvm.build(gelu, target='llvm')

    pt_out = F.gelu(pt_x)
    mod(tvm_x, tvm_out)
    print_diff(pt_out.numpy(), tvm_out.numpy())

if __name__ == '__main__':
    test_gelu()