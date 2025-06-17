import sys
sys.path.append('..')

import torch
import numpy as np
import tvm
import torch.nn.functional as F

from pe import GeLU

from utils import print_diff

def test_gelu():
    np_x = np.random.uniform(size=(1,10,100)).astype("float32")
    tvm_x, pt_x = tvm.nd.array(np_x), torch.from_numpy(np_x)
    tvm_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))

    gelu = tvm.compile(GeLU, target="llvm")

    pt_out = F.gelu(pt_x)
    gelu(tvm_x, tvm_out)
    print_diff(pt_out.numpy(), tvm_out.numpy())

#def test_self_attn(embed_dim=1024, num_heads=16):
#    device = 'cuda'
#    tvm_device, pt_device = get_devices(device)
#
#    # Build input
#    head_dim = embed_dim // num_heads
#    x = np.random.rand(1,32, embed_dim)
#    tvm_x, pt_x = get_tensors(x, device)
#
#    # PyTorch
#    rope = Rope2D(embed_dim)
#    pt_self_attn = SelfAttention(embed_dim, num_heads, rope)
#
#    # TVM
#
#    return

    
if __name__ == '__main__':
    test_gelu()
    #test_self_attn()