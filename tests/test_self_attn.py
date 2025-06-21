import sys
sys.path.append('..')

import torch
import numpy as np
import tvm
import torch.nn.functional as F
from core.vision_encoder.pe import SelfAttention
from core.vision_encoder.rope import Rope2D

from pe import SelfAttention as TVMSelfAttention
from utils import print_diff, get_tensors

def test_self_attn(width=1536, num_heads=16, grid_h=32, grid_w=32):
    DEVICE = 'cpu'
    SEQ_LEN =  grid_h*grid_w

    # Init PyTorch Self Attention
    pt_rope = Rope2D(width // num_heads)
    pt_rope.init_tensors()
    pt_rope.update_grid(DEVICE, grid_h, grid_w)
    pt_self_attn = SelfAttention(embed_dim=width, num_heads=num_heads, rope=pt_rope)

    # Init TVM Self Attention
    tvm_self_attn = tvm.compile(TVMSelfAttention, target="llvm")


    # Init Input
    np_x = np.random.uniform(size=(1, SEQ_LEN, width)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)

    pt_out = pt_self_attn(pt_x)

    return

    
if __name__ == '__main__':
    test_self_attn()