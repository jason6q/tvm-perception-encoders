import sys
sys.path.append('..')

import torch
import numpy as np
import tvm

from core.vision_encoder.rope import Rope2D
from core.vision_encoder.pe import SelfAttention
from utils import get_devices, get_tensors, print_diff

def test_self_attn(embed_dim=1024, num_heads=16):
    device = 'cuda'
    tvm_device, pt_device = get_devices(device)

    # Build input
    head_dim = embed_dim // num_heads
    x = np.random.rand(1,32, embed_dim)
    tvm_x, pt_x = get_tensors(x, device)

    # PyTorch
    rope = Rope2D(embed_dim)
    pt_self_attn = SelfAttention(embed_dim, num_heads, rope)

    # TVM


    return

    
if __name__ == '__main__':
    test_self_attn()