import sys
sys.path.append('..')

import torch
import numpy as np
import tvm
import torch.nn.functional as F
from core.vision_encoder.pe import SelfAttention 
from core.vision_encoder.rope import Rope2D

from utils import print_diff, get_tensors
from pe import bb_self_attn
from rope import build_axial_freqs

def test_self_attn(width=1536, num_heads=16, grid_h=32, grid_w=32):
    DEVICE = 'cpu'
    SEQ_LEN =  grid_h*grid_w

    # Init PyTorch Self Attention
    pt_rope = Rope2D(width // num_heads)
    pt_rope.init_tensors()
    pt_rope.update_grid(DEVICE, grid_h, grid_w)
    pt_self_attn = SelfAttention(embed_dim=width, num_heads=num_heads, rope=pt_rope)
    pt_self_attn.init_tensors()

    # Init TVM Self Attention
    #self_attn_ir = bb_self_attn()
    #self_attn_ir = build_self_attn()

    mod = bb_self_attn()
    ex = tvm.relax.build(mod, target='llvm')
    tvm_self_attn = tvm.relax.VirtualMachine(ex, device=tvm.cpu())
    #self_attn_ir = tvm.compile(TVMSelfAttention, target="llvm")

    # Init Input
    # NOTE: If doing RoPE2D, will have to reformat to [b s (h d)].
    np_x = np.random.uniform(size=(1, SEQ_LEN, width)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)
    tvm_qkv_w = tvm.nd.array(pt_self_attn.in_proj_weight.detach().numpy())
    tvm_qkv_b = tvm.nd.array(pt_self_attn.in_proj_bias.detach().numpy())
    tvm_linear_w = tvm.nd.array(pt_self_attn.out_proj.weight.detach().numpy())
    tvm_linear_b = tvm.nd.array(pt_self_attn.out_proj.bias.detach().numpy())
    tvm_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))

    # Infer
    pt_out = pt_self_attn(pt_x)

    freqs = build_axial_freqs(width // num_heads, grid_h, grid_w).astype("float32")
    print(freqs.shape)
    tvm_freqs = tvm.nd.array(freqs)
    print("Frequency Shape: ", freqs.shape)
    out = tvm_self_attn['self_attn'](
        tvm_x, tvm_qkv_w, tvm_qkv_b, tvm_linear_w, tvm_linear_b, tvm_freqs)

    # tvm_self_attn['main'](tvm_x, tvm_qkv_w, tvm_qkv_b, tvm_linear_w, tvm_linear_b, 
    #    tvm.nd.array(np.array(num_heads, dtype="int32")))

    return
    
if __name__ == '__main__':
    test_self_attn()