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

from tir_kernels.self_attn import project_score, fused_sdpa

def test_sdpa(width=1536, seq=1024, num_heads=16):
    head_dim = width // num_heads

    # fused_sdpa takes in a flattened shape.
    np_q = np.random.uniform(size=(1, num_heads, seq, head_dim))
    pt_q = torch.Tensor(np_q)
    tvm_q = torch.Tensor(np_q.transpose(0,2,1,3).reshape(1,seq,width))

    np_k = np.random.uniform(size=(1, num_heads, seq, head_dim))
    pt_k = torch.Tensor(np_k)
    tvm_k = torch.Tensor(np_k.transpose(0,2,1,3).reshape(1,seq,width))

    np_v = np.random.uniform(size=(1, num_heads, seq, head_dim))
    pt_v = torch.Tensor(np_v)
    tvm_v = torch.Tensor(np_v.transpose(0,2,1,3).reshape(1,seq,width))

    np_score = np.zeros_like(np_q)
    tvm_score = torch.Tensor(np_score.transpose(0,2,1,3).reshape(1,seq,width))

    pt_sdpa_out = F.scaled_dot_product_attention(pt_q, pt_k, pt_v)

    fused_sdpa_mod = tvm.IRModule({'fused_sdpa': fused_sdpa})
    tvm_fused_sdpa = tvm.build(fused_sdpa_mod, target="llvm")

    tvm_fused_sdpa(tvm_q, tvm_k, tvm_v, num_heads, tvm_score)
    tvm_score = tvm_score.numpy().reshape(1,seq,num_heads,head_dim).transpose(0,2,1,3)
    print_diff(pt_sdpa_out.numpy(), tvm_score)
    return

def test_project_score(width=1536, seq=1024):
    np_score = np.random.uniform(size=(1, seq, width)).astype("float32")
    np_linear_w = np.random.uniform(size=(width,width)).astype("float32")
    np_linear_b = np.ones((width,)).astype("float32") * 2
    tvm_score, pt_score = get_tensors(np_score)
    tvm_linear_w, pt_linear_w = get_tensors(np_linear_w)
    tvm_linear_b, pt_linear_b = get_tensors(np_linear_b)
    tvm_linear_out = tvm.nd.array(np.zeros_like(np_score).astype("float32"))

    tvm_project_score_mod = tvm.IRModule({'project_score': project_score})
    tvm_project_score = tvm.build(tvm_project_score_mod, target="llvm")
    #print(tvm_project_score_mod.script())

    # Compare
    pt_out = F.linear(pt_score, pt_linear_w, pt_linear_b)
    tvm_project_score(tvm_score, tvm_linear_w, tvm_linear_b, tvm_linear_out)

    #print(pt_out)
    #print(tvm_linear_out, tvm_linear_out.shape)
    print_diff(pt_out.numpy(), tvm_linear_out.numpy())

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

    #mod = bb_self_attn()
    #ex = tvm.relax.build(mod, target='llvm')
    #tvm_self_attn = tvm.relax.VirtualMachine(ex, device=tvm.cpu())
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

    #freqs = build_axial_freqs(width // num_heads, grid_h, grid_w).astype("float32")
    #print(freqs.shape)
    #tvm_freqs = tvm.nd.array(freqs)
    #print("Frequency Shape: ", freqs.shape)
    #out = tvm_self_attn['self_attn'](
    #    tvm_x, tvm_qkv_w, tvm_qkv_b, tvm_linear_w, tvm_linear_b, tvm_freqs)

    # tvm_self_attn['main'](tvm_x, tvm_qkv_w, tvm_qkv_b, tvm_linear_w, tvm_linear_b, 
    #    tvm.nd.array(np.array(num_heads, dtype="int32")))

    return
    
if __name__ == '__main__':
    test_project_score()
    test_sdpa()
    #test_self_attn()