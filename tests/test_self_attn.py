import sys
sys.path.append('..')
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import numpy as np
import tvm
import torch.nn.functional as F
from core.vision_encoder.rope import Rope2D
from einops import rearrange

from utils import print_diff, get_tensors
from pe import bb_self_attn
from rope import build_axial_freqs

from tir_kernels.self_attn import project_score, fused_sdpa, project_fused_qkv

#torch.manual_seed(42)
#np.random.seed(42)

class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]

        ## Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


def test_qkv_project(width=1536, seq=1024, num_heads=16):
    np_x = np.random.uniform(size=(1, seq, width)).astype("float32")
    np_linear_w = np.random.uniform(size=(3*width,width)).astype("float32")
    np_linear_b = np.ones((3*width,)).astype("float32") * 2

    tvm_x, pt_x = get_tensors(np_x)
    tvm_linear_w, pt_linear_w = get_tensors(np_linear_w)
    tvm_linear_b, pt_linear_b = get_tensors(np_linear_b)
    tvm_q_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))
    tvm_k_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))
    tvm_v_out = tvm.nd.array(np.zeros_like(np_x).astype("float32"))

    tvm_project_qkv_mod = tvm.IRModule({'project_fused_qkv': project_fused_qkv})
    tvm_project_qkv = tvm.build(tvm_project_qkv_mod, target="llvm")

    # Compare
    pt_out = F.linear(pt_x, pt_linear_w, pt_linear_b)
    proj = (
        pt_out.unflatten(-1, (3, width))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
        .contiguous()
    )
    pt_q, pt_k, pt_v = proj[0], proj[1], proj[2]
    tvm_project_qkv(tvm_x, tvm_linear_w, tvm_linear_b, tvm_q_out, tvm_k_out, tvm_v_out)

    #print(pt_out)
    #print(tvm_linear_out, tvm_linear_out.shape)
    print_diff(pt_q.numpy(), tvm_q_out.numpy())
    print_diff(pt_k.numpy(), tvm_k_out.numpy())
    print_diff(pt_v.numpy(), tvm_v_out.numpy())

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

    # This already expects the tensors to be Q,K, V
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
    pt_self_attn = SelfAttention(embed_dim=width, num_heads=num_heads, rope=pt_rope).eval()
    pt_self_attn.init_tensors()

    # Init TVM Self Attention
    mod = bb_self_attn()
    ex = tvm.relax.build(mod, target='llvm')
    tvm_self_attn = tvm.relax.VirtualMachine(ex, device=tvm.cpu())

    # Init Input
    # NOTE: If doing RoPE2D, will have to reformat to [b s (h d)].
    np_x = np.random.uniform(size=(1, SEQ_LEN, width)).astype("float32")
    #np_x = np.ones((1, SEQ_LEN, width)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)

    # I hypothesize the way weights are read in from pytorch are different from tvm.
    tvm_qkv_w = tvm.nd.array(pt_self_attn.in_proj_weight.detach().numpy())
    tvm_qkv_b = tvm.nd.array(pt_self_attn.in_proj_bias.detach().numpy())
    tvm_linear_w = tvm.nd.array(pt_self_attn.out_proj.weight.detach().numpy())
    tvm_linear_b = tvm.nd.array(pt_self_attn.out_proj.bias.detach().numpy())

    # Infer
    with torch.no_grad():
        pt_out = pt_self_attn(pt_x)

    freqs = build_axial_freqs(width // num_heads, grid_h, grid_w).astype("float32")
    tvm_freqs = tvm.nd.array(freqs)
    tvm_out = tvm_self_attn['self_attn'](
        tvm_x, tvm_qkv_w, tvm_qkv_b, tvm_linear_w, tvm_linear_b, tvm_freqs)
    print_diff(pt_out.numpy(), tvm_out.numpy())
    
if __name__ == '__main__':
    #test_sdpa()
    #test_project_score()
    test_qkv_project()
    test_self_attn()