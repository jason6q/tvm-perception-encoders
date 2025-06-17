"""
    The main modules to create the PE Model.
    We'll include a spec for every module for debugging purposes. You should
    be able to load the packed params independently for each module and test the output
    against the original pytorch model.
"""
from dataclasses import dataclass
from typing import Union, List, Optional, Dict
import math

import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn import op as F
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I

@dataclass
class SpatialPEConfig:
    test: int = None

class AttentionPooling:
    @T.prim_func
    def main(
        x: T.handle
    ):
        N = T.int32()

"""
    CDF of Gaussian Distribution with mean=0, std=1 :
        0.5 * (1 + erf(x/sqrt(2)))

    We're using the approximate formulation with tanh however. Hopefully this won't affect it too much.
"""
@I.ir_module
class GeLU:
    @T.prim_func
    def main(
        x: T.handle,
        out_x: T.handle
    ):
        N, SEQ, WIDTH = T.int32(), T.int32(), T.int32()
        X = T.match_buffer(x, [N, SEQ, WIDTH], "float32")
        OUT_X = T.match_buffer(out_x, [N, SEQ, WIDTH], "float32")

        for n, seq, w in T.grid(N, SEQ, WIDTH):
            with T.block("gelu"):
                vn, vs, vw = T.axis.remap("SSS", [n,seq,w])
                pi = T.float32(math.pi)
                OUT_X[vn, vs, vw] = 0.5 * X[vn, vs, vw] * ( 1 + T.tanh(T.sqrt(2/pi)*(X[vn,vs,vw] + 0.044715*T.pow(X[vn,vs,vw],3))))

class RoPE2DSelfAttention:
    @T.prim_func
    def main(
        x: T.handle, freqs: T.handle, 
        q_w: T.handle, q_b: T.handle, 
        k_w: T.handle, k_b: T.handle,
        v_w: T.handle, v_b: T.handle,
        out: T.handle
    ):
        N, EMBED_DIM, SEQ = T.int32(), T.int32(), T.int32()
        HEAD_DIM = T.int32()
        X = T.match_buffer(x, [N, EMBED_DIM, SEQ], "float32")
        FREQS = T.match_buffer(freqs, [N, SEQ, HEAD_DIM])

        # We're packing the weights here.
        QKV_W = T.match_buffer(q_w, [3*EMBED_DIM, EMBED_DIM], "float32")
        QKV_B = T.match_buffer(q_b, [3*EMBED_DIM], "float32")
        OUT = T.match_buffer(out, [], "float32")

        Q = T.alloc_buffer([N,], "float32")
        K = T.alloc_buffer([N,], "float32")
        V = T.alloc_buffer([N,], "float32")

        # Calculate Q, K

        # Calculate Softmax Attention

        # Calculate V

class NNLayerScale(nn.Module):
    def __init__(
        self, 
        dim, 
        init_values: float = 1e-5, 
        inplace: bool = False):

        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

        self.gamma = nn.Parameter([self.dim], dtype="float32")

    def forward(self, x: nn.Tensor):
        # We'll need the learned gamma parameter
        return x * self.gamma

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "x": nn.spec.Tensor([self.dim], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none"
                }
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

class NNSelfAttention(nn.Module):
    """
        PyTorch Reference: https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/pe.py#L90
    """
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int,
        # We Assume RoPE is in use from the pre-trained weights.
        # rope: Optional[nn.Module] = None
        ):
        super().__init__()

        # Calculate Self-Attention Dimensions.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Create Non-Bounded Parameters for QKV weights/biases
        # You could probably just store this as a regular Relax nn.Linear module as well.
        #self.in_proj_weight = nn.Parameter([3 * embed_dim, embed_dim], "float32")
        #self.in_proj_bias = nn.Parameter([3 * embed_dim], "float32")
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True, dtype="float32", out_dtype="float32")
        self.out_proj = nn.Linear(embed_dim * 3, embed_dim, bias=True, dtype="float32", out_dtype="float32")

        self.scale = self.head_dim ** (-0.5) # Softmax scale, 1/sqrt(d_k); used for numerical stability when d_k is large.

        self.attn = RoPE2DAttentionWithQKV

    def forward(self, x: nn.Tensor, freqs: nn.Tensor):#, attn_mask: nn.Tensor):
        # (embed_dim) -> (3 * embed_dim)
        proj = self.in_proj(x)

        # We'll want to break out the projections
        # into QKV now and calculate the attention and score
        # We'll use a custom TIR function that we wrote.
        q, k, v = R.op.call_tir(self.attn['forward'], )

        # We need to re-arrange QKV from B S (H D) -> B H S D (Einstein notation)

        return q,k,v


    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "x": nn.spec.Tensor(["n", self.embed_dim], "float32"),
                "freqs": nn.spec.Tensor(["n"], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none"
                }
            }
        }

        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

class ResidualAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def test(self, x: nn.Tensor):
        return x

    def forward_features(
        self,
        x: nn.Tensor,
        # layer_idx: int = -1, # We omit this from the original code because we aren't scanning through the layers; 
        # assume the model being loaded is already aligned to the down-stream task.
        norm: bool = False,
        strip_cls_token: bool = False
    ):
        return x

        
    def get_default_spec(self):
        """
            This allows us to expose only certain layers to the runtime.
        """
        mod_spec = {
            "test": {
                "x": nn.spec.Tensor(["n"], "int32"),
                "norm": bool,
                "strip_cls_token": bool
            },
            
            "forward_features": {
                "x": nn.spec.Tensor(["n"], "int32"),
            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)