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

@I.ir_module
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

@I.ir_module
class SelfAttention:
    @T.prim_func
    def project_fused_qkv(
        x: T.handle,
        qkv_w: T.handle, qkv_b: T.handle,
        out_q: T.handle, out_k: T.handle, out_v: T.handle
    ):
        N, NUM_HEADS, SEQ, HEAD_DIM, = T.int32(), T.int32(), T.int32(), T.int32()
        WIDTH = T.int32()

        # We're assuming weights are packed here.
        X = T.match_buffer(x, [N, NUM_HEADS, SEQ, HEAD_DIM], "float32")
        QKV_W = T.match_buffer(qkv_w, [3*WIDTH, WIDTH], "float32")
        QKV_B = T.match_buffer(qkv_b, [3*WIDTH], "float32")
        OUT_Q = T.match_buffer(out_q, [N, NUM_HEADS, SEQ, HEAD_DIM], "float32")
        OUT_K = T.match_buffer(out_k, [N, NUM_HEADS, SEQ, HEAD_DIM], "float32")
        OUT_V = T.match_buffer(out_v, [N, NUM_HEADS, SEQ, HEAD_DIM], "float32")

        # Maybe keep x packed as well?
        for n, nh, s, hd in T.grid(N, NUM_HEADS, SEQ, HEAD_DIM):
            with T.block('self_attn_qkv_out'):
                vn, vnh, vs, vhd = T.axis.remap("SSSS", [n, nh, s, hd])
                #head_idx = 
                #OUT_Q[vn, vnh, vs, vhd] = QKV_W[0,] + QKV_B[0]
                #OUT_K[vn, vnh, vs, vhd] = QKV_W[1*WIDTH,] + QKV_B[1*WIDTH]
                #OUT_V[vn, vnh, vs, vhd] = QKV_W[2*WIDTH,] + QKV_B[2*WIDTH]
    #@T.prim_func
    #def sdpa(q: T.handle, k: T.handle, v: T.handle):
    #    N = T.int32()
    #    return q

    @R.function
    def main(
        x: R.Tensor(("n","num_heads", "seq", "head_dim"), dtype="float32"),
        qkv_w: R.Tensor((3*"width", "width"), dtype="float32"), 
        qkv_b: R.Tensor((3*"width",), dtype="float32"),
        linear_w: R.Tensor(("width", "width"), dtype="float32"),
        linear_b: R.Tensor(("width", "width"), dtype="float32"),
        out: R.Tensor(("n",), dtype="float32") 
    ) -> R.Tensor(("n",), dtype="float32"):

        # Calculate Q, K, V

        # Apply RoPE2D to Q and K

        return out