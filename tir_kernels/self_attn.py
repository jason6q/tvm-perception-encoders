from tvm.script import tir as T
from tvm.script import ir as I
from tvm.script import relax as R

"""
    Basic Scalar Dot Product Attention operating on
    fused heads
"""

@T.prim_func
def project_score(s: T.handle, linear_w: T.handle, linear_b: T.handle, out: T.handle):
    return

@T.prim_func
def fused_sdpa( q: T.handle, k: T.handle, v: T.handle, num_heads: T.int32, score: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    Q = T.match_buffer((n, seq, width), "float32")
    K = T.match_buffer((n, seq, width), "float32")
    V = T.match_buffer((n, seq, width), "float32")
    SCORE = T.match_buffer((n, seq, width), "float32")

    # SDPA per head.
    # TODO: Add Fast Attention ONLINE Softmax calc?
    for _n, s, w in T.grid(n,seq, width):
        with T.block("fused_sdpa_qk"):
            vn, vs, vw = T.axis.remap("SSS", [_n, s, w])
            with T.init():
                SCORE[vn, vs, vw] = T.float32(0)

            # Calc dot products.
            # This is a row-wise dot product.
            for k in range(w // num_heads):
                SCORE[vn, vs, vw] += Q[n, vs, k] * K[n, vs, k] / math.sqrt
            

"""
    This just projects X into the Q,K,V space.
"""
@T.prim_func
def project_fused_qkv(
    n: T.int64, seq: T.int64, width: T.int64, 
    x: T.handle, 
    qkv_w: T.handle, qkv_b: T.handle, 
    out_q: T.handle, out_k: T.handle, out_v: T.handle
):
    T.func_attr({"global_symbol": "project_fused_qkv", "tir.noalias": True})

    # We're assuming weights are packed here.
    X = T.match_buffer(x, [n, seq, width], "float32")
    QKV_W = T.match_buffer(qkv_w, [3*width, width], "float32")
    QKV_B = T.match_buffer(qkv_b, [3*width], "float32")
    OUT_Q = T.match_buffer(out_q, [n, seq, width], "float32")
    OUT_K = T.match_buffer(out_k, [n, seq, width], "float32")
    OUT_V = T.match_buffer(out_v, [n, seq, width], "float32")

    # Maybe keep x packed as well?
    # Basic matmul operation.
    # NxSEQxWIDTH @ WIDTHxWIDTH = NxSEQxWIDTH
    for _n, s, w, wk in T.grid(n, seq, width, width):
        with T.block('self_attn_qkv_w'):
            vn, vs, vw, vwk = T.axis.remap("SSSS", [_n, s, w, wk])
            OUT_Q[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[vwk, vw]
            OUT_K[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[1*width+vwk, vw]
            OUT_V[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[2*width+vwk, vw]

    for n, s, w in T.grid(n, seq, width):
        with T.block('self_attn_qkv_b'):
            vn, vs, vw = T.axis.remap("SSS", [n, s, w])
            OUT_Q[vn, vs, vw] += QKV_B[vw]
            OUT_K[vn, vs, vw] += QKV_B[width + vw]
            OUT_V[vn, vs, vw] += QKV_B[2*width + vw]

"""
    TODO: A fused qkv including RoPE and SDPA into a single kernel?
"""