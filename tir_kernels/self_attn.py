from tvm.script import tir as T
from tvm.script import ir as I
from tvm.script import relax as R

"""
    Basic Scalar Dot Product Attention operating on
    fused heads with online safe softmax calculation
"""


"""
    Assume that that weights are already transposed.
"""
@T.prim_func
def project_score(score: T.handle, linear_w: T.handle, linear_b: T.handle, out: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()

    SCORE = T.match_buffer(score, (n,seq,width), "float32")
    OUT = T.match_buffer(out, (n,seq,width), "float32")
    LINEAR_W = T.match_buffer(linear_w, (width,width), "float32")
    LINEAR_B = T.match_buffer(linear_b, (width,), "float32") # mimic broadcasting in Pytorch (N,S,W) + (W)

    for _n, s, w in T.grid(n, seq, width):
        with T.block("project_score"):
            vn,vs,vw = T.axis.remap("SSS", [_n,s,w])
            with T.init():
                OUT[vn, vs,vw] = T.float32(0)
            P = T.alloc_buffer((1,), dtype="float32")
            P[0] = 0
            for k in T.serial(0,width):
                P[0] += SCORE[vn,vs,k] * LINEAR_W[vw,k] # Linear_w is not transposed. So row-wise multiplying
            OUT[vn,vs,vw] = P[0] + LINEAR_B[vw]

@T.prim_func
def fused_sdpa( q: T.handle, k: T.handle, v: T.handle, num_heads: T.int32, score: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    Q = T.match_buffer(q, (n, seq, width), "float32")
    K = T.match_buffer(k, (n, seq, width), "float32")
    V = T.match_buffer(v, (n, seq, width), "float32")
    SCORE_OUT = T.match_buffer(score, (n, seq, width), "float32")

    QK = T.alloc_buffer((n,num_heads,seq,seq), "float32")
    SOFT = T.alloc_buffer((n,num_heads,seq,seq), "float32")

    M = T.alloc_buffer((seq,), "float32")
    D = T.alloc_buffer((seq,), "float32")

    ## Calculate QK^T w/ online softmax?
    ## We'll have to split each head out of w
    for _n, h, s1, s2 in T.grid(n, num_heads, seq, seq):
        with T.block("fused_sdpa_qk"):
            vn, vh, vs1, vs2 = T.axis.remap("SSSS", [_n, h, s1,s2])
            with T.init():
                QK[vn, vh, vs1, vs2] = T.float32(0)

            # Calc dot products.
            # This is a row-wise dot product (Because we assume k isn't transposed.)
            head_dim = width // num_heads
            for k in T.serial(0, width // num_heads):
                QK[vn, vh, vs1, vs2] += Q[vn, vs1, head_dim*vh + k] * K[vn, vs2, head_dim*vh + k] / T.sqrt(T.float32(head_dim))

    ## Use an Safe Softmax Calculation to remove
    ## This is a CPU implementation.
    ## TODO: Do Parallel Equivalent.
    for _n, h, s in T.grid(n,num_heads, seq):
        with T.block("fused_softmax_qkv"):
            vn, vh, vs = T.axis.remap("SSS", [_n,h,s])

            with T.init():
                M[0] = QK[vn, vh, vs, 0]
                D[0] = T.float32(1)   

            for k in range(seq):
                x_k = QK[vn, vh, vs, k + 1]
                M[k] = T.max(M[k-1], x_k)
                D[k] = D[k-1] * T.exp(M[k-1] - M[k]) + T.exp(x_k - M[k])

            # Calculate Softmax
            # D_prev is the final accumulated normalizer.
            for k in range(seq):
                SOFT[vn, vh, vs, k] = T.exp(QK[vn,vh,vs,k]) / D[seq-1]

    ## Score calculation
    for _n, s, w in T.grid(n, seq, width):
        with T.block("fused_score"):
            vn, vs, vw = T.axis.remap("SSS", [_n,s,w])

            with T.init():
                SCORE_OUT[vn,vs,vw] = T.float32(0)

            head_dim = width // num_heads
            for k in range(seq):
                SCORE_OUT[vn, vs, vw] += SOFT[vn, vw // head_dim, vs, k] * V[vn, k, vw] 

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