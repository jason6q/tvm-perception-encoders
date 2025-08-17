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
def fused_sdpa( q: T.handle, k: T.handle, v: T.handle, dim_head: T.int64, score: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    Q = T.match_buffer(q, (n, seq, width), "float32")
    K = T.match_buffer(k, (n, seq, width), "float32")
    V = T.match_buffer(v, (n, seq, width), "float32")
    SCORE_OUT = T.match_buffer(score, (n, seq, width), "float32")

    QK = T.alloc_buffer((n,width // dim_head,seq,seq), "float32")
    SOFT = T.alloc_buffer((n, width // dim_head ,seq,seq), "float32")

    ## Calculate QK^T w/ online softmax?
    ## We'll have to split each head out of w
    for _n, h, s1, s2 in T.grid(n, width // dim_head, seq, seq):
        with T.block("fused_sdpa_qk"):
            vn, vh, vs1, vs2 = T.axis.remap("SSSS", [_n, h, s1,s2])
            with T.init():
                QK[vn, vh, vs1, vs2] = T.float32(0)

            # Calc dot products.
            # This is a row-wise dot product (Because we assume k isn't transposed.)
            for k in T.serial(0, dim_head):
                QK[vn, vh, vs1, vs2] += Q[vn, vs1, dim_head*vh + k] * K[vn, vs2, dim_head*vh + k] / T.sqrt(T.float32(dim_head))

    ## Use an Safe Softmax Calculation to remove
    ## This is a CPU implementation.
    ## TODO: Do Parallel Equivalent.
    for _n, h, s in T.grid(n,width // dim_head, seq):
        with T.block("fused_softmax_qkv"):
            vn, vh, vs = T.axis.remap("SSS", [_n,h,s])

            # We have to allocate the buffer here, if it were just
            # a regular variable it will not change within the for loop block.
            M_prev = T.alloc_buffer((), "float32")
            D_prev = T.alloc_buffer((), "float32")

            # init
            M_prev[()] = T.min_value("float32")
            D_prev[()] = T.float32(0)
            for k in range(seq):               # <-- min=0 OK
                x = QK[vn, vh, vs, k]
                M_k = T.max(M_prev[()], x)
                D_prev[()] = D_prev[()] * T.exp(M_prev[()] - M_k) + T.exp(x - M_k)
                M_prev[()] = M_k

            # Calculate Softmax
            # D_prev is the final accumulated normalizer.
            for k in range(seq):
                SOFT[vn, vh, vs, k] = T.exp(QK[vn,vh,vs,k] - M_prev[()]) / D_prev[()]

    ## Score calculation
    for _n, s, w in T.grid(n, seq, width):
        with T.block("fused_score"):
            vn, vs, vw = T.axis.remap("SSS", [_n,s,w])

            with T.init():
                SCORE_OUT[vn,vs,vw] = T.float32(0)

            for k in range(seq):
                SCORE_OUT[vn, vs, vw] += SOFT[vn, vw // dim_head, vs, k] * V[vn, k, vw] 

"""
    This just projects X into the Q,K,V space.
"""
@T.prim_func
def project_fused_qkv(
    x: T.handle, 
    qkv_w: T.handle, qkv_b: T.handle, 
    out_q: T.handle, out_k: T.handle, out_v: T.handle
):
    T.func_attr({"global_symbol": "project_fused_qkv", "tir.noalias": True})
    n, seq, width = T.int32(), T.int32(), T.int32()

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
            OUT_Q[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[vw, vwk]
            OUT_K[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[1*width+vw, vwk]
            OUT_V[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[2*width+vw, vwk]

    for n, s, w in T.grid(n, seq, width):
        with T.block('self_attn_qkv_b'):
            vn, vs, vw = T.axis.remap("SSS", [n, s, w])
            OUT_Q[vn, vs, vw] += QKV_B[vw]
            OUT_K[vn, vs, vw] += QKV_B[width + vw]
            OUT_V[vn, vs, vw] += QKV_B[2*width + vw]

"""
    TODO: A fused qkv including RoPE and SDPA into a single kernel?
"""