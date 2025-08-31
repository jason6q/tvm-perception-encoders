"""
"""
import tvm.script.tir as T


@T.prim_func
def safe_softmax(x: T.handle, soft_out: T.handle):
    n, h, seq, dim = T.int32(), T.int32(), T.int32(), T.int32()
    X = T.match_buffer(x, (n,h,seq,dim), "float32")
    SOFT_OUT = T.match_buffer(soft_out, [n,h,seq,dim], "float32")

    ## Use an Safe Softmax Calculation to remove
    ## This is a CPU implementation.
    ## TODO: Do Parallel Equivalent.
    for _n, h, s in T.grid(n,h,seq):
        with T.block("fused_softmax_qkv"):
            vn, vh, vs = T.axis.remap("SSS", [_n,h,s])

            # We have to allocate the buffer here, if it were just
            # a regular variable it will not change within the for loop block.
            M_prev = T.alloc_buffer((), "float32")
            D_prev = T.alloc_buffer((), "float32")

            # init
            # This can't be reduced I don't think
            M_prev[()] = T.min_value("float32")
            D_prev[()] = T.float32(0)
            for k in range(seq):
                x_in = X[vn, vh, vs, k]
                M_k = T.max(M_prev[()], x_in)
                D_prev[()] = D_prev[()] * T.exp(M_prev[()] - M_k) + T.exp(x_in - M_k)
                M_prev[()] = M_k

            # Calculate Softmax
            # D_prev is the final accumulated normalizer.
            for k in range(seq):
                SOFT_OUT[vn, vh, vs, k] = T.exp(X[vn,vh,vs,k] - M_prev[()]) / D_prev[()]