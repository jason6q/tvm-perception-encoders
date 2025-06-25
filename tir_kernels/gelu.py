import math

from tvm.script import tir as T

"""
    CDF of Gaussian Distribution with mean=0, std=1 :
        0.5 * (1 + erf(x/sqrt(2)))

    We're using the approximate formulation with tanh however. Hopefully this won't affect it too much.
"""
@T.prim_func
def gelu(
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