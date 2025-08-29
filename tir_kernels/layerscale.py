from tvm.script import tir as T

"""
    Maybe do in-place variant?
"""
@T.prim_func
def layer_scale(x: T.handle, gamma: T.handle, x_scaled_out: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    X = T.match_buffer(x, [n,seq,width], "float32")
    GAMMA = T.match_buffer(gamma, [width], "float32")
    X_SCALED_OUT = T.match_buffer(x_scaled_out, [n, seq, width], "float32")

    for n,s,w in T.grid(n, seq, width):
        with T.block("layer_scale"):
            vn,vs,vw = T.axis.remap("SSS", [n,s,w])

            with T.init():
                X_SCALED_OUT[vn,vs,vw] = T.float32(0)

            X_SCALED_OUT = X[vn,vs,vw] * GAMMA[vw]