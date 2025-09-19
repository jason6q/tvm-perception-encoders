"""
    This is a TVM implementation of the MLP module as seen
    in Perception Encoder's Residual Attention Block. It comes after the MHSA.

    Which looks like this:
        self.mlp = nn.Sequential(
        OrderedDict(
            [
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model)),
            ]
        )

        Might have to fuse gelu instead of using a separate tir kernel.
"""
import math

from tvm.script import tir as T

@T.prim_func
def mlp(x: T.handle, c_fc_w: T.handle, c_fc_b: T.handle, c_proj_w: T.handle, c_proj_b: T.handle, out: T.handle):
    n, seq, width, mlp_width = T.int32(), T.int32(), T.int32(), T.int32()

    X = T.match_buffer(x, [n, seq, width], "float32")
    C_FW_W = T.match_buffer(c_fc_w, [mlp_width, width], "float32")
    C_FW_B = T.match_buffer(c_fc_b, [mlp_width], "float32")
    C_PROJ_W = T.match_buffer(c_proj_w, [width, mlp_width], "float32")
    C_PROJ_B = T.match_buffer(c_proj_b, [width], "float32")

    OUT_1 = T.alloc_buffer([n,seq,mlp_width], "float32")
    OUT_2 = T.match_buffer(out, [n,seq,width], "float32")

    for n, s, mw, w in T.grid(n, seq, mlp_width, width):
        with T.block("mlp_c_fc_w"):
            vn, vs, vmw, vw = T.axis.remap("SSSR", [n,s,mw,w])

            with T.init():
                OUT_1[vn,vs,vmw] = T.float32(0)
            OUT_1[vn,vs,vmw] += X[vn,vs,vw] * C_FW_W[vmw, vw] 

    for n,s,mw in T.grid(n,seq,mlp_width):
        with T.block("mlp_c_fc_b"):
            vn, vs, vmw = T.axis.remap("SSS", [n,s,mw])
            OUT_1[vn,vs,vmw] += C_FW_B[vmw]

    for n, s, mw in T.grid(n,seq,mlp_width):
        with T.block("mlp_gelu"):
            vn, vs, vmw = T.axis.remap("SSS", [n,s,mw])
            pi = T.float32(math.pi)
            OUT_1[vn, vs, vmw] = 0.5 * OUT_1[vn, vs, vmw] * ( 1 + T.tanh(T.sqrt(2/pi)*(OUT_1[vn,vs,vmw] + 0.044715*T.pow(OUT_1[vn,vs,vmw],3))))

    for n, s, w, mw in T.grid(n, seq, width, mlp_width):
        with T.block("mlp_c_proj_w"):
            vn, vs, vw, vmw = T.axis.remap("SSSR", [n,s,w,mw])

            with T.init():
                OUT_2[vn,vs,vw] = T.float32(0)
            OUT_2[vn,vs,vw] += OUT_1[vn,vs,vmw] * C_PROJ_W[vw, vmw] 

    for n, s, w in T.grid(n, seq, width):
        with T.block("mlp_c_proj_b"):
            vn, vs, vw = T.axis.remap("SSS", [n,s,w])

            OUT_2[vn, vs, vw] += C_PROJ_B[vw]