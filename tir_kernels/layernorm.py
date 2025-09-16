"""
    TVM implementation of torch.nn.LayerNorm
    See: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    Operates per layer across the channel dimension. So you'll only have gamma, beta
    parameters per channel across the layer. The normalization happens over the features
    of that individual sample. 
"""
from tvm.script import tir as T
@T.prim_func
def layer_norm(x: T.handle, gamma_w: T.handle, beta_w: T.handle, out: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    X = T.match_buffer(x, [n,seq,width], "float32")
    GAMMA = T.match_buffer(gamma_w, [width], "float32")
    BETA = T.match_buffer(beta_w, [width], "float32")
    OUT = T.match_buffer(out, [n,seq,width], "float32")
    STD = T.alloc_buffer([n, seq], "float32")
    MU = T.alloc_buffer([n, seq], "float32")

    # Calculate mean and standard deviation.
    # TODO: Implement online algorithm instead. This is the naive approach.
    for _n, s, w in T.grid(n, seq, width):
        with T.block("layer_norm_mu"):
            vn,vs,vw = T.axis.remap("SSR", [_n,s,w])
            with T.init():
                MU[vn, vs] = T.float32(0)
            MU[vn, vs] += X[n, vs, vw]

    for _n, s in T.grid(n, seq):
        with T.block("layer_norm_div"):
            vn, vs = T.axis.remap("SS", [_n,s])
            MU[vn,vs] = MU[vn,vs] / width

    for _n, s, w in T.grid(n, seq, width):
        with T.block("layer_norm_std"):
            vn,vs,vw = T.axis.remap("SSR", [_n,s,w])

            with T.init():
                STD[vn, vs] = T.float32(0)
            STD[vn,vs] += T.pow((X[vn,vs,vw] - MU[vn,vs]), 2)

    for _n, s in T.grid(n, seq):
        with T.block("layer_norm_div"):
            vn, vs = T.axis.remap("SS", [_n,s])
            STD[vn,vs] = T.sqrt(STD[vn,vs] / width)
            MU[vn,vs] /= width

    # Calculate Layer Norm with gamma beta.
    for _n, s, w in T.grid(n, seq, width):
        with T.block("layer_norm"):
            vn,vs,vw = T.axis.remap("SSS", [_n,s,w])
            OUT[vn,vs,vw] = ((X[vn,vs,vw] - MU[vn,vs])*GAMMA[vw]/STD[vn,vs]) + BETA[vw]