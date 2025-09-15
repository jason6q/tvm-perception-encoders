"""
    TVM implementation of torch.nn.LayerNorm
    See: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    Operates per layer across the channel dimension. So you'll only have gamma, beta
    parameters per channel across the layer. The normalization happens over the features
    of that individual sample. 
"""
from tvm.script import tir as T
@T.prim_func
def layer_norm(x: T.handle, gamma_w: T.handle, beta_w: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    X = T.match_buffer(x, [n,seq,width], "float32")
    GAMMA = T.match_buffer(gamma_w, [width], "float32")
    BETA = T.match_buffer(beta_w, [width], "float32")

    for _n, s, w in T.grid(n, seq, width):
        with T.block("layer_norm"):
            vn,vs,vw = T.axis.remap("SSS", [_n,s,w])