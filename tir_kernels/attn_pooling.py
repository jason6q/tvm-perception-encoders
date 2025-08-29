
from tvm.script import tir as T

@T.prim_func
def attn_pool(x: T.handle):
    n, seq, width = T.int32(), T.int32(), T.int32()
    X = T.match_buffer(x, (n,seq,width), "float32")

    return