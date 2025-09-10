"""
    TVM implementation of torch.nn.LayerNorm
    See: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
"""
from tvm.script import tir as T
@T.prim_func
def layer_norm(x, gamma_w, beta_w):

    return