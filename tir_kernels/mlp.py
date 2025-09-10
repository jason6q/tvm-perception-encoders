"""
    This is a TVM implementation of the MLP module as seen
    in Perception Encoder's Residual Attention Block.

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
from tvm.script import tir as T
@T.prim_func
def mlp():
    return
