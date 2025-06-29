"""
    The main modules to create the PE Model.
    We'll include a spec for every module for debugging purposes. You should
    be able to load the packed params independently for each module and test the output
    against the original pytorch model.
"""
from dataclasses import dataclass
from typing import Union, List, Optional, Dict

import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm import relax

from tir_kernels.self_attn import project_fused_qkv

@dataclass
class SpatialPEConfig:
    test: int = None
    
    
def bb_self_attn():
    """
        Use Relax block builder api to build self attention module.
    """
    bb = relax.BlockBuilder()

    n,seq,width = T.int64(), T.int64(), T.int64()

    x = relax.Var("x", R.Tensor((n, seq, width), "float32"))
    qkv_w = relax.Var("qkv_weight", R.Tensor((3*width, width), "float32"))
    qkv_b = relax.Var("qkv_bias", R.Tensor((3*width,), "float32"))
    linear_w = relax.Var("linear_weight", R.Tensor((3*width, width), "float32"))
    linear_b = relax.Var("linear_bias", R.Tensor((3*width,), "float32"))

    with bb.function("self_attn", [x, qkv_w, qkv_b, linear_w, linear_b]):
        with bb.dataflow():
            tir_gv = bb.add_func(project_fused_qkv, "project_fused_qkv")
            gv = bb.emit(
                relax.call_tir(
                    tir_gv,
                    args=[n, seq, width, x, qkv_w, qkv_b],
                    out_sinfo=[
                        R.Tensor((n,seq,width), "float32"),
                        R.Tensor((n,seq,width), "float32"),
                        R.Tensor((n,seq,width), "float32")
                    ]
                )
            )
            bb.emit_output(gv)
        bb.emit_func_output(gv)
    mod = bb.get()
    return mod