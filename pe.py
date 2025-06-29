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


#def build_self_attn():
#    # TODO: Try out BlockBuilder instead. This is rediculous.
#    # Define Symbolic shapes
#    n, seq, width = tir.Var("n", "int64"), tir.Var("seq", "int64"), tir.Var("width", "int64")
#    num_heads = tir.Var("num_heads", "int64")
#
#    # Function inputs using symbolic shapes
#    x = R.Var("x", R.TensorStructInfo((n,seq,width), "float32"))
#    qkv_w = R.Var("qkv_w", R.TensorStructInfo((3 * width, width), "float32"))
#    qkv_b = R.Var("qkv_b", R.TensorStructInfo((3 * width,), "float32"))
#    linear_w = R.Var("linear_w", R.TensorStructInfo((width,), "float32"))
#    linear_b = R.Var("linear_b", R.TensorStructInfo((width,), "float32"))
#
#    # Calculate Q, K, V
#    mod = IRModule()
#    mod["project_fused_qkv"] = project_fused_qkv
#    gv_qkv = mod.get_global_var("project_fused_qkv")
#
#    attn_call = R.call_tir(
#        gv_qkv,
#        (n, seq, width, x, qkv_w, qkv_b),
#        out_sinfo=[
#            R.TensorStructInfo((n,seq,width), "float32"),
#            R.TensorStructInfo((n,seq,width), "float32"),
#            R.TensorStructInfo((n,seq,width), "float32")
#        ]
#    )
#    #out_q = R.TupleGetItem(qkv, 0)
#
#    #out_k = R.TupleGetItem(qkv, 1)
#    #out_v = R.TupleGetItem(qkv, 2)
#
#    attn_fn = R.Function(
#        params=[x, qkv_w, qkv_b],
#        body = attn_call,
#        ret_struct_info=R.TupleStructInfo([
#            R.TensorStructInfo((n,seq,width), "float32"),
#            R.TensorStructInfo((n,seq,width), "float32"),
#            R.TensorStructInfo((n,seq,width), "float32")
#        ])
#    )
#
#    mod["self_attn"] = attn_fn
#    return mod
#
##class SelfAttention:
##
##    @R.Function
##    def main(
##        x: R.Tensor(("n","seq", "width"), dtype="float32"),
##        qkv_w: R.Tensor(("3 * width", "width"), dtype="float32"), 
##        qkv_b: R.Tensor(("3 * width",), dtype="float32"),
##        linear_w: R.Tensor(("width", "width"), dtype="float32"),
##        linear_b: R.Tensor(("width",), dtype="float32"),
##        num_heads: R.Tensor((), "int32"),
##    ) -> R.Tensor(("n", "seq", "width"), dtype="float32"):
##        #n = R.Var("n", shape=(), dtype="int32")
##        #seq = R.Var("seq", shape=(), dtype="int32")
##        #width = R.Var("width", shape=(), dtype="int32")
##        n,seq,width = T.var("int64"), T.var("int64"), T.var("int64")
##        R.match_shape(x,(n,seq,width))
##
##
##        # Apply RoPE2D to Q and K
##
##        # SDPA
##        #sdpa_res = R.call_tir(
##        #    SelfAttention,sdpa,
##        #    (out_q,out_k,out_v),
##        #    out_sinfo=[
##        #        R.Tensor((n, seq, width), "float32")
##        #    ]
##        #)
##
##        # Linear
##        return x