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

from tir_kernels.layerscale import layer_scale
from tir_kernels.layernorm import layer_norm
from tir_kernels.mlp import mlp
from tir_kernels.self_attn import project_fused_qkv, fused_sdpa, project_score
from tir_kernels.rope import apply_fused_rope2d

@dataclass
class SpatialPEConfig:
    test: int = None
    
def bb_self_attn():
    """
        Use Relax block builder api to build self attention module.
    """
    bb = relax.BlockBuilder()

    n, seq, width, dim_head = T.int64(), T.int64(), T.int64(), T.int64() # Relax requires int64 but tir is int32

    x = relax.Var("x", R.Tensor((n, seq, width), "float32"))
    qkv_w = relax.Var("qkv_weight", R.Tensor((3*width, width), "float32"))
    qkv_b = relax.Var("qkv_bias", R.Tensor((3*width,), "float32"))
    linear_w = relax.Var("linear_weight", R.Tensor((width, width), "float32"))
    linear_b = relax.Var("linear_bias", R.Tensor((width,), "float32"))
    freqs = relax.Var("freqs", R.Tensor((1, seq, dim_head), "float32"))

    with bb.function("self_attn", [x, qkv_w, qkv_b, linear_w, linear_b, freqs]):
        # Get Q,K,V values 
        with bb.dataflow():
            project_fused_qkv_gv = bb.add_func(project_fused_qkv, "project_fused_qkv")
            # Calculate QKV
            qkv_res = bb.emit(
                relax.call_tir(
                    project_fused_qkv_gv,
                    args=[x, qkv_w, qkv_b],
                    out_sinfo=[
                        R.Tensor((n,seq,width), "float32"),
                        R.Tensor((n,seq,width), "float32"),
                        R.Tensor((n,seq,width), "float32")
                    ]
                )
            )
            q = bb.emit(relax.TupleGetItem(qkv_res, 0))
            k = bb.emit(relax.TupleGetItem(qkv_res, 1))
            v = bb.emit(relax.TupleGetItem(qkv_res, 2))

        # Apply RoPE2D to Q,K,V
        with bb.dataflow():
            # Apply RoPE2D
            apply_rope2d_gv = bb.add_func(apply_fused_rope2d, "apply_rope2d")
            # TODO: Consider packing Q,K,V back into a 3*width tensor.
            q = bb.emit(relax.call_tir(apply_rope2d_gv, args=[q,freqs],out_sinfo=R.Tensor((n,seq,width),"float32")))
            k = bb.emit(relax.call_tir(apply_rope2d_gv, args=[k,freqs],out_sinfo=R.Tensor((n,seq,width),"float32")))

        ## Scalar Dot Product Attention
        ## Softmax(QK^T/sqrt(dim_head))*V
        with bb.dataflow():
            # Apply Scalar Dot Product
            sdpa_gv = bb.add_func(fused_sdpa, "fused_sdpa")
            score = bb.emit(relax.call_tir(
                sdpa_gv,
                args=[q, k, v, dim_head],
                out_sinfo=R.Tensor((n,seq,width), "float32")
            ))

            ## Apply linear projection
            # Embed all the heads into one embedding using a linear projection
            project_score_gv = bb.add_func(project_score, "project_score")
            out = bb.emit(relax.call_tir(
                project_score_gv,
                args=[score, linear_w, linear_b],
                out_sinfo=R.Tensor((n,seq,width), "float32")
            ))
            bb.emit_output(out)

        bb.emit_func_output(out)
    mod = bb.get()
    return mod

def bb_res_attn_block():
    """
        Residual Attention Block to make up the transformer.
        LayerNorm(Linear(LayerNorm(MHSA + x)) + x)
    """
    bb = relax.BlockBuilder()
    n, seq, width, mlp_width, dim_head = T.int64(), T.int64(), T.int64(), T.int64(), T.int64() # Relax requires int64 but tir is int32

    x = relax.Var("x", R.Tensor((n, seq, width), "float32"))
    attn_in_proj_w = relax.Var("attn_in_proj_w", R.Tensor((3*width, width), "float32"))
    attn_in_proj_b = relax.Var("attn_in_proj_b", R.Tensor((3*width,), "float32"))
    attn_out_proj_w = relax.Var("attn_out_proj_w", R.Tensor((width, width), "float32"))
    attn_out_proj_b = relax.Var("attn_out_proj_b", R.Tensor((width,), "float32"))
    ln_1_b = relax.Var("ln_1_b", R.Tensor((width,), "float32"))
    ln_1_w = relax.Var("ln_1_w", R.Tensor((width,), "float32"))
    ln_2_b = relax.Var("ln_2_b", R.Tensor((width,), "float32"))
    ln_2_w = relax.Var("ln_2_w", R.Tensor((width,), "float32"))
    ls_1_b = relax.Var("ls_1_b", R.Tensor((width,), "float32"))
    ls_1_w = relax.Var("ls_1_w", R.Tensor((width,), "float32"))
    ls_2_b = relax.Var("ls_2_b", R.Tensor((width,), "float32"))
    ls_2_w = relax.Var("ls_2_w", R.Tensor((width,), "float32"))
    mlp_c_fc_b = relax.Var("mlp_c_fc_b", R.Tensor((width,), "float32"))
    mlp_c_fc_w = relax.Var("mlp_c_fc_w", R.Tensor((mlp_width, width), "float32"))
    mlp_c_proj_b = relax.Var("mlp_c_proj_b", R.Tensor((mlp_width,), "float32"))
    mlp_c_proj_w = relax.Var("mlp_c_proj_w", R.Tensor((width, mlp_width), "float32"))
    freqs = relax.Var("freqs", R.Tensor((1, seq, dim_head), "float32"))

    # Create Relax Module for Self Attention
    self_attn = bb_self_attn()

    with bb.function("res_attn_block", [
        x,
        attn_in_proj_b,
        attn_in_proj_w,
        attn_out_proj_b,
        attn_out_proj_w,
        ln_1_b, 
        ln_1_w, 
        ln_2_b,
        ln_2_w,
        ln_2_b,
        ls_1_b,
        ls_1_w,
        ls_2_b,
        ls_2_w,
        mlp_c_fc_b,
        mlp_c_fc_w,
        mlp_c_proj_b,
        mlp_c_proj_w
    ]):
        # x = x + LS1(MHSA(LN1(x)))
        # x = x + LS2(MLP(LN2(x)))
        layernorm_gv = bb.add_func(layer_norm, "layer_norm")
        layerscale_gv = bb.add_func(layer_scale, "layer_scale")
        mlp_gv = bb.add_func(mlp, "mlp")

        with bb.dataflow():
            ln1_out = bb.emit(relax.call_tir(
                layernorm_gv,
                args=[x, ls_1_w, ls_2_w],
                out_sinfo=R.Tensor((width,), "float32")
            ))
            bb.emit_output(ln1_out)

        with bb.dataflow():
            mhsa_out = bb.emit(relax.Call(self_attn['self_attn'], [
                ln1_out,
                attn_in_proj_w, attn_in_proj_b,
                attn_out_proj_w, attn_out_proj_b,
                freqs
            ]))
            bb.emit_output(mhsa_out)

        with bb.dataflow():
            ls1_out = bb.emit(relax.call_tir(
                layerscale_gv,
                args=[],
                out_sinfo=None
            ))
            bb.emit_output(ls1_out)
        res_x = bb.emit(x + ls1_out)

        with bb.dataflow():
            mlp_out = bb.emit(relax.call_tir(
                mlp_gv,
                args=[
                    res_x,
                    mlp_c_fc_b,
                    mlp_c_fc_w,
                    mlp_c_proj_b,
                    mlp_c_proj_w
                ],
                out_sinfo=None
            ))
            bb.emit_output(mlp_out)

        with bb.dataflow():
            ls2_out = bb.emit(relax.call_tir(
                layerscale_gv,
                args=[],
                out_sinfo=None
            ))
            bb.emit_output(ls2_out)
        res_x = bb.emit(x + ls2_out)
        bb.emit_func_output(res_x)

    mod = bb.get()
    return mod

def bb_pe_spatial():
    """
        Whole transformer model. Load all weights in from
        PE Spatial here as well.
    """
    return