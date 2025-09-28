import sys
sys.path.append('..')
from utils import print_diff, get_tensors

import tvm
import numpy as np
from pe import bb_res_attn_block
from perception_models.core.vision_encoder.pe import ResidualAttentionBlock, Rope2D

from rope import build_axial_freqs

def test_res_attn_block(n=1, seq=2, width=1536, mlp_width=1024, num_heads=16, grid_h=32, grid_w=32, num_layers=2, 
                        mlp_ratio=8960/1536, ls_init_value=0.1):
    # Initialize some test weights.
    np_x = np.random.uniform(size=(n,seq,width)).astype("float32")
    tvm_x, pt_x = get_tensors(np_x)

    np_attn_in_proj_w = np.random.uniform(size=(3*width, width)).astype("float32")
    tvm_attn_in_proj_w, pt_attn_in_proj_w = get_tensors(np_attn_in_proj_w)

    np_attn_in_proj_b = np.random.uniform(size=(3*width)).astype("float32")
    tvm_attn_in_proj_b, pt_attn_in_proj_b = get_tensors(np_attn_in_proj_b)

    np_attn_out_proj_w = np.random.uniform(size=(width, width)).astype("float32")
    tvm_attn_out_proj_w, pt_attn_out_proj_w = get_tensors(np_attn_out_proj_w)

    np_attn_out_proj_b = np.random.uniform(size=(width)).astype("float32")
    tvm_attn_out_proj_b, pt_attn_out_proj_b = get_tensors(np_attn_out_proj_b)

    np_ln_1_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_1_w, pt_ln_1_w = get_tensors(np_ln_1_w)

    np_ln_1_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_1_b, pt_ln_1_b = get_tensors(np_ln_1_b)

    np_ln_2_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_2_w, pt_ln_2_w = get_tensors(np_ln_2_w)

    np_ln_2_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_2_b, pt_ln_2_b = get_tensors(np_ln_2_b)

    np_ls_1_gamma = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_1_gamma, pt_ls_1_gamma = get_tensors(np_ls_1_gamma)

    np_ls_2_gamma = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_2_gamma, pt_ls_2_gamma = get_tensors(np_ls_2_gamma)

    np_mlp_c_fc_w = np.random.uniform(size=(mlp_width, width)).astype("float32")
    tvm_mlp_c_fc_w, pt_mlc_c_fc_w = get_tensors(np_mlp_c_fc_w)

    np_mlp_c_fc_b = np.random.uniform(size=(mlp_width)).astype("float32")
    tvm_mlp_c_fc_b, pt_mlc_c_fc_b = get_tensors(np_mlp_c_fc_b)

    np_mlp_c_proj_w = np.random.uniform(size=(width, mlp_width)).astype("float32")
    tvm_mlp_c_proj_w, pt_mlc_c_proj_w = get_tensors(np_mlp_c_proj_w)

    np_mlp_c_proj_b = np.random.uniform(size=(width)).astype("float32")
    tvm_mlp_c_proj_b, pt_mlc_c_proj_b = get_tensors(np_mlp_c_proj_b)

    freqs = build_axial_freqs(width // num_heads, grid_h, grid_w).astype("float32")
    tvm_freqs = tvm.nd.array(freqs)

    # Initialize PyTorch Module
    pt_rope = Rope2D(width // num_heads)
    pt_rope.init_tensors()
    pt_rope.update_grid('cpu', grid_h, grid_w)
    pt_res_attn = ResidualAttentionBlock(width, num_heads, mlp_ratio, ls_init_value, rope = pt_rope)

    # Initialize TVM Module
    res_attn_block_mod = bb_res_attn_block()
    ex = tvm.relax.build(res_attn_block_mod, target='llvm')
    res_attn_block = tvm.relax.VirtualMachine(ex, device=tvm.cpu())

    # Parameters follow the order of modules and their weights
    #tvm_res_attn_out = res_attn_block['res_attn_block'](
    #    tvm_x,
    #    tvm_ln_1_w, tvm_ln_1_b,
    #    tvm_attn_in_proj_w, tvm_attn_in_proj_b,
    #    tvm_attn_out_proj_w, tvm_attn_out_proj_b,
    #    tvm_ls_1_gamma,
    #    tvm_ln_2_w, tvm_ln_2_b,
    #    tvm_mlp_c_fc_w, tvm_mlp_c_fc_b,
    #    tvm_mlp_c_proj_w, tvm_mlp_c_proj_b,
    #    tvm_ls_2_gamma,
    #    tvm_freqs
    #)

    return

if __name__ == '__main__':
    test_res_attn_block()