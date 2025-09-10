import sys
sys.path.append('..')
from utils import print_diff, get_tensors

import numpy as np
from pe import bb_res_attn_block
from perception_models.core.vision_encoder.pe import ResidualAttentionBlock, Rope2D

def test_res_attn_block(width=1536, num_heads=16, grid_h=32, grid_w=32, num_layers=2, 
                        mlp_ratio=8960/1536, ls_init_value=0.1):
    # Initialize some test weights.
    np_attn_in_proj_b = np.random.uniform(size=(3*width, width)).astype("float32")
    tvm_attn_in_proj_b, pt_attn_in_proj_b = get_tensors(np_attn_in_proj_b)
    np_attn_in_proj_w = np.random.uniform(size=(width)).astype("float32")
    tvm_attn_in_proj_w, pt_attn_in_proj_w = get_tensors(np_attn_in_proj_w)
    np_attn_out_proj_b = np.random.uniform(size=(width, width)).astype("float32")
    tvm_attn_out_proj_b, pt_attn_out_proj_b = get_tensors(np_attn_out_proj_b)
    np_attn_out_proj_w = np.random.uniform(size=(width)).astype("float32")
    tvm_attn_out_proj_w, pt_attn_out_proj_w = get_tensors(np_attn_out_proj_w)
    np_ln_1_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_1_b, pt_ln_1_b = get_tensors(np_ln_1_b)
    np_ln_1_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_1_w, pt_ln_1_w = get_tensors(np_ln_1_w)
    np_ln_2_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_2_b, pt_ln_2_b = get_tensors(np_ln_2_b)
    np_ln_2_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ln_2_w, pt_ln_2_w = get_tensors(np_ln_2_w)
    np_ls_1_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_1_b, pt_ls_1_b = get_tensors(np_ls_1_b)
    np_ls_1_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_1_w, pt_ls_1_w = get_tensors(np_ls_1_w)
    np_ls_2_b = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_2_b, pt_ls_2_b = get_tensors(np_ls_2_b)
    np_ls_2_w = np.random.uniform(size=(width)).astype("float32")
    tvm_ls_2_w, pt_ls_2_w = get_tensors(np_ls_2_w)
    np_mlp_c_fc_b = np.random.uniform(size=(width)).astype("float32")
    tvm_mlp_c_fc_b, pt_mlc_c_fc_b = get_tensors(np_mlp_c_fc_b)
    np_mlp_c_fc_w = np.random.uniform(size=(width)).astype("float32")
    tvm_mlp_c_fc_w, pt_mlc_c_fc_w = get_tensors(np_mlp_c_fc_w)
    np_mlp_c_proj_b = np.random.uniform(size=(width)).astype("float32")
    tvm_mlp_c_proj_b, pt_mlc_c_proj_b = get_tensors(np_mlp_c_proj_b)
    np_mlp_c_proj_w = np.random.uniform(size=(width)).astype("float32")
    tvm_mlp_c_proj_w, pt_mlc_c_proj_w = get_tensors(np_mlp_c_proj_w)

    # Initialize PyTorch Module
    pt_rope = Rope2D(width // num_heads)
    pt_rope.init_tensors()
    pt_rope.update_grid('cpu', grid_h, grid_w)
    pt_res_attn = ResidualAttentionBlock(width, num_heads, mlp_ratio, ls_init_value, rope = pt_rope)

    # Initialize TVM Module
    attn_block = bb_res_attn_block()

    return

if __name__ == '__main__':
    test_res_attn_block()