"""
    TVM Translation of 2D RoPE for Vision Transformers.
    Refer to implementation in perception_models: https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/rope.py

    We aren't using learned rotations.

    We'll simplify our implementation here and just feed the frequencies through 
    each attention layer rather than having an independent module like in the PyTorch code.
"""
import math

import numpy as np
from einops import repeat, einsum

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I


I.ir_module
class RoPE2D:
    @T.prim_func
    def tir_project_img(image: T.handle, weights: T.handle, out: T.handle):
        # We project the patches via something similar to conv operator with stride equal to patch dims.
        # We aren't using bias for Vision Transformer.
        N,IN_CH,H,W = T.int32(), T.int32(),  T.int32(), T.int32(), T.int32()
        PATCH_W, PATCH_H = T.int32(), T.int32()
        STRIDE_W, STRIDE_H = PATCH_W, PATCH_H
        GRID_W, GRID_H = W // STRIDE_W, H // STRIDE_H
        OUT_CH = T.int32()

        IMG = T.match_buffer(image, [N,IN_CH,H,W], "float32")
        W = T.match_buffer(weights, [IN_CH,PATCH_H,PATCH_W,OUT_CH], "float32") 
        OUT = T.match_buffer(out, [N,OUT_CH], "float32") 

        for n, grid_w, grid_h in T.grid(N, GRID_W, GRID_H):
            for i,j,k,l in T.grid(IN_CH, PATCH_H, PATCH_W, OUT_CH):
                with T.block("OUT"):
                    vi,vj,vk,vl = T.axis.remap("SRRRS", [i,j,k,l])
                    with T.init():
                        OUT[n,vl]= T.float32(0)

                    # Calculate patch offset on image.
                    offset_hj, offset_wk = grid_h*PATCH_H + vj, grid_w*PATCH_W + vk

                    # Multiply weights with image
                    OUT[n,vl] = IMG[n,vi,offset_hj,offset_wk]
        return
    @T.prim_func
    def tir_apply_rope(input: T.handle, axial_freqs: T.handle):
        # Dynamic shapes.
        # Could make hd (Head Dimension) fixed
        b, seq, hd = T.int32(), T.int32(), T.int32()


def build_axial_freqs(
    head_dim: int, 
    grid_height: int, # The grid of patches
    grid_width: int,
    theta=10000,
    ):
    """
        Since we aren't using learned rotations we can build out the frequencies
        in advance and store them as TVM Arrays. This will be sent during
        the forward pass of our entire model.

        TODO: Cache in the future?
    """

    # These account for the variables m,n in the RoPE equation
    # Having these variables allow us to calculate the relative position
    # using m-n.
    x_pos = np.arange(grid_width)
    y_pos = np.arange(grid_height)

    # Each contiguous pair of elements in our embedding will be matched.
    # See paper for theta formula
    freq_dim = head_dim // 2
    print("Frequency Dimension: ", freq_dim)
    freqs = 1.0 / (theta ** (np.arange(0, freq_dim, 2)[: (freq_dim // 2)] / freq_dim))

    # For every frequency value, we'll need to multiply that against each of the token
    # positions. There is a fixed number of positions determined by the grid_width and grid_height.
    # So for grid_width possible positions, you should have grid_width * num_of_freqs values.
    freqs_x = np.einsum("..., f -> ... f", x_pos.astype(freqs.dtype), freqs)
    freqs_x = repeat(freqs_x, "... n -> ... (n r)", r=2) # Repeat this across the final axis to account for the pair 

    # Same goes for grid_height: grid_height * num_of_freqs values.
    freqs_y = np.einsum("..., f -> ... f", y_pos.astype(freqs.dtype), freqs)
    freqs_y = repeat(freqs_y, "... n -> ... (n r)", r=2) # Repeat this across the final axis to account for the pair 

    # Now build out the axial frequency grid
    # We want a matrix of the shape (grid_height, grid_width, freq_dim) where the last dimension contains the pair of freqs_x,freqs_y
    # for that x,y patch position.
    freqs_y = freqs_y[:, None] # Copy across columns, x positions
    freqs_x = freqs_x[None, :] # Copy across rows, y positions
    freqs_y = np.broadcast_to(freqs_y, (grid_height, grid_width, freq_dim))
    freqs_x = np.broadcast_to(freqs_x, (grid_height, grid_width, freq_dim))

    # Flatten it out into a regular token embedding sequence now.
    # (B,H,W,freq_dim*2) -> (B,H*W,freq_dim*2)
    freqs = np.concatenate([freqs_x,freqs_y], axis=-1).reshape(grid_height * grid_width, -1)
    return freqs[None,:]