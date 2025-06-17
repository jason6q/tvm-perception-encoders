"""
    TVM Translation of 2D RoPE for Vision Transformers.
    Refer to implementation in perception_models: https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/rope.py

    We aren't using learned rotations.

    We'll simplify our implementation here and just feed the frequencies through 
    each attention layer rather than having an independent module like in the PyTorch code.

    TODO: Add assertions and checks for invalid shapes.
"""
import math

import numpy as np
from einops import repeat, einsum

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I


"""
    Convert an image into patch embeddings to be used for the Vision Transformer.
    This is basically just a convolution operator with the strides equal to the patch
    width and height.
"""
@I.ir_module
class ImagePatchEmbedding:
    @T.prim_func
    def main(img: T.handle, weights: T.handle, out: T.handle):
        N,IN_CH,IMG_H,IMG_W = T.int32(), T.int32(), T.int32(), T.int32()
        PATCH_H,PATCH_W,OUT_CH = T.int32(), T.int32(), T.int32()
        IMG = T.match_buffer(img, [N,IN_CH,IMG_H,IMG_W], "float32")
        PARAM = T.match_buffer(weights, [OUT_CH, IN_CH, PATCH_H,PATCH_W], "float32") 
        OUT = T.match_buffer(out, [N, OUT_CH, (IMG_H*IMG_W)//(PATCH_H*PATCH_W)], "float32")
        _OUT = T.alloc_buffer([N, OUT_CH, IMG_H // PATCH_H, IMG_W // PATCH_W], "float32") 

        # Iterate through grid by strides.
        for n,out_ch,grid_h,grid_w in T.grid(N,OUT_CH, IMG_H // PATCH_H, IMG_W // PATCH_W):
            with T.block("conv2d"):
                vn,vout_ch,vgrid_h,vgrid_w = T.axis.remap("SSSS", [n,out_ch,grid_h,grid_w])
                with T.init():
                    _OUT[vn,vout_ch,vgrid_h,vgrid_w] = T.float32(0)
                # Iterate through weights.
                for in_ch,patch_h,patch_w in T.grid(IN_CH,PATCH_H,PATCH_W):
                    with T.block("patch"):
                        vin_ch,vpatch_h,vpatch_w = T.axis.remap("RRR", [in_ch,patch_h,patch_w])
                        offset_h, offset_w = vgrid_h*PATCH_H + vpatch_h, vgrid_w*PATCH_W + vpatch_w
                        _OUT[vn,vout_ch,vgrid_h,vgrid_w] += PARAM[vout_ch,vin_ch,vpatch_h,vpatch_w] * IMG[vn,vin_ch,offset_h,offset_w]

        # Re-shape
        # Not really sure if I should do a re-shape here; might be expensive.
        GRID_H, GRID_W = IMG_H // PATCH_H, IMG_W // PATCH_W
        for n,out_ch,grid_y,grid_x in T.grid(N,OUT_CH, GRID_H, GRID_W):
            with T.block("reshape_patch_embed"):
                vn,vout_ch,vgrid_y,vgrid_x = T.axis.remap("SSSS", [n,out_ch,grid_y,grid_x])
                with T.init():
                    OUT[vn,vout_ch, vgrid_y*GRID_W + vgrid_x] = T.float32(0)
                OUT[vn,vout_ch,vgrid_y*GRID_W + vgrid_x] += _OUT[vn,vout_ch,vgrid_y,vgrid_x]

"""
    See Page 7 of RoFormer for the formulation only applying euler's formulation.
"""
@I.ir_module
class RoPE2D:
    @T.prim_func
    def half_rotate(x: T.handle, rot_x: T.handle):
        N, NUM_HEADS, SEQ_LEN, HEAD_DIM = T.int32(), T.int32(), T.int32(), T.int32()

        X = T.match_buffer(x, [N, NUM_HEADS, SEQ_LEN, HEAD_DIM], "float32")
        ROT_X = T.match_buffer(rot_x, [N, NUM_HEADS, SEQ_LEN, HEAD_DIM], "float32")

        # We're going to rotate the second half of x and negate it for euler's
        # rotation matrix. We can then decompose the operation into a sum of
        # hadamard products later on. This will also interlace the elements.
        for n, num_heads, seq_len, embed_dim in T.grid(N, NUM_HEADS, SEQ_LEN, HEAD_DIM // 2):
            with T.block("half_rotate"):
                vn, vnh, vsl, ved = T.axis.remap("SSSS", [n, num_heads, seq_len, embed_dim])
                ROT_X[vn, vnh, vsl, ved*2] = -X[vn, vnh, vsl, HEAD_DIM // 2 + ved]
                ROT_X[vn, vnh, vsl, ved*2 + 1] = X[vn, vnh, vsl, ved]

    @T.prim_func
    def apply_rot_embed(
        embed: T.handle, freqs: T.handle, out_embed: T.handle):
        N, NUM_HEADS, SEQ_LEN, HEAD_DIM = T.int32(), T.int32(), T.int32(), T.int32()
        E = T.match_buffer(embed, [N, NUM_HEADS, SEQ_LEN, HEAD_DIM], "float32")
        OUT_E = T.match_buffer(out_embed, [N, NUM_HEADS, SEQ_LEN, HEAD_DIM], "float32")
        FREQS = T.match_buffer(freqs, [N, SEQ_LEN, HEAD_DIM], "float32")

        ROT_E = T.alloc_buffer([N, NUM_HEADS, SEQ_LEN, HEAD_DIM], "float32")

        # Copied from half_rotate func. Not sure how to call prim_func within prim_func at the moment.
        # Could use Relax function as main instead though.
        for n, num_heads, seq_len, embed_dim in T.grid(N, NUM_HEADS, SEQ_LEN, HEAD_DIM // 2):
            with T.block("half_rotate"):
                vn, vnh, vsl, ved = T.axis.remap("SSSS", [n, num_heads, seq_len, embed_dim])
                ROT_E[vn, vnh, vsl, ved*2] = -E[vn, vnh, vsl, HEAD_DIM // 2 + ved]
                ROT_E[vn, vnh, vsl, ved*2 + 1] = E[vn, vnh, vsl, ved]

        # Freqs should already be aligned 1-d row major with Q and K
        for n, num_heads, seq_len, head_dim in T.grid(N, NUM_HEADS, SEQ_LEN, HEAD_DIM):
            # Cosine is easy. Just multiply in order.
            with T.block("rot_embed"):
                vn, vnum_heads, vseq_len, vhead_dim = T.axis.remap("SSSS", [n, num_heads, seq_len, head_dim])
                OUT_E[vn, vnum_heads, vseq_len, vhead_dim] = T.cos(FREQS[vn, vseq_len, vhead_dim]) * E[vn, vnum_heads, vseq_len, vhead_dim] \
                                                             + T.sin(FREQS[vn, vseq_len, vhead_dim]) * ROT_E[vn, vnum_heads, vseq_len, vhead_dim]

    @T.prim_func
    def main(
        x: T.handle, freqs: T.handle, 
        q_w: T.handle, q_b: T.handle, 
        k_w: T.handle, k_b: T.handle,
        v_w: T.handle, v_b: T.handle,
        out: T.handle
    ):
        N, EMBED_DIM, SEQ = T.int32(), T.int32(), T.int32()
        HEAD_DIM = T.int32()
        X = T.match_buffer(x, [N, EMBED_DIM, SEQ], "float32")
        FREQS = T.match_buffer(freqs, [N, SEQ, HEAD_DIM])

        # We're packing the weights here.
        QKV_W = T.match_buffer(q_w, [3*EMBED_DIM, EMBED_DIM], "float32")
        QKV_B = T.match_buffer(q_b, [3*EMBED_DIM], "float32")
        OUT = T.match_buffer(out, [], "float32")

        Q = T.alloc_buffer([N,], "float32")
        K = T.alloc_buffer([N,], "float32")
        V = T.alloc_buffer([N,], "float32")

        # Calculate Q, K

        # Calculate Softmax Attention

        # Calculate V

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
    # NOTE: We are using a variant of RoPE2D that concatenates instead of adding.
    # therefore, half of the embedding space will be y rotations.
    freqs = np.concatenate([freqs_x,freqs_y], axis=-1).reshape(grid_height * grid_width, -1)

    return freqs[None,:]