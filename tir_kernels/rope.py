from tvm.script import tir as T

"""
    Convert an image into patch embeddings to be used for the Vision Transformer.
    This is basically just a convolution operator with the strides equal to the patch
    width and height.
"""
@T.prim_func
def image_patch_embed(img: T.handle, weights: T.handle, out: T.handle):
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
            ROT_X[vn, vnh, vsl, ved*2] = -X[vn, vnh, vsl, ved*2 + 1]
            ROT_X[vn, vnh, vsl, ved*2 + 1] = X[vn, vnh, vsl, ved*2]

"""
    This one assumes your QKV values aren't fused... So you would have to already
    have your embeddings be (N,H,S,D//H)
"""
@T.prim_func
def apply_rope2d(
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
            ROT_E[vn, vnh, vsl, ved*2] = -E[vn, vnh, vsl, ved*2 + 1]
            ROT_E[vn, vnh, vsl, ved*2 + 1] = E[vn, vnh, vsl, ved*2]

    # Freqs should already be aligned 1-d row major with Q and K
    for n, num_heads, seq_len, head_dim in T.grid(N, NUM_HEADS, SEQ_LEN, HEAD_DIM):
        # Cosine is easy. Just multiply in order.
        with T.block("rot_embed"):
            vn, vnum_heads, vseq_len, vhead_dim = T.axis.remap("SSSS", [n, num_heads, seq_len, head_dim])
            OUT_E[vn, vnum_heads, vseq_len, vhead_dim] = T.cos(FREQS[vn, vseq_len, vhead_dim]) * E[vn, vnum_heads, vseq_len, vhead_dim] \
                                                            + T.sin(FREQS[vn, vseq_len, vhead_dim]) * ROT_E[vn, vnum_heads, vseq_len, vhead_dim]
"""
    This is applying RoPE2D to an input of shape (N, SEQ, WIDTH).
    This will also fuse the half rotate operation.
"""
@T.prim_func
def apply_fused_rope2d(embed: T.handle, freqs: T.handle, out_embed: T.handle):
    N,SEQ,WIDTH,H = T.int32(), T.int32(), T.int32(), T.int32()
    EMBED = T.match_buffer(embed, [N,SEQ,WIDTH], "float32")
    FREQS = T.match_buffer(freqs, [N,SEQ,H], "float32")
    OUT_EMBED = T.match_buffer(out_embed, [N,SEQ,WIDTH], "float32")

    for n, seq, width in T.grid(N, SEQ, WIDTH):
        with T.block("fused_rope2d"):
            vn, vseq, vwidth = T.axis.remap("SSS", [n, seq, width])
            OUT_EMBED[vn, vseq, vwidth] = T.cos(FREQS[vn,vseq, vwidth % H]) * EMBED[vn, vseq, vwidth] + \
                T.sin(FREQS[vn, vseq, vwidth % H]) *((vwidth % 2)*EMBED[vn, vseq, vwidth - 1] - (1 - (vwidth % 2))*EMBED[vn,vseq,vwidth + 1])