import sys
sys.path.append('..')

import tvm
import torch
import torch.nn.functional as F
import numpy as np

import perception_models.core.vision_encoder.rope as pe_rope
from rope import build_axial_freqs
from tir_kernels.rope import image_patch_embed, half_rotate, apply_rope2d, apply_fused_rope2d

from einops import rearrange, repeat

torch.set_printoptions(precision=8)

np.random.seed(42)
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def test_half_rotate():
    x = np.arange(100).reshape(1,1,1,-1).astype("float32")
    tvm_x, pt_x = tvm.nd.array(x), torch.from_numpy(x)
    tvm_out = tvm.nd.array(np.zeros_like(x))

    half_rotate_ir = tvm.IRModule({"half_rotate": half_rotate})
    half_rotate_mod = tvm.build(half_rotate_ir, target="llvm")

    half_rotate_mod(tvm_x, tvm_out)

    pt_out = rotate_half(pt_x)
    print("Mean Absolute Difference of Half Rotate: ", (abs(pt_out.numpy() - tvm_out.numpy())).mean())

def test_rope2d(
    batch: int = 1,
    width: int = 1536, num_heads: int = 16,
    patch_size = 14, img_w: int = 448, img_h: int = 448,
) -> None:
    dim_head = width // num_heads
    patch_w, patch_h = patch_size, patch_size
    grid_h, grid_w = img_h // patch_h, img_w // patch_w
    seq = grid_h * grid_w

    # Build input embedding
    np_img = np.random.uniform(size=(batch, 3,img_h,img_w)).astype("float32")
    tvm_img = tvm.nd.array(np_img)

    image_patch_embed_ir = tvm.IRModule({'image_patch_embed': image_patch_embed})
    image_patch_embed_mod = tvm.build(image_patch_embed_ir, target="llvm")
    weights = tvm.nd.array(np.random.uniform(size=(width,3,patch_h, patch_w)).astype("float32"))
    out = tvm.nd.array(np.zeros((batch,width,seq), dtype="float32"))
    image_patch_embed_mod(tvm_img, weights, out)

    np_x, pt_x = out.numpy(), torch.tensor(out.numpy()) # [N, EMBED_DIM, SEQ]
    tvm_x = tvm.nd.array(np_x)
    print("Image Patch Embedding Shape: ", np_x.shape)

    # Get freqs for grid.
    # dim_head because axial frequencies is per head in MHA.
    freqs = build_axial_freqs(dim_head, grid_h, grid_w).astype("float32")
    tvm_freqs = tvm.nd.array(freqs)
    print("Frequency Shape: ", freqs.shape)

    ## Test RoPE2D
    pe_rope2d = pe_rope.Rope2D(dim=dim_head)

    device = torch.device('cpu')
    pe_rope2d.init_tensors()
    pe_rope2d.update_grid(device, grid_h, grid_w)
    np_q = np.arange(batch*num_heads*seq*dim_head)
    np_q = (np_q.reshape((batch,num_heads,seq,dim_head)) / np_q.max()).astype("float32")
    #np_q = np.random.uniform(size=(batch,num_heads, seq, dim_head)).astype("float32")

    pt_q, tvm_q = torch.tensor(np_q), tvm.nd.array(np_q)
    #np_k = np.random.uniform(size=(batch,num_heads,seq, dim_head)).astype("float32")
    np_k = np.arange(batch*num_heads*seq*dim_head)
    np_k = (np_q.reshape((batch,num_heads,seq,dim_head)) / np_k.max()).astype("float32")
    pt_k, tvm_k = torch.tensor(np_k), tvm.nd.array(np_k)
    print("Q Shape: ", np_q.shape)
    print("K Shape: ", np_k.shape)
    print("freqs Shape: ", tvm_freqs.shape)

    assert num_heads*dim_head == width, '{num_heads}*{dim_head} should be {width}'

    # Calculate PyTorch RoPE2D
    pt_q_rope, pt_k_rope = pe_rope2d(pt_q, pt_k)

    # Calculate TVM RoPE2D
    tvm_rope2d_ir = tvm.IRModule({'apply_rope2d': apply_rope2d})
    tvm_rope2d_mod = tvm.build(tvm_rope2d_ir, target="llvm")

    tvm_outq = tvm.nd.array(np.zeros_like(np_q).astype("float32"))
    tvm_outk = tvm.nd.array(np.zeros_like(np_k).astype("float32"))
    tvm_rope2d_mod(tvm_q, tvm_freqs, tvm_outq)
    np_outq = tvm_outq.numpy()

    mad = np.mean(abs(np_outq - pt_q_rope.numpy()))
    print(f"Mean-Absolute Difference RoPE2D: {mad}")

    # Calculate TVM Fused RoPE2D
    tvm_fused_rope2d_ir = tvm.IRModule({'apply_fused_rope2d': apply_fused_rope2d})
    tvm_fused_rope2d_mod = tvm.build(tvm_fused_rope2d_ir, target="llvm")

    np_q_permuted = np_q.transpose((0,2,1,3)).reshape((batch,seq,num_heads*dim_head))
    tvm_q = tvm.nd.array(np_q_permuted.astype("float32"))
    tvm_outq = tvm.nd.array(np.zeros_like(np_q_permuted).astype("float32"))
    tvm_fused_rope2d_mod(tvm_q, freqs, tvm_outq)

    np_outq = tvm_outq.numpy().reshape((batch,seq,num_heads,dim_head)).transpose((0,2,1,3))

    mad = np.mean(abs(np_outq - pt_q_rope.numpy()))
    print(f"Mean-Absolute Difference Fused RoPE2D: {mad}")

def test_embed(
    dim: int = 256, num_heads: int = 4,
    grid_w: int = 64, grid_h: int = 32,
    img_w: int = 768, img_h: int = 384
) -> None:
    dim_head = dim // num_heads
    # Build out input grid
    img_w, img_h = 768, 384
    patch_w, patch_h = img_w // grid_w, img_h // grid_h
    patch_embed_dim = dim // num_heads # Should be equal to head dim.
    seq_num = patch_w * patch_h

    print(f"Grid Size: {grid_h},{grid_w}")
    print(f"Patch Size: {patch_h},{patch_w}")
    print(f"Patch Embedding Size: {patch_embed_dim}")
    print(f"Number of patches: {seq_num}")

    # Test projection
    image_patch_embed_ir = tvm.IRModule({'image_patch_embed': image_patch_embed})
    image_patch_embed_mod = tvm.build(image_patch_embed_ir, target="llvm")

    img = tvm.nd.array(np.random.uniform(size=(1,3,img_h,img_w)).astype("float32"))
    weights = tvm.nd.array(np.random.uniform(size=(dim_head,3,patch_h, patch_w)).astype("float32"))
    out = tvm.nd.array(np.zeros((1,dim_head,grid_h*grid_w), dtype="float32"))
    image_patch_embed_mod(img, weights, out)
    out_pt = F.conv2d(torch.tensor(img.numpy()), torch.tensor(weights.numpy()), stride=(patch_h,patch_w))
    out_pt = out_pt.reshape(1,dim_head,grid_h*grid_w)
    print("PT Image Embedding Shape: ", out_pt.shape)
    print("TVM Image Embedding Shape: ", out.numpy().shape)
    print("Mean Absolute Difference of Image Embedding: ", (abs(out.numpy() - out_pt.numpy())).mean())

def test_freqs(
    dim: int = 256, num_heads: int = 4, 
    grid_w: int = 64, grid_h: int = 32
) -> None:
    dim_head = dim // num_heads

    print(f"Head Dimension: {dim_head}")
    pe_rope2d = pe_rope.Rope2D(dim=dim_head)
    device = torch.device('cpu')

    pe_rope2d.init_tensors()
    pe_rope2d.update_grid(device, grid_h, grid_w)
    rope2d_freq = pe_rope2d.freq
    my_rope2d_freq = build_axial_freqs(dim_head, grid_h, grid_w)
    mad = np.mean(abs(rope2d_freq.numpy() - my_rope2d_freq))
    print(f"Axial Freq Shape: {my_rope2d_freq.shape}")
    print(f"Mean Absolute Difference of RoPE 2D Axial Frequencies: {mad}")

if __name__ == '__main__':
    print("Test generating axial frequencies")
    test_freqs()
    test_embed()
    test_half_rotate()
    test_rope2d()