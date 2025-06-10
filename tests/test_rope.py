import sys
sys.path.append('..')

import tvm
import torch
import torch.nn.functional as F
import numpy as np

import core.vision_encoder.rope as pe_rope
from rope import build_axial_freqs, ImagePatchEmbedding, RoPE2DAttention

np.random.seed(42)

def test_rope2d(
    dim: int = 256, num_heads: int = 4,
    grid_w: int = 64, grid_h: int = 32,
    img_w: int = 768, img_h: int = 384
) -> None:
    dim_head = dim // num_heads
    patch_w, patch_h = img_w // grid_w, img_h // grid_h

    # Build input embedding
    np_img = np.random.uniform(size=(1, 3,img_h,img_w)).astype("float32")
    tvm_img = tvm.nd.array(np_img)

    img_embed = tvm.compile(ImagePatchEmbedding, target="llvm")
    weights = tvm.nd.array(np.random.uniform(size=(dim_head,3,patch_h, patch_w)).astype("float32"))
    out = tvm.nd.array(np.zeros((1,dim_head,grid_h*grid_w), dtype="float32"))
    img_embed(tvm_img, weights, out)

    np_x, pt_x = out.numpy(), torch.tensor(out.numpy()) # [N, HEAD_DIM, SEQ]
    tvm_x = tvm.nd.array(np_x)
    print(np_x.shape)

    ## Test RoPE2D
    #pe_rope2d = pe_rope.Rope2D(dim=dim_head)
    #device = torch.device('cpu')
    #pe_rope2d.init_tensors()
    #pe_rope2d.update_grid(device, grid_h, grid_w)

    #tvm_rope2d = tvm.compile(RoPE2DAttention, target="llvm")

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
    img_embed = tvm.compile(ImagePatchEmbedding, target="llvm")

    img = tvm.nd.array(np.random.uniform(size=(1,3,img_h,img_w)).astype("float32"))
    weights = tvm.nd.array(np.random.uniform(size=(dim_head,3,patch_h, patch_w)).astype("float32"))
    out = tvm.nd.array(np.zeros((1,dim_head,grid_h*grid_w), dtype="float32"))
    img_embed(img, weights, out)
    out_pt = F.conv2d(torch.tensor(img.numpy()), torch.tensor(weights.numpy()), stride=(patch_h,patch_w))
    out_pt = out_pt.reshape(1,dim_head,grid_h*grid_w)
    print("Image Embedding Shape: ", out.numpy().shape)
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
    test_rope2d()