import sys
sys.path.append('..')

import tvm
import torch
import torch.nn.functional as F
import numpy as np

import core.vision_encoder.rope as pe_rope
from rope import build_axial_freqs, ImagePatchEmbedding

np.random.seed(42)

def test_freqs() -> None:
    dim = 256
    num_heads=4
    grid_w, grid_h = 64,32
    dim_head = dim // num_heads

    print(f"Head Dimension: {dim_head}")
    pe_rope2d = pe_rope.Rope2D(dim=dim_head)
    device = torch.device('cpu')

    pe_rope2d.init_tensors()
    pe_rope2d.update_grid(device, grid_h, grid_w)
    rope2d_freq = pe_rope2d.freq
    my_rope2d_freq = build_axial_freqs(dim_head, grid_h, grid_w)
    mad = np.mean(abs(rope2d_freq.numpy() - my_rope2d_freq))

    print(f"Mean Absolute Difference of RoPE 2D Axial Frequencies: {mad}")

    # Build out input grid
    img_w, img_h = 768, 384
    patch_w, patch_h = img_w // grid_w, img_h // grid_h
    patch_embed_dim = dim // num_heads # Should be equal to head dim.
    seq_num = patch_w * patch_h
    seq = np.random.rand(1, seq_num, grid_w*grid_h)

    print(f"Grid Size: {grid_h},{grid_w}")
    print(f"Patch Size: {patch_h},{patch_w}")
    print(f"Patch Embedding Size: {patch_embed_dim}")
    print(f"Number of patches: {seq_num}")
    print(f"Axial Freq Shape: {my_rope2d_freq.shape}")
    print(f"Sequence Shape: {seq.shape}")

    # Test RoPE 2D applied to an embedding.
    rope2d_lib = tvm.compile(ImagePatchEmbedding, target="llvm")

    img = tvm.nd.array(np.random.uniform(size=(1,3,img_h,img_w)).astype("float32"))
    weights = tvm.nd.array(np.random.uniform(size=(dim_head,3,patch_h, patch_w)).astype("float32"))
    out = tvm.nd.array(np.zeros((1,dim_head,grid_h*grid_w), dtype="float32"))
    rope2d_lib(img, weights, out)
    out_pt = F.conv2d(torch.tensor(img.numpy()), torch.tensor(weights.numpy()), stride=(patch_h,patch_w))
    out_pt = out_pt.reshape(1,dim_head,grid_h*grid_w)
    print("Image Patch Embedding Loss: ", (abs(out.numpy() - out_pt.numpy())).mean())



if __name__ == '__main__':
    print("Test generating axial frequencies")
    test_freqs()