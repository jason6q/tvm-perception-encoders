import sys
sys.path.append('..')

import torch
import numpy as np

import core.vision_encoder.rope as pe_rope
from rope import build_axial_freqs, apply_rope

from compile import compile
from utils import get_devices, get_tensors, print_diff

def test_freqs():
    dim = 256
    num_heads=4
    grid_w, grid_h = 32,16
    dim_head = dim // num_heads

    print(f"Head Dimension: {dim_head}")
    pe_rope2d = pe_rope.Rope2D(dim=dim_head)
    device = torch.device('cpu')

    pe_rope2d.init_tensors()
    pe_rope2d.update_grid(device, grid_h, grid_w)
    rope2d_freq = pe_rope2d.freq
    my_rope2d_freq = build_axial_freqs(dim_head, grid_h, grid_w)
    mad = np.mean(abs(rope2d_freq.numpy() - my_rope2d_freq))

    print(f"Mean Absolute Difference of RoPE 2D Implementations: {mad}")

def test_rope():
    dim = 256
    num_heads=4
    grid_w, grid_h = 32,16
    dim_head = dim // num_heads

    compile(apply_rope, device)


if __name__ == '__main__':
    print("Test generating axial frequencies")
    test_freqs()

    print("Test applying RoPE2D to an embedding.")