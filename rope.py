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
#import tvm
#import tvm.relax.frontend.nn as nn


def build_axial_freqs(
    head_dim: int, 
    grid_height: int, # The grid of patches
    grid_width: int,
    num_freqs=1,
    max_freq=10 # Not sure how to determine this value...
    ):
    """
        Since we aren't using learned rotations we can build out the frequencies
        in advance and store them as TVM Arrays. This will be sent during
        the forward pass of our entire model.

        TODO: Cache in the future?
    """
    dim = head_dim // 2 

    x_pos = np.arange(grid_width)
    y_pos = np.arange(grid_height)
    print(f"x_pos: {x_pos}")
    print(f"y_pos: {y_pos}")

    # Each contiguous pair of elements in our embedding will be matched
    # to a frequency value, theta_t in [1,max_freq / 2] * pi
    freqs = np.linspace(1.0, max_freq / 2, dim) * math.pi
    print(freqs)

    # Recollect that we are using Euler's formula and its equivalence to a rotation matrix.
    # We'll have to build out the rotation matrices for each pair of elements in the embedding
    # all in a single matrix later on.
    freqs = np.einsum("..., f -> ... f", x_pos.astype(freqs.dtype), freqs)
    print(freqs)
    freqs = repeat(freqs, "... n -> ... (n r)", r=2)
    print(freqs)
    print(freqs.shape)

    return

def apply_rope():
    return

if __name__ == '__main__':
    build_axial_freqs(10, 10, 10)