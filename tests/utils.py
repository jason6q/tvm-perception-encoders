"""
    Basic helper functions for testing.
"""
from typing import Tuple, Any

import tvm
import torch
import numpy as np

def get_devices(device: str = 'cuda') -> Tuple[Any, Any]:
    """
        Return a tuple with the TVM device and Torch device, useful
        when comparing between TVM and Torch modules
    """
    return (tvm.cuda(0), torch.device('cuda')) if device == 'cuda' else (tvm.cpu(0), torch.device('cpu'))

def get_tensors(arr: np.ndarray, device: str = 'cpu') -> Tuple[tvm.nd.array, np.ndarray]:
    """
        Build a tuple with the TVM array on the correct device along with the Torch array
        on the correct device.
    """
    tvm_device, pt_device = get_devices(device)
    tvm_arr = tvm.nd.array(arr, device=tvm_device)
    pt_arr = torch.from_numpy(arr).to(pt_device)
    return tvm_arr, pt_arr
    
def mad(x: np.ndarray, y: np.ndarray) -> float:
    """
        Mean Absolute Difference
    """
    return np.abs(x - y).mean()

def mse(x: np.ndarray, y: np.ndarray) -> float:
    """
        Mean Squared Error
    """
    return ((x - y)**2).mean()

def rel_diff(x: np.ndarray, y:np.ndarray) -> float:

    return

def print_diff(x: np.ndarray, y: np.ndarray):
    """
        Just print all the different metrics
    """
    print(f"Mean absolute Difference: {mad(x,y):.8f}")
    print(f"Mean Squared Error: {mad(x,y):.8f}")
    return

def find_mad_idx(A: np.ndarray, B: np.ndarray):
    """
        Sometimes it can be tricky to figure out the broadcasting logic
        between TVM code and PyTorch code. So this is primarily used
        for debugging purposes where we search for the MAD
        per element in B from A. Each element in A will be mapped to the indexing
        of B that is the MAD.

        Advised to use small matrices for this. Not perfect, you may get ambiguous mappings.
    """
    assert A.shape == B.shape, "A and B must have the same shape."
    assert len(A.shape) == 4, "Shape size must be 4"

    # A very sub-optimal implementation
    n,h,seq,dim = A.shape
    A_mins = np.zeros_like(A, dtype=np.int32)
    for ni in range(n):
        for hi in range(h):
            for seqi in range(seq):
                for dimi in range(dim):
                    diff = np.abs(B - A[ni,hi,seqi,dimi])
                    A_mins[ni,hi,seqi,dimi] = np.argmin(diff)

    return A_mins