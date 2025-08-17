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
    return abs(x - y).mean()

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