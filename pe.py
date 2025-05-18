"""
    The main modules to create the PE Model.
    We'll include a spec for every module for debugging purposes. You should
    be able to load the packed params independently for each module and test the output
    against the original pytorch model.
"""
from typing import Union, List, Optional, Dict

import tvm.relax.frontend.nn as nn
from tvm.runtime.ndarray import NDArray

def TIRRoPE():
    """
        We can just calculate RoPE via TIR. Not a difficult function.
    """
    return

class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1e-5, inplace: bool =False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

        self.gamma = nn.Parameter([self.dim], dtype="float32")

    def forward(self, x):
        # We'll need the learned gamma parameter
        return x * self.gamma

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "x": nn.spec.Tensor([self.dim], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none"
                }
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

class TIRLayerScale(nn.Module):
    def __init__(self):
        pass

class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

class SelfAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int,):
        super().__init__()

    def forward(self, x: nn.Tensor):
        return

    def get_default_spec(self):
        mod_spec = {
            "forward": {
                "x": nn.spec.Tensor([self.dim], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none"
                }
            }
        }

        return nn.spec.Modulespec.from_raw(mod_spec, self)

class ResidualAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def test(self, x: nn.Tensor):
        return x

    def forward_features(
        self,
        x: nn.Tensor,
        # layer_idx: int = -1, # We omit this from the original code because we aren't scanning through the layers; 
        # assume the model being loaded is already aligned to the down-stream task.
        norm: bool = False,
        strip_cls_token: bool = False
    ):
        return x

        
    def get_default_spec(self):
        """
            This allows us to expose only certain layers to the runtime.
        """
        mod_spec = {
            "test": {
                "x": nn.spec.Tensor(["n"], "int32"),
                "norm": bool,
                "strip_cls_token": bool
            },
            
            "forward_features": {
                "x": nn.spec.Tensor(["n"], "int32"),
            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)