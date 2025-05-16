from typing import Union, List, Optional, Dict

import tvm.relax.frontend.nn as nn
from tvm.runtime.ndarray import NDArray

#class LayerScale(nn.Module):
#    def __init__(self, dim, init_values=1e-5, inplace=False):
#        super().__init__()
#        self.inplace = inplace
#        self.dim = dim
#        self.init_values = init_values
#
#    def forward(self, x):
#        return x
#
#    def init_tensors(self):
#        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def test(self, x: nn.Tensor):
        return x

    def forward_features(
        self,
        x: nn.Tensor,
        # layer_idx: int = -1, # We omit this from the original code because we aren't scanning through the layers; assume the model being loaded is already aligned.
        norm: bool = False,
        strip_cls_token: bool = False
    ):
        

        
    def get_default_spec(self):
        """
            This allows us to expose only certain layers to the runtime.
        """
        mod_spec = {
            "test": {
                "x": nn.spec.Tensor(["n"], "int32")
            },
            
            "forward_features": {

            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)