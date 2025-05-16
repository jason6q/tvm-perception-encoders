import dataclasses
from logging import getLogger

import torch
import tvm
import tvm.relax as R
#import tvm.nd.array as NDArray
import numpy as np

from pe import VisionTransformer
from compile import compile

logging = getLogger()

@dataclasses.dataclass
class PESpatialConfig:
    """
        Configuration parameters for PE Spatial
    """

@dataclasses.dataclass
class PEMappingTVM:
    """
        A map from torch to TVM parameters.
    """

def map_torch2tvm(state_dict: dict) -> PEMappingTVM:
    """
        Given a state_dict of a spatial PE model, map it to the correct parameter
        scheme for our TVM model.
    """
    logging.info("Extracting Pytorch weights and mapping to TVM...")
    for mod, param in state_dict.items():
        print(mod)

    return

if __name__ == '__main__':
    PE_SPATIAL = '/home/jq/Storage/Model-Weights/HuggingFace-Cache/PE-Spatial-G14-448.pt'
    LIB_PATH = 'pe_spatial.lib'

    state_dict = torch.load(PE_SPATIAL, map_location='cpu')
    device = tvm.device('cuda', 0)

    # NOTE: Will need to do this in native code as well.
    mapping: PEMappingTVM = map_torch2tvm(state_dict)
    
    # Compile model into TVM lib
    mod = VisionTransformer()
    mod, params = mod.export_tvm(spec=mod.get_default_spec())
    print(mod)

    vm = compile(mod, device)

    # Run
    x = tvm.nd.array(np.array([1,2,3,4], dtype=np.int32))
    y = vm['test'](x)
    print(y)

    # Test through TVM Runtime Virtual Machine
    #ex = tvm.runtime.load_module(LIB_PATH)
    #vm = R.VirtualMachine(ex, device)

    # Modules that ned to be supported:
    # Convultional Block
    # LN Pre
    # Positional Embeddings
    # Bunch of Res Blocks
    # In project, out project, linear layers, mlp layers, repeat.