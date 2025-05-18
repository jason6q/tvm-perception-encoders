import dataclasses
from logging import getLogger

import torch

logging = getLogger()

@dataclasses.dataclass
class PESpatialConfig:
    """
        Configuration parameters for PE Spatial
    """

@dataclasses.dataclass
class PEMappingTVM:
    """
        A map from torch to TVM modules.
    """

    
def select_params(mod_name: str, state_dict: dict) -> torch.Tensor :
    """
        Helper function to select a specific weight in the PE Pytorch Model Parameters
    """
    assert mod_name in state_dict.keys(), f"{mod_name} does not exist."
    return state_dict[mod_name]

def map_torch2tvm(state_dict: dict) -> PEMappingTVM:
    """
        Given a state_dict of a spatial PE model, map it to the correct parameter
        scheme for our TVM model.
    """
    logging.info("Extracting Pytorch weights and mapping to TVM...")
    for mod, param in state_dict.items():
        print(mod)

    return