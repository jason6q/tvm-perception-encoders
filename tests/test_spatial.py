import torch

def test_spatial(state_dict: dict):
    for module_name, param in state_dict.items():
        print(module_name)

    return

if __name__ == '__main__':
    PE_SPATIAL = '/home/jq/Storage/Model-Weights/HuggingFace-Cache/PE-Spatial-G14-448.pt'

    state_dict = torch.load(PE_SPATIAL, map_location='cpu')

    test_spatial(state_dict)