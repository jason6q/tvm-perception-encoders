import torch

def load_pe_spatial(
    pt_weights: str = '/home/jq/Storage/Model-Weights/HuggingFace-Cache/PE-Spatial-G14-448.pt',
    device = 'cpu'
    ):
    """
        Load all the PE Spatial Weights

        Weights for Residual Block:
        1. attn.in_proj [3*WIDTH, WIDTH]
        2. attn.out_proj [WIDTH, WIDTH]
        3. ln_1, ln_2 [WIDTH]
        4. ls_1, ls_2 [WIDTH]
        5. mlp_c_fc [MLP_WIDTH,WIDTH]
        6. mlp_c_proj [WIDTH, MLP_WIDTH]

        Where WIDTH is the dimension of the model and MLP_WIDTH is WIDTH*MLP_RATIO, 
        MLP_RATIO=8960 / 1536
    """
        
    state_dict = torch.load(pt_weights, map_location=device)
    for mod, param in state_dict.items():
        print(mod, param.shape)

    return

if __name__ == '__main__':
    load_pe_spatial()