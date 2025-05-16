"""
  Just use Hugging Face to load the weights.
"""
from pathlib import Path

import numpy as np
#import fire
from huggingface_hub import hf_hub_download

PE_CORE_MODELS = ['PE-Core-B16-224', 'PE-Core-L14-336', 'PE-Core-G14-448']
PE_LANG_MODELS = ['PE-Lang-L14-448', 'PE-Lang-G14-448']
PE_SPATIAL_MODELS = ['PE-Spatial-G14-448']
HF_CACHE='/home/jq/Storage/Model-Weights/HuggingFace-Cache'

def download_weights(name='all'):
    if name == 'all':
        for hf_name in PE_CORE_MODELS + PE_LANG_MODELS + PE_SPATIAL_MODELS:
            repo_id = f"facebook/{hf_name}"
            hf_hub_download(repo_id=repo_id, filename="config.yaml", local_dir=HF_CACHE)
            return hf_hub_download(repo_id=repo_id, filename=f"{hf_name}.pt", local_dir=HF_CACHE)
    elif hf_name in PE_CORE_MODELS + PE_LANG_MODELS + PE_SPATIAL_MODELS:
            repo_id = f"facebook/{hf_name}"
            hf_hub_download(repo_id=repo_id, filename="config.yaml", local_dir=HF_CACHE)
            return hf_hub_download(repo_id=repo_id, filename=f"{hf_name}.pt", local_dir=HF_CACHE)

    raise Exception("Invalid name specified.")
    

if __name__ == '__main__':
    download_weights()
