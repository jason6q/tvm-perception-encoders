# tvm-perception-encoders
Apache's TVM implementation of Meta's Perception Encoders. Simply to showcase how to convert a new model with the TVM framework; without tracing.

Hopefully you can learn a thing or two!

## Setup

When setting up a new conda environment; make sure you don't have any clashing variables in your system's environment.

```
conda create -n tvm-perception-encoders -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11

conda activate tvm-perception-encoders
pip install -e .
```
**NOTE: TVM does not support python 3.12 so we'll use 3.11 for now**


Build the latest TVM from git; you're not going to get the latest version from PyPi.
```
chmod u+x build_tvm.sh
./build_tvm.sh
```


Set a conda specific environment variable
```
conda env config vars set TVM_HOME=$(pwd)/tvm
conda env config vars set PYTHONPATH=$(pwd)/tvm/python:$PYTHONPATH
conda deactivate
conda activate tvm-perception-encoders
cd tvm/python && pip install -e .
```

Download all the weights via `huggingface_hub`
```
python download_weights.py --all
```

## Conversion to TVM
We'll convert the model via the usage of Relax, Tensor Expression, and TIR in TVM.

The modules we'll create will be based off the official PyTorch code located here:
https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/pe.py

We'll need to implement these 6 main modules in order to be able to have a working perception encoder for spatial predictions.

1. LayerScale
2. AttentionPooling
3. SelfAttention
4. ResidualAttentionBlock
5. Transformer
6. VisionTransformer

Each of these modules may come with their own hidden set of obstacles and constraints. You would hope that the translation be seamless between PyTorch's NN module to TVM's Relax NN module; but that usually isn't the case. We will have to resort going into Tensor Expression or TIR to accomplish some of the necessary translations. Or we can go use them for fun because kernel code is fascinating in TIR to leverage some XGBoost optimizations.

### Rotary Positional Embeddings
The model makes use of RoPE so we'll have to get that implemented here. Refer to the [paper](https://arxiv.org/pdf/2104.09864) for an in depth understanding of how these positional embeddings work.

### Layer Scale
Used with the Residual Attention Blocks.

### Vision Transformer

## Custom Optimizations

## Profiling TensorRT vs TVM CUDA
As a bonus lets do some profiling...

## Compiling CPP Code
This code is going to act as our front end to interface with the compiled model. We're doing this in CPP because typically edge-compute devices run on native code and the utility of TVM shines in these type of domains.