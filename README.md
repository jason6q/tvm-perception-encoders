# TVM Perception Encoders (WIP)
Apache's TVM implementation of Meta's Perception Encoders. Most of it is written in TVScript with TensorIR. Not really using a lot of Relax methods here.

I chose to translate Perception Encoders because it's fairly recent as of the date of writing this and I read a lot of computer vision papers.

**Only supporting Spatial PE for now.**

## Preliminary Knowledge.
Before getting started it would be wise to have read the following pieces of ML literature:

0. [Perception Encoders](https://arxiv.org/pdf/2504.13181)
1. Attention Is All You Need
2. Rotary Positional Embeddings RoFormer / RoPE 2D for Vision Transformers
3. Auto-Tuning with XGBoost
4. CUDA Threads, Blocks, Grids, Warps, etc...
5. Optimization via Stochastic Scheduling
6. Operator Fusion / Decomposition
7. Operator Lowering
8. Accelerator Compilation.
9. Metal (Apple) support.
10. CLIP
11. Vision Transformers
12. [GeLU (Guassian Error Linear Units)](https://arxiv.org/pdf/1606.08415)

### Extra Knowledge
Not necessary but it may fill some small gaps when implementing the architecture.

Going Deeper with Image Transformers https://arxiv.org/pdf/2103.17239
- Useful to know if you are wondering what that LayerScale module is for. It's a regularization technique during training. The gamma parameter that is learned during training is what is scaling it.

Attention Pooling is taken from the CLIP Paper.


## The Full Translated TVM Architecture
TODO: Include DrawIO graph.


## Setup
When setting up a new conda environment; make sure you don't have any clashing variables in your system's environment. We'll also have to clone the [perception_models](https://github.com/facebookresearch/perception_models) repository to validate that our translation is correct in `tests/`

```
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models && pip install -e .

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


Set a few conda specific environment variable to be able to load the source-built TVM package.
```
conda env config vars set TVM_HOME=$(pwd)/third-party/tvm
conda env config vars set PYTHONPATH=$(pwd)/third-party/tvm/python:$PYTHONPATH
conda deactivate
conda activate tvm-perception-encoders
cd tvm/python && pip install -e .
```

Download all the weights via `huggingface_hub`
```
python download_weights.py --all
```

### Compiling CPP Code
This code is going to act as our front end to interface with the compiled model. We're doing this in CPP because typically edge-compute devices run on native code and the utility of TVM shines in these type of domains.


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

So, we'll implement each module in two ways. Leveraging the Relax NN Modules to the best of our capabilities and also a TIR version of it.

## Profiling TensorRT vs TVM CUDA
As a bonus lets do some profiling...


## Tips:
`R.call_dps_packed`
- This will call a destination passing style primitive function. Make sure to understand that the final argument in a DPS primitive function is the output buffer. It follows this convention `prim_func(in0,..., inK, out)`. **It will automatically return that out buffer!** Therefore, in your Relax function you can simply do `lv0 = R.call_dps_packed(...)`

## Custom Optimizations

## Notes & Additional Materials
Just a bunch of notes and other things used to get this going.

### Rotary Positional Embeddings
The model makes use of RoPE so we'll have to get that implemented here. Refer to the [paper](https://arxiv.org/pdf/2104.09864) for an in depth understanding of how these positional embeddings work.

### Layer Scale
Used with the Residual Attention Blocks.

### Transformer

### Vision Transformer
