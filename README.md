# TVM Perception Encoders (WIP)
Apache's TVM implementation of Meta's Perception Encoders. Most of it is written in TVMScript with TensorIR. Not really using a lot of Relax methods here.

I chose to translate Perception Encoders because it's fairly recent as of the date of writing this and I read a lot of computer vision papers.

**Only supporting Spatial PE for now.**

## Preliminary Knowledge.
Before getting started it would be wise to have read the following pieces of ML literature:

### Model Architecture
Papers to read related to the architecture of the model.
- [Perception Encoders](https://arxiv.org/pdf/2504.13181)
    - The main architecture we will be implementing
- [GeLU (Guassian Error Linear Units)](https://arxiv.org/pdf/1606.08415)
    - Main activation function used throughout.
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
    - Multi-head Self Attention
- [RoFormer: Enhanced Transformer With Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
    - One Dimensional Case for RoPE
- [Rotary Positional Embedding for Vision Transformer](https://arxiv.org/pdf/2403.13298)
    - Two Dimensional Case for RoPE
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
    - Pre-training method for embeddings using image and caption pairs.
- [An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/pdf/2010.11929)
    - Vision Transformer
- [Attention Pooling]()
    - Pool
- [Going Deeper with Image Transformers](https://arxiv.org/pdf/2103.17239)
    - Layer Scale Regularization during training.

### Optimization
Some optimization techniques to know what's happening under the hood with TVM.

- [Scheduling with TVM]()
    - Automatically via meta-schedule like using XGBoost to search kernel space.
    - Stochastic scheduling to random search kernel space

- [Flash Attention](https://arxiv.org/pdf/2205.14135)
    - Look into the [Online Normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- CUDA Threads, Blocks, Grids, Warps, etc...
- Operator Fusion / Decomposition
- Operator Lowering
- Accelerator Compilation.
- Metal (Apple) support.

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