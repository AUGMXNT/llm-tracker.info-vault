The instructions from the repo were missing some packages and had some weird issues w/ CUDA paths etc. Here's a more reliable way to get things working:
```bash
# env
mamba create -n unsloth python=3.11
mamba activate unsloth

# cuda
mamba install -c "nvidia/label/cuda-12.3.2" cuda-toolkit
conda env config vars set CUDA_PATH="$CONDA_PREFIX"
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
mamba activate unsloth
# mamba install gxx=12.3.0

# pytorch
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# bnb
pip install bitsandbytes

# fa2
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation

# rest
pip install xformers
pip install triton
pip install ipython

# unsloth
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"
```