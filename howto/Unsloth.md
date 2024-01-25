```
mamba create -n unsloth python=3.11
mamba activate python
export LD_LIBRARY_PATH=/opt/cuda/lib64

# Instructions
mamba install cudatoolkit xformers bitsandbytes pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c xformers -c conda-forge -y
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

pip install triton
pip install ipython
pip install flash-attn --no-build-isolation

```