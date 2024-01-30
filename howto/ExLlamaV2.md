```
# Environment
mamba env remove --name exllamav2
mamba create -n exllamav2 python=3.11 -y
mamba activate exllamav2

# CUDA
mamba install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
mamba install gxx=12.2 ninja cmake -y
conda env config vars set CUDA_PATH="$CONDA_PREFIX"
conda env config vars set CUDA_HOME="$CONDA_PREFIX"

mamba activate exllamav2

# PyTorch
pip install torch torchvision torchaudio

# Flash Attention 2
pip install flash-attn --no-build-isolation

# ExLLamaV2
git clone https://github.com/turboderp/exllamav2
pip install -r requirements.txt
python setup.py install
```

If you have `exllamav2_ext` issues try:
```
rm -rf ~/.cache/torch_extensions
```