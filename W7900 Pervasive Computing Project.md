In January 2024 I applied for the [Hackster.io AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) and a W7900 card was delivered on 2024-04-19.
- Application: https://www.hackster.io/contests/amd2023/hardware_applications/16885
- Project: https://www.hackster.io/lhl/ultra-low-latency-local-voice-assistant-avatar-4c48f2
- Repo:

I will be keeping a log here for now...


# 2024-04-20

## Flash Attention - Not Working
Last update 2024-04-08 - FA 2.5.5 being worked on internally
- https://github.com/ROCm/flash-attention/issues/35#issuecomment-2042391285

## xformers - Not Working
Neither the upstream or AMD's ROCm fork compile:
See: https://github.com/facebookresearch/xformers/issues/1026

We need to compile from code
```
# Make sure we have the ROCm version of PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

# You can double check
python -c "import torch; print(torch.version.hip)"

# Install from source - on a Ryzen 5600G takes ~
pip install ninja
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

pip wheel -v --no-build-isolation git+https://github.com/ROCm/xformers.git@main#egg=xformers


# Double check
python -m xformers.info
```

## vllm - Not Working
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

git clone https://github.com/vllm-project/vllm.git
cd vllm
sudo docker build --build-arg BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1" --build-arg FX_GFX_ARCHS="gfx1100" --build-arg BUILD_FA=0 -f Dockerfile.rocm -t vllm-rocm .

# pip install -r requirements-rocm.txt
# VLLM_TARGET_DEVICE=rocm pip install -e .
```

## ExLlamaV2
```
$ GPU_MAX_HW_QUEUES=1 python test_inference.py -m /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/ -s
 -- Model: /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/
 -- Options: []
 -- Loading model...
 -- Loaded model in 2802.5172 seconds
 -- Loading tokenizer...
 -- Measuring token speed...
...

 ** Position  3968 + 128 tokens:    7.0301 t/s

```
* 39GiB VRAM usages at 4096 tokens
* Insanely long (46min lol) load times on machines w/ 16GiB RAM