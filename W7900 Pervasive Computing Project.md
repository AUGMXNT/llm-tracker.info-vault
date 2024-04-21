In January 2024 I applied for the [Hackster.io AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) and a W7900 card was delivered on 2024-04-19.
- Application: https://www.hackster.io/contests/amd2023/hardware_applications/16885
- Project: https://www.hackster.io/lhl/ultra-low-latency-local-voice-assistant-avatar-4c48f2
- Repo:

I will be keeping a log here for now...


# 2024-04-20

## Flash Attention - Latest Not Working
Last update 2024-04-08 - FA 2.5.5 being worked on internally
- https://github.com/ROCm/flash-attention/issues/35#issuecomment-2042391285
- Currently  gfx1100 not supported

2.0.4 Forward Pass only
```
git clone https://github.com/ROCm/flash-attention
cd flash-attention
git fetch
git branch -a
git checkout howiejay/navi_support
python setup.py install
python -c "import flash_attn; print(flash_attn.__version__)"
```
- howiejay is no longer working on this project: https://github.com/Dao-AILab/flash-attention/issues/707#issuecomment-2049042957
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

## bitsandbytes - Works
ROCM fork works (0.44.0.dev0)
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -S .
make
pip install .
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

## ExLlamaV2 - works

Inference Speed:
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

...

 ** Position  8064 + 128 tokens:    4.5124 t/s
```
* 39GiB VRAM usages at 4096 tokens
* Insanely long (46min lol) load times on machines w/ 16GiB RAM - 30s w/ 64GiB of RAM
* `GPU_MAX_HW_QUEUES=1` not required w/ fast loading?

w/ FA2 2.0.4, no difference in perf
```
 ** Position  3968 + 128 tokens:    6.9836 t/s
```

Prompt Processing Speed:
```
$ GPU_MAX_HW_QUEUES=1 python test_inference.py -m /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/ -ps
Successfully preprocessed all matching files.
 -- Model: /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/
 -- Options: []
 -- Loading model...
 -- Loaded model in 3402.6222 seconds
 -- Loading tokenizer...
 -- Warmup...
 -- Measuring prompt speed...
 ** Length   128 tokens:    154.0550 t/s
 ** Length   256 tokens:    269.5589 t/s
 ** Length   384 tokens:    358.5119 t/s
 ** Length   512 tokens:    359.8361 t/s
 ** Length   640 tokens:    365.1964 t/s
 ** Length   768 tokens:    429.5664 t/s
 ** Length   896 tokens:    426.6023 t/s
 ** Length  1024 tokens:    430.6259 t/s
 ** Length  2048 tokens:    416.8521 t/s
 ** Length  3072 tokens:    394.7572 t/s
 ** Length  4096 tokens:    363.3365 t/s
 ** Length  8192 tokens:    283.3092 t/s
```


## llama.cpp - works
```
git clone https://github.com/ggerganov/llama.cpp/
cd llama.cpp
make LLAMA_HIPBLAS=1

$ ./llama-bench -m Meta-Llama-3-70B-Q4_K_M.gguf -p 3968
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon PRO W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 | pp 3968    |    255.59 ± 0.94 |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 | tg 128     |     11.34 ± 0.01 |

build: b8109bc0 (2701)
```
- llama3 template/stop tokens still in progress: https://github.com/ggerganov/llama.cpp/issues/6747
- https://github.com/ggerganov/llama.cpp/pull/6745
- https://github.com/ggerganov/llama.cpp/pull/6751