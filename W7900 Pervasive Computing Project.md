In January 2024 I applied for the [Hackster.io AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) and a W7900 card was delivered on 2024-04-19.
- Application: https://www.hackster.io/contests/amd2023/hardware_applications/16885
- Project: https://www.hackster.io/lhl/ultra-low-latency-local-voice-assistant-avatar-4c48f2
- Repo:

I will be keeping a log here for now...


# 2024-04-20
All tests on an Ubuntu 22.04 LTS HWE box w/ ROCm native install:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html

## PyTorch - Works
https://pytorch.org/get-started/locally/
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0
```
- ROCm 6.0 requires nightly
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
* Only 1% difference w/ FA2 2.0.4-rocm-howiejay
* About 3min to load 70B model (132GiB), 40GiB memory,  3.3 tok/s bs=1 inference speed
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

Just for a baseline comparison, the W7900 is about 15% slower in prefill and 20% slower in generation than a 7900 XTX (lower TDP, slower clocks and memory?)
```
$ ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon PRO W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2193.89 ± 3.09 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |     93.94 ± 0.18 |

build: 784e11de (2725)
```
- For comparison numbers, see: https://llm-tracker.info/howto/AMD-GPUs#llamacpp

## MLC - works
Install:
```
# Env
mamba create -n mlc python=3.11

# Pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

# Install
# https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm57 mlc-ai-nightly-rocm57
python -c "import mlc_llm; print(mlc_llm)"

# Required otherwise errors
mamba install conda-forge::lld

mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```
- https://github.com/mlc-ai/relax/issues/316
- https://github.com/mlc-ai/mlc-llm/issues/2144
- see also: https://github.com/mlc-ai/mlc-llm/issues/2160

Convert model
See: https://llm.mlc.ai/docs/compilation/convert_weights.html
```
mkdir dist

# Convert - takes about 10min for 70B
mlc_llm convert_weight /models/hf/NousResearch_Meta-Llama-3-70B/ --quantization q4f16_1 -o dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC

mlc_llm gen_config /models/hf/NousResearch_Meta-Llama-3-70B/ --quantization q4f16_1 --conv-template llama-3 -o dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/

mlc_llm chat dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/


mlc_llm bench dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/ --generate-length 4096

[2024-04-21 13:19:32] INFO model_metadata.py:96: Total memory usage: 40345.77 MB (Parameters: 37849.77 MB. KVCache: 0.00 MB. Temporary buffer: 2496.00 MB)
[2024-04-21 13:19:32] INFO model_metadata.py:105: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`

Statistics:
----------- prefill -----------
throughput: 36.910 tok/s
total tokens: 7 tok
total time: 0.190 s
------------ decode ------------
throughput: 12.041 tok/s
total tokens: 4096 tok
total time: 340.169 s

# --prompt "000...."

Statistics:
----------- prefill -----------
throughput: 95.501 tok/s
total tokens: 3376 tok
total time: 35.351 s
------------ decode ------------
throughput: 10.686 tok/s
total tokens: 128 tok
total time: 11.979 s
```
- 42.8GiB memory usage
- llama.cpp has about the same inference speed, 2.5X prompt processing
- exllama has 50% slower inference speed, but 4X prompt processing

## Whisper


## StyleTTS2 - works
```
python -c "import nltk; nltk.download('punkt')"

RTF = 0.306594
```