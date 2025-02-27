
# llama.cpp Setup

Install the oneAPI Base Toolkit:
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
- I use Arch Linux on most of my dev systems, and the AUR is out of date, so you should download directly, install (it won't deal with any dependencies so you will need to make sure you have them installed yourself) - it installs into `~/intel/oneapi`

To use SYCL/IPEX-LLM, etc, you will need to load env vars, eg:
```
source ~/intel/oneapi/setvars.sh
```
- Requires a POSIX compatible shell to run (eg, fish won't work)

If you have multiple oneAPI Base Toolkit versions installed (IPEX-LLM for example requires 2024.2.1, which is not the latest version):
```
source ~/intel/oneapi/2024.2/oneapi-vars.sh 
```

# PyTorch Setup

## Install the Intel Support Packages
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/884eaa22-d56f-45dc-9a65-901f1c625f9e/l_intel-for-pytorch-gpu-dev_p_0.5.3.36_offline.sh
sh ./l_intel-for-pytorch-gpu-dev_p_0.5.3.36_offline.sh
```
- https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html#inpage-nav-2
- [Option 2D: Install Using Offline Installation Scripts](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html#collapseCollapsible1730140914547)

## Setup env vars:
```
# if installed as root
source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh

# else
source ~/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh
```

Note: If you load the latest [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) (2025.0.1 in Dec 2024), the basic PyTorch inference and training example work *except* for those using `torch.compile`, which fail. I have to assum that the separate PyTorch specific support package gets folded into the base toolkit eventually, but who knows.
## Install PyTorch XPU
```
mamba create -n pytorch python=3.12
mamba activate pytorch

# Make sure we install 2.5.x to be compatible w/ the pytorch-prereqs (otherwise it moves up to 2.6 and will cause problems)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/test/xpu

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
```
- https://pytorch.org/docs/stable/notes/get_start_xpu.html
- https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html

OK, that should be it. See also: https://www.reddit.com/r/LocalLLaMA/comments/1hfrdos/comment/m2e9nd5/

# vLLM Setup
If we already have PyTorch setup...

## Docker (FAILS)
```
# paru -S docker docker-compose docker-buildx
# systemctl enable --now docker

❯ sudo docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .                                                       (vllm) 

[+] Building 273.0s (14/17)                                                                                            docker:default
 => [internal] load build definition from Dockerfile.xpu                                                                         0.0s
 => => transferring dockerfile: 2.69kB                                                                                           0.0s
 => WARN: LegacyKeyValueFormat: "ENV key=value" should be used instead of legacy "ENV key value" format (line 65)                0.0s
 => [internal] load metadata for docker.io/intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04                                     0.0s
 => [internal] load .dockerignore                                                                                                0.0s
 => => transferring context: 387B                                                                                                0.0s
 => [vllm-base  1/11] FROM docker.io/intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04                                           0.2s
 => [internal] load build context                                                                                                0.5s
 => => transferring context: 48.03MB                                                                                             0.4s
 => [vllm-base  2/11] RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor  2.4s
 => [vllm-base  3/11] RUN apt-get update -y &&     apt-get install -y --no-install-recommends --fix-missing     curl     ffmpe  51.9s
 => [vllm-base  4/11] WORKDIR /workspace/vllm                                                                                    0.0s 
 => [vllm-base  5/11] COPY requirements-xpu.txt /workspace/vllm/requirements-xpu.txt                                             0.0s 
 => [vllm-base  6/11] COPY requirements-common.txt /workspace/vllm/requirements-common.txt                                       0.0s 
 => [vllm-base  7/11] RUN --mount=type=cache,target=/root/.cache/pip     pip install --no-cache-dir     -r requirements-xpu.t  173.5s 
 => [vllm-base  8/11] RUN git clone https://github.com/intel/pti-gpu &&     cd pti-gpu/sdk &&     git checkout 6c491f07a777ed8  44.2s 
 => [vllm-base  9/11] COPY . .                                                                                                   0.3s 
 => ERROR [vllm-base 10/11] RUN --mount=type=bind,source=.git,target=.git     if [ "$GIT_REPO_CHECK" != 0 ]; then bash tools/ch  0.3s 
------                                                                                                                                
 > [vllm-base 10/11] RUN --mount=type=bind,source=.git,target=.git     if [ "$GIT_REPO_CHECK" != 0 ]; then bash tools/check_repo.sh; fi:                                                                                                                                    
0.286 Repo is dirty                                                                                                                   
------

 1 warning found (use docker --debug to expand):
 - LegacyKeyValueFormat: "ENV key=value" should be used instead of legacy "ENV key value" format (line 65)
Dockerfile.xpu:48
--------------------
  47 |     ARG GIT_REPO_CHECK
  48 | >>> RUN --mount=type=bind,source=.git,target=.git \
  49 | >>>     if [ "$GIT_REPO_CHECK" != 0 ]; then bash tools/check_repo.sh; fi
  50 |     
--------------------
ERROR: failed to solve: process "/bin/sh -c if [ \"$GIT_REPO_CHECK\" != 0 ]; then bash tools/check_repo.sh; fi" did not complete successfully: exit code: 1
```
- https://docs.vllm.ai/en/latest/getting_started/xpu-installation.html#quick-start-using-dockerfile

## Source (FAILS)
Build from source with some coaxing but fails running
```
# env
mamba create --name vllm --clone pytorch
mamba activate vllm

# don't mess w/ our pytorch
python use_existing_torch.py
pip install -r requirements-common.txt
pip install -r requirements-build.txt
pip install -v -r requirements-xpu.txt

VLLM_TARGET_DEVICE=xpu python setup.py install

# hmm
pip install --no-cache-dir aiohttp
```

# Writeups
## Testing llama.cpp with Intel's Xe2 iGPU (Core Ultra 7 258V w/ Arc Graphics 140V)
https://www.reddit.com/r/LocalLLaMA/comments/1gheslj/testing_llamacpp_with_intels_xe2_igpu_core_ultra/

I have a Lunar Lake laptop (see my [in-progress Linux review](https://github.com/lhl/linuxlaptops/wiki/2024-MSI-Prestige-13-AI--Evo-A2VM)) and recently sat down and did some testing on how llama.cpp works with it.
- Chips and Cheese has the [most in-depth analysis of the iGPU](https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels) which includes architectural and real world comparisons w/ the prior-gen Xe-LPG, as well as RDNA 3.5 (in the AMD Ryzen AI 9 HX 370 w/ Radeon 890M).
- The 258V has 32GB of LPDDR5-8533, which has a theoretical maximum memory bandwidth of  136.5 GB/s. Chips and Chesee did some [preliminary MBW testing](https://chipsandcheese.com/i/149978169/cache-and-memory-bandwidth) and found actual throughput to be around 80 GB/s (lower than Strix Point), but MBW test is hard... 
- The 140V Xe2 GPU on the 258V has Vector Engines with 2048-bit XMX units that Intel specs at 64 INT8 TOPS. Each XMX can do INT8 4096 OPS/clock or FP16 2048 OPS/clock, so that would be a max theoretical 32 FP16 TOPS.

For my testing, I use Llama 2 7B  (specifically the q4_0 quant from [TheBloke/Llama-2-7B-GGUF]) as my standard benchmark (it is well quantified and has max compatibility). All testing was done with very-up-to-date HEAD compiles (`build: ba6f62eb (4008)`) of llama.cpp. The system itself is running [CachyOS](https://cachyos.org/), a performance focused Arch Linux derivative, and it is running the latest 6.12 kernel `6.12.0-rc5-1-mainline` and `linux-firmware-git` and `mesa-git` for the maximum support for Lunar Lake/Xe2.

My system is running at PL 28W (BIOS: performance), with the performance governor, EPP, and EPB.

It turns out there are quite a few ways to run llama.cpp - I skipped the NPU since it's a PITA to setup, but maybe I'll get bored sometime. Here's my results:

| Backend                                                                                                          | pp512 t/s | tg128 t/s | t/TFLOP | MBW % |
| ---------------------------------------------------------------------------------------------------------------- | --------: | --------: | ------: | ----: |
| [CPU](https://github.com/ggerganov/llama.cpp/)                                                                   |     25.05 |     11.59 |   52.74 | 30.23 |
| [Vulkan](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan)                                |     44.65 |      5.54 |    1.40 | 14.45 |
| [SYCL FP32](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md)                             |    180.77 |     14.39 |    5.65 | 37.53 |
| [SYCL FP16](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md)                             |    526.38 |     13.51 |   16.45 | 35.23 |
| [IPEX-LLM](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md) |    708.15 |     24.35 |   22.13 | 63.51 |
- pp is prompt processing (also known as prefill, or input) - this is the speed at which any system prompt, context, previous conversation turns, etc are passed in and is compute bound
- tg is token generation (aka output) - this is the speed at which new tokens are generated and is generally memory bandwidth bound
- I've included a "t/TFLOP" compute efficiency metric for each Backend and also a MBW % which just calculates the percentage of the tg vs the theoretical max tg (136.5 GB/s / 3.56GB model size)
- The CPU backend doesn't have native FP16. TFLOPS is calculated based on the maximum FP32 that AVX2 provides for the 4 P-Cores (486.4 GFLOPS) at 3.8GHz (my actual all-core max clock). For those interested on llama.cpp's CPU optimizations, I recommend reading jart's writeup [LLaMA Now Goes Faster on CPUs](https://justine.lol/matmul/)
- For CPU, I use `-t 4`, which uses all 4 of the (non-hyperthreaded) P-cores, which is the most efficient setting. This basically doesn't matter for the rest of the GPU methods.

For SYCL and IPEX-LLM you will need to install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). I used version 2025.0.0 for SYCL, but IPEX-LLM's llama.cpp requires 2024.2.1
- Setup docs to [Run llama.cpp with IPEX-LLM on Intel GPU](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md) - as of testing, the llama.cpp was based off of a 2024-08-22 version

The IPEX-LLM results are much better than all the other Backends, but it's worth noting that despite the docs suggesting otherwise, with the Xe2 Arc 140V GPU atm, it **doesn't** seem to work with k-quants ([related to this error?](https://github.com/intel-analytics/ipex-llm/issues/11080)). Still, at 35% faster pp and 80% faster tg than SYCL FP16, it's probably worth trying to use this if you can.

## vs Apple M4
I haven't seen any M4 inference numbers, yet, but this chart/discussion [Performance of llama.cpp on Apple Silicon M-series #4167](https://github.com/ggerganov/llama.cpp/discussions/4167) is a good reference. The M3 Pro (18 CU) has [12.78 FP16 TFLOPS](https://www.cpu-monkey.com/en/igpu-apple_m3_pro_18_core) and at 341.67 t/s pp, that gives a ~26.73 t/TFLOP for Metal performance. The new M4 Pro (20 CU) has an [expected 17.04 TFLOPS](https://www.cpu-monkey.com/en/igpu-apple_m4_pro_20_core) so at the same efficiency you'd expect ~455 t/s for pp. For MBW, we can again run similar back-calculations. The M3 Pro has 150 GB/s MBW and generates 30.74 t/s tg for a 73% MBW efficiency. at 273 GB/s of MBW, we'd expect the M4 Pro to have a ballpark tg of ~56 t/s.

## vs AMD Ryzen AI
The [Radeon 890M](https://www.techpowerup.com/gpu-specs/radeon-890m.c4224) on the top-end Ryzen AI Strix Point chips have 16CUs and a [theoretical 23.76 TFLOPS](https://gpuspecs.com/theoretical-performance-calculator), and with LPDDR5-7500, 120GB/s of MBW. Recently AMD just published an article [Accelerating Llama.cpp Performance in Consumer LLM Applications with AMD Ryzen™ AI 300 Series](https://community.amd.com/t5/ai/accelerating-llama-cpp-performance-in-consumer-llm-applications/ba-p/720311) testing the performance of a Ryzen AI 9 HX 375 with a Intel Core Ultra 7 258V. It mostly focuses on CPU and they similarly note that llama.cpp's Vulkan backend works awfully on the Intel side, so they claim to compare Mistral 7B 0.3 performance w/ IPEX-LLM, however they don't publish any actual performance numbers, just a percentage difference!

Now, I don't have a Strix Point chip, but I do have a 7940HS with a Radeon 780M (16.59 TFLOPS) and dual channel DDR-5600 (89.6 GB/s MBW) so I ran the same benchhmark on a Mistral 7B 0.3 (q4_0) and did do some ballpark estimates:

| Hardware         | Backend | TFLOPS | MBW (GB/s) | pp512 t/s | tg128 t/s | t/TFLOP | MBW % |
|------------------|---------|--------:|-----------:|-----------:|----------:|--------:|------:|
| Arc 140V         | IPEX-LLM| 32      | 136.5      | 656.5      | 22.98     | 20.52   | 64.48 |
| Radeon 780M      | ROCm    | 16.59   | 89.6       | 240.79     | 18.61     | 14.51   | 79.55 |
| Radeon 890M (est)| ROCm    | 23.76   | 120        | 344.76     | 24.92     | 14.51   | 79.55 |
| Strix Halo (est) | ROCm    | 59.40   | 256        | 861.89     | 53.17     | 14.51   | 79.55 |


I just applied the same efficiency from the 780M results onto the 890M specs to get a projected performance number.

## Efficiency Numbers
Writeup here: https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/

https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF
https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true



https://github.com/ggerganov/llama.cpp/discussions/4167

https://community.amd.com/t5/ai/accelerating-llama-cpp-performance-in-consumer-llm-applications/ba-p/720311

https://github.com/intel/intel-npu-acceleration-library

-v 
--progress
## CPU
```
❯ ./llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf -t 4                                                                      (base) 
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |       4 |         pp512 |         25.05 ± 0.04 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |       4 |         tg128 |         11.59 ± 0.40 |

build: ba6f62eb (4008)




❯ ./llama-bench -t 4 -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf                                                      (base) 
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CPU        |       4 |         pp512 |         21.90 ± 0.02 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CPU        |       4 |         tg128 |         12.36 ± 0.43 |

build: ba6f62eb (4008)

❯ ./llama-bench -t 8 -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf                                                      (base) 
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CPU        |       8 |         pp512 |         30.35 ± 0.68 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CPU        |       8 |         tg128 |         11.76 ± 0.26 |

build: ba6f62eb (4008)

```
## Vulkan
```
❯ ./llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf                                                                           (base) 
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: Intel(R) Graphics (LNL) (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | warp size: 32
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         pp512 |         44.65 ± 0.27 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         tg128 |          5.54 ± 0.01 |

build: ba6f62eb (4008)



```


## SYCL FP16
```
source ~/intel/oneapi/setvars.sh

$ rm -rf build
$ cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
$ cmake --build build --config Release -j -v

build/bin/llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf -t 4

$ build/bin/llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: yes
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |       4 |         pp512 |        526.38 ± 1.74 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |       4 |         tg128 |         13.51 ± 0.06 |

$ build/bin/llama-bench -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: yes
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | SYCL       |  99 |       4 |         pp512 |        442.29 ± 1.83 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | SYCL       |  99 |       4 |         tg128 |         12.47 ± 0.21 |

build: ba6f62eb (4008)

$ build/bin/llama-bench -m ~/ai/models/gguf/Mistral-Nemo-Instruct-2407.Q4_K_M.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: yes
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | SYCL       |  99 |       4 |         pp512 |        253.21 ± 0.75 |
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | SYCL       |  99 |       4 |         tg128 |          7.95 ± 0.02 |

build: ba6f62eb (4008)
```


## SYCL FP32
```
rm -rf build
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j -v

$ build/bin/llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |       4 |         pp512 |        180.77 ± 0.26 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |       4 |         tg128 |         14.39 ± 0.08 |

$ build/bin/llama-bench -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | SYCL       |  99 |       4 |         pp512 |        208.20 ± 0.52 |

Broadcast message from gdm@p13 on tty1 (Sat 2024-11-02 00:54:17 JST):

The system will suspend now!

| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | SYCL       |  99 |       4 |         tg128 |         12.40 ± 0.04 |

build: 0a683e80 (3998)

$ build/bin/llama-bench -m ~/ai/models/gguf/Mistral-Nemo-Instruct-2407.Q4_K_M.gguf -t 4
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------------: | -------------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|   20.4|     64|    1024|   32| 15064M|            1.6.31294|
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | SYCL       |  99 |       4 |         pp512 |        129.65 ± 0.17 |
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | SYCL       |  99 |       4 |         tg128 |          7.88 ± 0.03 |
```

IPEX-LLM
```
source ~/intel/oneapi/2024.2/oneapi-vars.sh 

$ ./llama-bench -m ~/ai/models/gguf/llama-2-7b.Q4_0.gguf 
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|    1.6|     64|    1024|   32| 15064M|            1.3.31294|
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |         pp512 |    630.16 ± 2.45 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | SYCL       |  99 |         tg128 |     19.22 ± 0.03 |

build: 1d5f8dd (1)


$ ./llama-bench -m ~/ai/models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf 
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|    1.6|     64|    1024|   32| 15064M|            1.3.31294|
Sub-group size 8 is not supported on the device
Exception caught at file:/home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl.cpp, line:3164, func:operator()
SYCL error: CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)): Meet error in this line code!
  in function ggml_sycl_op_mul_mat at /home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl.cpp:3164
/home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl/common.hpp:103: SYCL error
libggml.so(+0x79517) [0x7ff60ba79517]
libggml.so(ggml_abort+0xd8) [0x7ff60ba794a8]
libggml.so(+0x1f2e98) [0x7ff60bbf2e98]
libggml.so(+0x229198) [0x7ff60bc29198]
libggml.so(_Z25ggml_sycl_compute_forwardR25ggml_backend_sycl_contextP11ggml_tensor+0x5ef) [0x7ff60bbf57ef]
libggml.so(+0x24033f) [0x7ff60bc4033f]
libggml.so(ggml_backend_sched_graph_compute_async+0x548) [0x7ff60bae60f8]
libllama.so(llama_decode+0xbc7) [0x7ff60d273d47]
./llama-bench() [0x41a580]
./llama-bench() [0x416ccd]
/usr/lib/libc.so.6(+0x261ce) [0x7ff60d5d81ce]
/usr/lib/libc.so.6(__libc_start_main+0x8a) [0x7ff60d5d828a]
./llama-bench() [0x41565e]
Aborted (core dumped)

$ ./llama-bench -m ~/ai/models/gguf/Mistral-Nemo-Instruct-2407.Q4_K_M.gguf 
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Graphics [0x64a0]|    1.6|     64|    1024|   32| 15064M|            1.3.31294|
Sub-group size 8 is not supported on the device
Exception caught at file:/home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl.cpp, line:3164, func:operator()
SYCL error: CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)): Meet error in this line code!
  in function ggml_sycl_op_mul_mat at /home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl.cpp:3164
/home/runner/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml/src/ggml-sycl/common.hpp:103: SYCL error
libggml.so(+0x79517) [0x7f0cfb879517]
libggml.so(ggml_abort+0xd8) [0x7f0cfb8794a8]
libggml.so(+0x1f2e98) [0x7f0cfb9f2e98]
libggml.so(+0x229198) [0x7f0cfba29198]
libggml.so(_Z25ggml_sycl_compute_forwardR25ggml_backend_sycl_contextP11ggml_tensor+0x5ef) [0x7f0cfb9f57ef]
libggml.so(+0x24033f) [0x7f0cfba4033f]
libggml.so(ggml_backend_sched_graph_compute_async+0x548) [0x7f0cfb8e60f8]
libllama.so(llama_decode+0xbc7) [0x7f0cfd073d47]
./llama-bench() [0x41a580]
./llama-bench() [0x416ccd]
/usr/lib/libc.so.6(+0x261ce) [0x7f0cfce281ce]
/usr/lib/libc.so.6(__libc_start_main+0x8a) [0x7f0cfce2828a]
./llama-bench() [0x41565e]
Aborted (core dumped)
```

# Comparisons
## 7900 XTX
```
$ ./llama-bench -m /models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         pp512 |      2892.62 ± 23.56 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         tg128 |         81.10 ± 0.06 |

build: 1804adb0 (4006)
```
## W7900
```
❯ ./llama-bench -m /models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         pp512 |       2557.94 ± 6.10 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         tg128 |         76.29 ± 0.12 |

build: 1804adb0 (4006)
```
## RTX 3090
```
 CUDA_VISIBLE_DEVICES=1 ./llama-bench -m /models/llm/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         pp512 |      5408.04 ± 27.49 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         tg128 |        150.65 ± 0.90 |

build: 1804adb0 (4006)
```
## RTX 4090
```
❯ CUDA_VISIBLE_DEVICES=0 ./llama-bench -m /models/llm/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         pp512 |     12329.83 ± 43.77 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         tg128 |        172.31 ± 0.04 |

build: 1804adb0 (4006)
```
