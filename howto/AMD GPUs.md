As of August 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is available for Linux or [Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html). It's best to check the latest docs for information:
* https://rocm.docs.amd.com/en/latest/
# Hardware
These are the latest officially supported cards:
* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
* https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html
If you have a supported family, you can usually use set `HSA_OVERRIDE_GFX_VERSION` to the closest supported version (eg, `HSA_OVERRIDE_GFX_VERSION=10.3.0`) and get things working.

## RDNA3 (eg 7900 XT, XTX)
As of ROCm 5.7, Radeon RX 7900 XTX, XT, and PRO W7900 are officially supported and many old hacks are no longer necessary:
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html
- https://are-we-gfx1100-yet.github.io/
* [https://news.ycombinator.com/item?id=36574179](https://news.ycombinator.com/item?id=36574179)

## AMD APU
Performance 65W 7940HS w/ 64GB of DDR5-5600 (83GB/s theoretical memory bandwidth): [https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589)
* On small (7B) models that fit within the UMA VRAM, ROCm performance is very similar to my M2 MBA's Metal performance. Inference is barely faster than CLBlast/CPU though (~10% faster).
* On a big (70B) model that doesn't fit into allocated VRAM, the ROCm inferences slower than CPU w/ -ngl 0 (CLBlast crashes), and CPU perf is about as expected - about 1.3 t/s inferencing a Q4_K_M. Besides being slower, the ROCm version also caused amdgpu exceptions that killed Wayland 2/3 times (I'm running Linux 6.5.4, ROCm 5.6.1, mesa 23.1.8).

Note: BIOS allows me to set up to 8GB for VRAM in BIOS (UMA_SPECIFIED GART), ROCm does not support GTT (about 35GB/64GB if it did support it, which is not enough for a 70B Q4_0, not that you'd want to at those speeds).

Vulkan drivers can use GTT memory dynamically, but w/ MLC LLM, Vulkan version is 35% slower than CPU-only llama.cpp. Also, the max GART+GTT is still too small for 70B models.
* It may be possible to unlock more UMA/GART memory: [https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056](https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056)
* There is custom allocator that may allow PyTorch to use GTT memory (only useful for PyTorch inferencing obviously): [https://github.com/pomoke/torch-apu-helper](https://github.com/pomoke/torch-apu-helper)
* A writeup of someone playing around w/ ROCm and SD on an older APU: [https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/](https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/)
## Radeon VII
We have some previous known good memory timings for an old Radeon VII card:
```
sudo sh -c 'echo manual > /sys/class/drm/card0/device/power_dpm_force_performance_level'
sudo sh -c 'echo 8 > /sys/class/drm/card0/device/pp_dpm_sclk'
sudo amdmemorytweak --gpu 0 --ref 7500 --rtp 6 --rrds 3 --faw 12 --ras 19 --rc 30 --rcdrd 11 --rp 11
```

# Linux
## Arch Linux Setup
Arch Linux setup is fairly straightforward (can be easier than the official install!) but is community supported by [rocm-arch](https://github.com/rocm-arch/rocm-arch). If you're running an Arch system already, this should be fine, but if you're running a system dedicated to ML, then you should prefer Ubuntu.

Install ROCm:
```
# all the amd gpu compute stuff
yay -S rocm-hip-sdk rocm-ml-sdk rocm-opencl-sdk

# third party monitoring
yay -S amdgpu_top radeontop
```
Install conda (mamba)
```
yay -S mambaforge
/opt/mambaforge/bin/mamba init fish
```
Create Environment
```
mamba create -n llm
mamba activate llm
```

## Ubuntu
Ubuntu is the most well documented of the officially supported distros:
* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html
* I recommend using the latest LTS (22.04.3) with the HWE kernel
	* https://ubuntu.com/kernel/lifecycle
* The install documents are pretty much complete
	* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html
* You can now use `apt install rocm` to install "everything" (except the drivers, you'll still need `amdgpu-dkms` first).
* Be sure also to look at the "post-install instructions"
	* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/post-install.html

### HWE Kernel
```bash
sudo apt install --install-recommends linux-generic-hwe-22.04
reboot
```
* https://ubuntu.com/kernel/lifecycle

### Prereqs
```bash
# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add the AMDGPU repository for the driver.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.0/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
sudo apt update

# Add the ROCm repository.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 jammy main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600

```
* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html

### Install
```bash
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
# See prerequisites. Adding current user to Video and Render groups
sudo usermod -a -G render,video $LOGNAME
sudo apt update

# Driver
sudo apt install --install-recommends amdgpu-dkms

# Everything else
sudo apt install --install-recommends rocm

reboot
```

## cmath
You may run into some compile errors. You will need `libstdc++-12-dev` in Ubuntu:
```
/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/cuda_wrappers/cmath:27:15: fatal error: 'cmath' file not found
#include_next <cmath>

sudo apt install libstdc++-12-dev
```
## llama.cpp
llama.cpp has  ROCm support built-in now:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_HIPBLAS=1
```
* https://github.com/ggerganov/llama.cpp/#hipblas
* You can use `LLAMA_HIP_UMA=1` for unified memory for APUs

7900 XT + 7900 XTX used together segfaults :(
```
$ ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 2 ROCm devices:
  Device 0: Radeon RX 7900 XT, compute capability 11.0, VMM: no
  Device 1: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
Segmentation fault (core dumped)
```

7900 XT:
```
$ CUDA_VISIBLE_DEVICES=0 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XT, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2065.04 ± 4.61 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |     96.58 ± 0.02 |

build: b7e7982 (1787)
```

7900 XTX:
```
$ CUDA_VISIBLE_DEVICES=1 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2424.44 ± 1.23 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |    118.93 ± 0.04 |

build: b7e7982 (1787)
```

While the Radeon 7900 XTX has  theoretically competitive memory bandwidth and compute, in practice, with ROCm 6.0, hipBLAS still falls behind cuBLAS in llama.cpp:

|  | [7900 XT](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912) | [7900 XTX](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941) | [RTX 3090](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622) | [RTX 4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) |
| ---- | ---- | ---- | ---- | ---- |
| Memory GB | 20 | 24 | 24 | 24 |
| Memory BW GB/s | 800 | 960 | 936.2 | 1008 |
| FP32 TFLOPS | 51.48 | 61.42 | 35.58 | 82.58 |
| FP16 TFLOPS | 103.0 | 122.8 | 35.58 | 82.58 |
| Prompt tok/s | 2065 | 2424 | 2764 | 4650 |
| Prompt % | -14.8% | 0% | +14.0% | +91.8% |
| Inference tok/s | 96.6 | 118.9 | 136.1 | 162.1 |
| Inference % | -18.8% | 0% | +14.5% | +36.3% |
* Tested 2024-01-08 with llama.cpp `b737982 (1787)` and latest ROCm (`dkms amdgpu/6.3.6-1697589.22.04`, `rocm 6.0.0.60000-91~22.04` ) and CUDA (`dkms nvidia/545.29.06, 6.6.7-arch1-1`, `nvcc cuda_12.3.r12.3/compiler.33492891_0` ) on similar platforms (5800X3D for Radeons, 5950X for RTXs)

## ExLlamaV2
Install is straightforward:
```bash
mamba create -n exllamav2 python=3.11
mamba activate exllamav2

# PyTorch: https://pytorch.org/get-started/locally/
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7

# Regular install
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
```

7900 XT
```bash
$ CUDA_VISIBLE_DEVICES=0 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -ps
...
 ** Length  4096 tokens:   3457.0153 t/s

$ CUDA_VISIBLE_DEVICES=0 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -s
...
 ** Position  3968 + 128 tokens:   57.9066 t/s
```

7900 XTX
```bash
$ CUDA_VISIBLE_DEVICES=1 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -ps
...
 ** Length  4096 tokens:   3927.6424 t/s

$ CUDA_VISIBLE_DEVICES=1 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -s
...
 ** Position  3968 + 128 tokens:   61.2481 t/s
```

Running with both GPUs work, although it defaults to loading everything onto one. If you force the VRAM, interestingly, you can get batch=1 inference to perform slightly better:
```
$ python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -ps -gs 4,4
...
 ** Length  4096 tokens:   3458.9969 t/s

$ python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -s -gs 4,4
...
 ** Position  3968 + 128 tokens:   65.2594 t/s 
```

The ROCm kernel is very un-optimized vs the CUDA version, but you can see while inference performance is much lower than llama.cpp, the prompt processing remains ExLlama's strength (this is especially important for long context scenarios like long, multi-turn conversations or RAG).

|  | [7900 XT](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912) | [7900 XTX](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941) | [RTX 3090](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622) | [RTX 4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) |
| ---- | ---- | ---- | ---- | ---- |
| Memory GB | 20 | 24 | 24 | 24 |
| Memory BW GB/s | 800 | 960 | 936.2 | 1008 |
| FP32 TFLOPS | 51.48 | 61.42 | 35.58 | 82.58 |
| FP16 TFLOPS | 103.0 | 122.8 | 35.58 | 82.58 |
| Prompt tok/s | 3457 | 3928 | 5863 | 13955 |
| Prompt % | -12.0% | 0% | +49.3% | +255.3% |
| Inference tok/s | 57.9 | 61.2 | 116.5 | 137.6 |
| Inference % | -5.4% | 0% | +90.4% | +124.8% |
* Tested 2024-01-08 with ExLlamaV2 `3b0f523` and latest ROCm (`dkms amdgpu/6.3.6-1697589.22.04`, `rocm 6.0.0.60000-91~22.04` ) and CUDA (`dkms nvidia/545.29.06, 6.6.7-arch1-1`, `nvcc cuda_12.3.r12.3/compiler.33492891_0` ) on similar platforms (5800X3D for Radeons, 5950X for RTXs)
## vLLM
vLLM supports ROCm starting w/ v0.2.4, but only on MI200 cards...
https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-rocm

CURRENT STATUS: failed to get it working on RDNA3...

It looks like there is a Navi3 Flash Attention branch now: https://github.com/ROCmSoftwarePlatform/flash-attention/issues/27
```
git clone https://github.com/ROCmSoftwarePlatform/flash-attention
git branch -a
git switch howiejay/navi_support
export GPU_ARCHS=gfx1100
pip install .
```
* https://github.com/ROCm/composable_kernel/discussions/1032

Now let's continue:
```bash
git clone https://github.com/vllm-project/vllm 
cd vllm
pip install xformers==0.0.23 --no-deps
bash patch_xformers.rocm.sh

pip install -U -r requirements-rocm.txt

export GPU_ARCHS=gfx1100
python setup.py install # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation

# Error - to work around, we just remove the quantization plugins from `setup.py`
/home/lhl/vllm/vllm/csrc/quantization/gptq/q_gemm.hip:530:20: error: no viable overloaded '='                                                                
            res2.x = __half_as_ushort(__float2half(0));                       
            ~~~~~~ ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                       
/opt/rocm-6.0.0/include/hip/amd_detail/amd_hip_fp16.h:122:21: note: candidate function not viable: no known conversion from 'unsigned short' to 'const __half
' for 1st argument                                                                                                                                           
            __half& operator=(const __half&) = default;                                                                                                      
                    ^                                    

# Compile finishes and installs but when we try to run...
(vllm) lhl@rocm:~/vllm$ python -m vllm.entrypoints.api_server
/home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/vllm-0.2.7+rocm603-py3.11-linux-x86_64.egg/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/vllm-0.2.7+rocm603-py3.11-linux-x86_64.egg/vllm/engine/arg_utils.py", line 6, in <module>
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
  File "/home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/vllm-0.2.7+rocm603-py3.11-linux-x86_64.egg/vllm/config.py", line 9, in <module>
    from vllm.utils import get_cpu_memory, is_hip
  File "/home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/vllm-0.2.7+rocm603-py3.11-linux-x86_64.egg/vllm/utils.py", line 11, in <module>
    from vllm._C import cuda_utils
ImportError: /home/lhl/miniforge3/envs/vllm/lib/python3.11/site-packages/vllm-0.2.7+rocm603-py3.11-linux-x86_64.egg/vllm/_C.cpython-311-x86_64-linux-gnu.so: undefined symbol: _Z9gptq_gemmN2at6TensorES0_S0_S0_S0_b
```

## TensorFlow
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/tensorflow-install.html
```
mamba create -n tf python=3.10
sudo apt install rocm-libs rccl
pip install protobuf=3.19.0
pip install tensorflow-rocm
python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'
```
* Try out: https://cprimozic.net/notes/posts/machine-learning-benchmarks-on-the-7900-xtx/
* Can run script, says it's using ROCm Fusion, but runs on CPU?
```
# get device list
rocminfo

# try hip devices
export HIP_VISIBLE_DEVICES=1
python bench.py

024-01-08 08:53:52.438031: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2015] Ignoring visible gpu device (device: 0, name: Radeon RX 7900 XTX, pci bus id: 0000:0c:00.0) with AMDGPU version : gfx1100. The supported AMDGPU versions are gfx1030, gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942.
```
Apparently you need to build your own TF for `gfx1100` support...
* https://gist.github.com/briansp2020/1e8c3e5735087398ebfd9514f26a0007
* https://cprimozic.net/notes/posts/setting-up-tensorflow-with-rocm-on-7900-xtx/
* https://gist.github.com/BloodBlight/0d36b33d215056395f34db26fb419a63

# Windows

## llama.cpp

For an easy time, go to [llama.cpp's release page](https://github.com/ggerganov/llama.cpp/releases) and download a `bin-win-clblast` version.

In the Windows terminal, run it with `-ngl 99` to load all the layers into memory.

```
.\main.exe -m model.bin -ngl 99
```

On a Radeon 7900XT, you should get about double the performance of CPU-only execution.
### Compile for ROCm
This was last update 2023-09-03 so things might change, but here's how I was able to get things working in Windows.
#### Requirements
* You'll need [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/) installed. Install it with the basic C++ environment.
* Follow AMD's directions and [install the ROCm software for Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/index.html).
* You'll need `git` if you want to pull the latest from the repo (you can either get the [official Windows installer](https://git-scm.com/download/win) or use a package manager like [Chocolatey](https://chocolatey.org/) to `choco install git`) - note, as an alternative, you could just download the Source code.zip from the [https://github.com/ggerganov/llama.cpp/releases/](https://github.com/ggerganov/llama.cpp/releases/)
#### Instructions
First, launch "x64 Native Tools Command Prompt" from the Windows Menu (you can hit the Windows key and just start typing x64 and it should pop up).
```
# You should probably change to a folder you want first for grabbing the source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Make a build folder
mkdir build
cd build

# Make sure the HIP stuff gets picked up
cmake.exe .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLAMA_HIPBLAS=on  -DCMAKE_C_COMPILER="clang.exe" -DCMAKE_CXX_COMPILER="clang++.exe" -DAMDGPU_TARGETS="gfx1100" -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"

# This should build binaries in a bin/ folder
cmake.exe --build .
```

That's it, now you have compiled executables in `build/bin`.

Start a new terminal to run llama.CPP
```
# You can do this in the GUI search for "environment variable" as well
setx /M PATH "C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"

# Or for session
set PATH="C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"
```

If you set just the global you may need to start a new shell before running this in the `llama.cpp` checkout. You can double check it'S working by outputing the path `echo %PATH%` or just running `hipInfo` or another exe in the ROCm bin folder.

NOTE: If your PATH is wonky for some reason you may get missing .dll errors. You can either fix that, or if all else fails, copy the missing files from `"C:\Program Files\AMD\ROCm\5.5\bin` into the `build/bin` folder since life is too short.
#### Results
Here's my `llama-bench` results running a llama2-7b q4_0 and q4_K_M:
```
C:\Users\lhl\Desktop\llama.cpp\build\bin>llama-bench.exe -m ..\..\meta-llama-2-7b-q4_0.gguf -p 3968 -n 128 -ngl 99
ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, compute capability 11.0
| model                      	|   	size | 	params | backend	| ngl | test   	|          	t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA v2 7B mostly Q4_0    	|   3.56 GiB | 	6.74 B | ROCm   	|  99 | pp 3968	|	882.92 ± 1.10 |
| LLaMA v2 7B mostly Q4_0    	|   3.56 GiB | 	6.74 B | ROCm   	|  99 | tg 128 	| 	94.55 ± 0.07 |

build: 69fdbb9 (1148)

C:\Users\lhl\Desktop\llama.cpp\build\bin>llama-bench.exe -m ..\..\meta-llama-2-7b-q4_K_M.gguf -p 3968 -n 128 -ngl 99
ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, compute capability 11.0
| model                      	|   	size | 	params | backend	| ngl | test   	|          	t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA v2 7B mostly Q4_K - Medium |   3.80 GiB | 	6.74 B | ROCm   	|  99 | pp 3968	|	858.74 ± 1.32 |
| LLaMA v2 7B mostly Q4_K - Medium |   3.80 GiB | 	6.74 B | ROCm   	|  99 | tg 128 	| 	78.78 ± 0.04 |

build: 69fdbb9 (1148)
```
### Unsupported Architectures
On Windows, it may not be possible to apply an `HSA_OVERRIDE_GFX_VERSION` override. In that case, these instructions for compiling custom kernels may help: [https://www.reddit.com/r/LocalLLaMA/comments/16d1hi0/guide_build_llamacpp_on_windows_with_amd_gpus_and/](https://www.reddit.com/r/LocalLLaMA/comments/16d1hi0/guide_build_llamacpp_on_windows_with_amd_gpus_and/)

## Misc
Here's a ROCm fork of DeepSpeed: https://github.com/ascent-tek/rocm_containers/blob/main/README_DeepSpeed.md