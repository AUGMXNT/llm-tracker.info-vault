As of August 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is available for Linux or [Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html). It's best to check the latest docs for information:
* https://rocm.docs.amd.com/en/latest/

# Hardware
These are the latest officially supported cards:
* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
* https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html
If you have a supported family, you can usually use set `HSA_OVERRIDE_GFX_VERSION` to the closest supported version (eg, if you have a gfx1031 card you can try `HSA_OVERRIDE_GFX_VERSION=10.3.0` and get things working).

Here's also an interesting 2024-06 writeup of supporting mixed architecture ROCm overrides: https://adamniederer.com/blog/rocm-cross-arch.html
## RDNA3 (eg 7900 XT, XTX)
As of ROCm 5.7, Radeon RX 7900 XTX, XT, and PRO W7900 are officially supported and many old hacks are no longer necessary:
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html
- https://are-we-gfx1100-yet.github.io/
* [https://news.ycombinator.com/item?id=36574179](https://news.ycombinator.com/item?id=36574179)
* I posted my 7900XT/XTX results on Reddit, some conversation here: https://www.reddit.com/r/LocalLLaMA/comments/191srof/amd_radeon_7900_xtxtx_inference_performance/

## AMD APU
Performance 65W 7940HS w/ 64GB of DDR5-5600 (83GB/s theoretical memory bandwidth): [https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589)
* On small (7B) models that fit within the UMA VRAM, ROCm performance is very similar to my M2 MBA's Metal performance. Inference is barely faster than CLBlast/CPU though (~10% faster).
* On a big (70B) model that doesn't fit into allocated VRAM, the ROCm inferences slower than CPU w/ -ngl 0 (CLBlast crashes), and CPU perf is about as expected - about 1.3 t/s inferencing a Q4_K_M. Besides being slower, the ROCm version also caused amdgpu exceptions that killed Wayland 2/3 times (I'm running Linux 6.5.4, ROCm 5.6.1, mesa 23.1.8).

Note: BIOS allows me to set up to 8GB for VRAM in BIOS (UMA_SPECIFIED GART), ROCm does not support GTT (about 35GB/64GB if it did support it, which is not enough for a 70B Q4_0, not that you'd want to at those speeds).

Vulkan drivers can use GTT memory dynamically, but w/ MLC LLM, Vulkan version is 35% slower than CPU-only llama.cpp. Also, the max GART+GTT is still too small for 70B models.
* It may be possible to unlock more UMA/GART memory: [https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056](https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056)
* There is custom allocator that may allow PyTorch to use GTT memory (only useful for PyTorch inferencing obviously): [https://github.com/pomoke/torch-apu-helper](https://github.com/pomoke/torch-apu-helper)
* A writeup of someone playing around w/ ROCm and SD on an older APU: [https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/](https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/)
## AMD NPU (RyzenAI)
The AMD NPU, starting with the 10 TOPS version in the 7X40 (Phoenix Point), 16 TOPS version in the 8X40 (Hawk Point) and 50 TOPS in the Ryzen AI 3XX (Strix Point) are variants of the Xilinx Vitis platform, which AMD has labeled "Ryzen AI." It has it's own drivers and software stack (separate from ROCm). Maybe it'll get folded in one day? Who knows.
- https://ryzenai.docs.amd.com/en/latest/
- https://github.com/amd/RyzenAI-SW (includes a list of software projects)

I won't be spending too much time on this since my 7940HS that I have is 10 TOPS, which is pretty useless, but here are some links and resources:
- [LLMs on RyzenAI with Pytorch](https://github.com/amd/RyzenAI-SW/blob/main/example/transformers/models/llm/docs/README.md)
- [Optimum-AMD](https://github.com/huggingface/optimum-amd) - a HF package for getting NPU acceleration w/ transformers (and ONNX runtime for ROCm)
- Two Japanese Linux setup blogs (somehow Japanese devs must have more patience than English-speaking ones?)
	- https://vengineer.hatenablog.com/entry/2024/06/08/080000
	- https://zenn.dev/haxibami/articles/archlinux-amd-gpu
- AMD implemented an SLM model (AMD-135M) recently (2024-09 announcement) that includes a speculative decode implementation tested on a 7940HS for CPU and NPU. The implementation and benchmarks may be of interest
	- https://community.amd.com/t5/ai/amd-unveils-its-first-small-language-model-amd-135m/ba-p/711368
	- https://github.com/AMD-AIG-AIMA/AMD-LLM
	- https://github.com/AMD-AIG-AIMA/AMD-LLM/blob/main/speculative_decoding/codellama_spec.py
	- https://github.com/AMD-AIG-AIMA/AMD-LLM?tab=readme-ov-file#speculative-decoding

## Radeon VII
We have some previous known good memory timings for an old Radeon VII card:
```shell
sudo sh -c 'echo manual > /sys/class/drm/card0/device/power_dpm_force_performance_level'
sudo sh -c 'echo 8 > /sys/class/drm/card0/device/pp_dpm_sclk'
sudo amdmemorytweak --gpu 0 --ref 7500 --rtp 6 --rrds 3 --faw 12 --ras 19 --rc 30 --rcdrd 11 --rp 11
```
While launched relatively recently (2019), the Radeon VII (gfx906; Radeon Pro VII, MI50) has been deprecated in ROCm, which according to AMD means:
```
The current ROCm release has limited support for this hardware. Existing features and capabilities are maintained, but no new features or optimizations will be added. A future ROCm release will remove support.
```
This is a shame because while it's a bit weak on compute (27 FP16 TFLOPS), it has 16GB HBM2 w/ 1.02 TB/s of MBW, which is not too shabby for inference.
# RDNA3 (navi3x) on Linux
I have several gfx1100 RDNA3 cards, so this will be the the most detailed section of my guide. Some of this may be applicable to different generation GPUs, likely won't be fully tested.

## Driver and ROCm Setup
### Arch Linux
Arch Linux setup is fairly straightforward (can be easier than the official install!) but is community supported by [rocm-arch](https://github.com/rocm-arch/rocm-arch). If you're running an Arch system already, this should be fine, but if you're running a system dedicated to ML, then you should probably prefer Ubuntu LTS for official support.

Install ROCm:
```shell
# all the amd gpu compute stuff
yay -S rocm-hip-sdk rocm-ml-sdk rocm-opencl-sdk

# third party monitoring
yay -S amdgpu_top radeontop
```
Install conda (mamba)
```shell
yay -S mambaforge
/opt/mambaforge/bin/mamba init fish
```
Create Environment
```shell
mamba create -n llm
mamba activate llm
```

### Ubuntu LTS
Ubuntu is the most well documented of the officially supported distros:
* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html
* I recommend using the latest LTS (22.04.4) with the HWE kernel
	* https://ubuntu.com/kernel/lifecycle
* The install documents are pretty much complete
	* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html
* You can now use `apt install rocm` to install "everything" (except the drivers, you'll still need `amdgpu-dkms` first).
* Be sure also to look at the "post-install instructions"
	* https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/post-install.html
#### HWE Kernel
```bash
sudo apt install --install-recommends linux-generic-hwe-22.04
reboot
```
* https://ubuntu.com/kernel/lifecycle
#### Prereqs
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
#### Install
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
#### cmath
You may run into some compile errors. You will need `libstdc++-12-dev` in Ubuntu:
```shell
/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/cuda_wrappers/cmath:27:15: fatal error: 'cmath' file not found
#include_next <cmath>

sudo apt install libstdc++-12-dev
```


## LLM Inferencing
### llama.cpp
llama.cpp has ROCm support built-in now (2023-08):
```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_HIPBLAS=1
```
* https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#hipblas
* You can use `LLAMA_HIP_UMA=1` for unified memory for APUs but it'll be slower if you don't use it
* `uname -a` , `dkms status` and `apt list | grep rocm | grep '\[installed\]'` to get version numbers of kernel and libs
* If you can't get ROCm working, Vulkan is a universal/easy option, but gains and should still give decent gains over CPU inference

Let's run some testing with [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) (Q4_0).

7900 XT + 7900 XTX used together segfaulted on `b7e7982 (1787)` (tested 2024-01-08) but ran with `6db2b41a (1988)` (tested 2024-01-28)
```shell
$ ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 2 ROCm devices:
  Device 0: Radeon RX 7900 XT, compute capability 11.0, VMM: no
  Device 1: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2408.34 ± 1.55 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |    107.15 ± 0.04 |

build: 6db2b41a (1988)
```
- last tested: 2024-01-28

7900 XT:
```shell
$ CUDA_VISIBLE_DEVICES=0 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XT, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2366.44 ± 4.39 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |     97.17 ± 0.02 |

build: 6db2b41a (1988)
```
- last tested: 2024-01-28 

7900 XTX:
```shell
$ CUDA_VISIBLE_DEVICES=1 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2575.87 ± 9.76 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |    119.09 ± 0.06 |

build: 6db2b41a (1988)
```

While the Radeon 7900 XTX has  theoretically competitive memory bandwidth and compute, in practice, with ROCm 6.0, hipBLAS still falls behind cuBLAS in llama.cpp:

|  | [7900 XT](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912) | [7900 XTX](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941) | [RTX 3090](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622) | [RTX 4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) |
| ---- | ---- | ---- | ---- | ---- |
| Memory GB | 20 | 24 | 24 | 24 |
| Memory BW GB/s | 800 | 960 | 936.2 | 1008 |
| Memory BW % | -16.7% | 0% | -2.5% | +5.0% |
| FP32 TFLOPS | 51.48 | 61.42 | 35.58 | 82.58 |
| FP16 TFLOPS | 103.0 | 122.8 | 71/142* | 165.2/330.3* |
| FP16 TFLOPS % | -16.1% | 0% | +15.6%* | +169.0%* |
| Prompt tok/s | 2366 | 2576 | 3251 | 5415 |
| Prompt % | -8.2% | 0% | +26.2% | +110.2% |
| Inference tok/s | 97.2 | 119.1 | 134.5 | 158.4 |
| Inference % | -18.4% | 0% | +12.9% | +33.0% |
* Tested 2024-01-28 with llama.cpp `6db2b41a (1988)` and latest ROCm (`dkms amdgpu/6.3.6-1697589.22.04`, `rocm 6.0.0.60000-91~22.04` ) and CUDA (`dkms nvidia/545.29.06, 6.7.0-arch3-1`, `nvcc cuda_12.3.r12.3/compiler.33492891_0` ) on similar platforms (5800X3D for Radeons, 5950X for RTXs)
* RTX cards have much better FP16/BF16 Tensor FLOPS performance that the inferencing engines are taking advantage of. FP16 FLOPS (32-bit/16-bit accumulation numbers) sourced from Nvidia docs ([3090](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf), [4090](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)_)
#### Vulkan and CLBlast
```bash
### CPU
make clean && make LLAMA_CLBLAST=1
./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968

### CLBlast
# actually we don't have to build CLBlast...
# sudo apt install cmake pkg-config opencl-headers ocl-icd-opencl-dev
sudo apt install libclblast-dev pkg-config
make clean && make LLAMA_CLBLAST=1
GGML_OPENCL_DEVICE=1 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968

### Vulkan
# You could install amdvlk but there's no PPA? https://github.com/GPUOpen-Drivers/AMDVLK#install-with-pre-built-driver
sudo apt install libvulkan-dev vulkan-tools
make clean && make LLAMA_VULKAN=1

### ROCm
# See above for requirements
make clean && make LLAMA_HIPBLAS=1
CUDA_VISIBLE_DEVICES=1 ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
```

|  | 5800X3D CPU | 7900 XTX CLBlast | 7900 XTX Vulkan | 7900 XTX ROCm |
| ---- | ---- | ---- | ---- | ---- |
| Prompt tok/s | 24.5 | 219 | 758 | 2550 |
| Inference tok/s | 10.7 | 35.4 | 52.3 | 119.0 |
* Tested 2024-01-29 with llama.cpp `d2f650cb (1999)` and latest on a 5800X3D w/ DDR4-3600 system with CLBlast `libclblast-dev 1.5.2-2`, Vulkan  `mesa-vulkan-drivers 23.0.4-0ubuntu1~22.04.1`, and ROCm (`dkms amdgpu/6.3.6-1697589.22.04`, `rocm 6.0.0.60000-91~22.04`) 
#### Radeon VII
The Radeon VII was a Vega 20 XT (GCN 5.1) card that was released in February 2019 at $700. It has 16GB of HDM2 memory with a 1024GB/s of memory bandwidth and 26.88 TFLOPS of FP16. Honestly, while the prefill probably doesn't have much more that could be squeezed from it, I would expect with optimization, you would be able to double inference performance (if you could use all its memory bandwidth). 

Radeon Vega VII
```shell
CUDA_VISIBLE_DEVICES=0 ./llama-bench -m llama2-7b-q4_0.gguf -p 3968
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
​￼ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon VII, compute capability 9.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |    432.28 ± 0.93 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |     54.42 ± 0.55 |

build: fea4fd4b (2023)
```
- Tested 2024-02-02 on a Ryzen 5 2400G system with `rocm-core 5.7.1-1`

System Info
```shell
> inxi
CPU: quad core AMD Ryzen 5 2400G with Radeon Vega Graphics (-MT MCP-)
speed/min/max: 1827/1600/3600 MHz Kernel: 6.7.2-arch1-1 x86_64
```

### ExLlamaV2
We'll use `main` on [TheBloke/Llama-2-7B-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ) for testing (GS128 No Act Order).

Install is straightforward:
```shell
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
```shell
$ CUDA_VISIBLE_DEVICES=0 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -ps
...
 ** Length  4096 tokens:   3457.0153 t/s

$ CUDA_VISIBLE_DEVICES=0 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -s
...
 ** Position  3968 + 128 tokens:   57.9066 t/s
```

7900 XTX
```shell
$ CUDA_VISIBLE_DEVICES=1 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -ps
...
 ** Length  4096 tokens:   3927.6424 t/s

$ CUDA_VISIBLE_DEVICES=1 python test_inference.py -m /data/models/gptq/TheBloke_Llama-2-7B-GPTQ -s
...
 ** Position  3968 + 128 tokens:   61.2481 t/s
```

Running with both GPUs work, although it defaults to loading everything onto one. If you force the VRAM, interestingly, you can get batch=1 inference to perform slightly better:
```shell
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
### MLC (NOT WORKING)
### Setup
```shell
mamba create -n mlc python=3.11
mamba install -c conda-forge libgcc-ng
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-rocm57 mlc-ai-nightly-rocm57

export PATH=/opt/rocm/llvm/bin:$PATH

# Missing dependencies
pip install tqdm
pip install safetensors
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
```
* https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
* https://github.com/mlc-ai/mlc-llm/issues/1216
* https://llm.mlc.ai/docs/install/tvm.html#option-1-prebuilt-package

Make a model: https://llm.mlc.ai/docs/compilation/compile_models.html
```shell
mkdir dist

mlc_chat convert_weight  /data/models/hf/augmxnt_shisa-7b-v1 -o dist/shisa-7b-v1-q4f16_1 --quantization q4f16_1

mlc_chat gen_config /data/models/hf/augmxnt_shisa-7b-v1 --conv-template llama-2 -o dist/shisa-7b-v1-q4f16_1 --quantization q4f16_1

$ mlc_chat compile dist/shisa-7b-v1-q4f16_1/mlc-chat-config.json --device rocm -o dist/libs/shisa-7b-v1-q4f16_1-rocm.so

[2024-01-08 23:19:58] INFO auto_config.py:69: Found model configuration: dist/shisa-7b-v1-q4f16_1/mlc-chat-config.json
[2024-01-08 23:19:58] INFO auto_device.py:76: Found device: rocm:0
[2024-01-08 23:19:58] INFO auto_device.py:76: Found device: rocm:1
[2024-01-08 23:19:58] INFO auto_target.py:62: Found configuration of target device "rocm:0": {"thread_warp_size": 32, "mtriple": "amdgcn-amd-amdhsa-hcc", "max_threads_per_block": 1024, "max_num_threads": 256, "kind": "rocm", "max_shared_memory_per_block": 65536, "tag": "", "mcpu": "gfx1100", "keys": ["rocm", "gpu"]}
[2024-01-08 23:19:58] INFO auto_target.py:94: Found host LLVM triple: x86_64-unknown-linux-gnu
[2024-01-08 23:19:58] INFO auto_target.py:95: Found host LLVM CPU: znver3
[2024-01-08 23:19:58] INFO auto_config.py:151: Found model type: mistral. Use `--model-type` to override.
Compiling with arguments:
  --config          MistralConfig(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_hidden_layers=32, rms_norm_eps=1e-05, vocab_size=120128, position_embedding_base=10000.0, num_key_value_heads=8, head_dim=128, sliding_window_size=4096, prefill_chunk_size=4096, attention_sink_size=4, tensor_parallel_shards=1, kwargs={})
  --quantization    GroupQuantize(name='q4f16_1', kind='group-quant', group_size=32, quantize_dtype='int4', storage_dtype='uint32', model_dtype='float16', num_elem_per_storage=8, num_storage_per_group=4, max_int_value=7)
  --model-type      mistral
  --target          {"thread_warp_size": 32, "host": {"mtriple": "x86_64-unknown-linux-gnu", "tag": "", "kind": "llvm", "mcpu": "znver3", "keys": ["cpu"]}, "mtriple": "amdgcn-amd-amdhsa-hcc", "max_threads_per_block": 1024, "max_num_threads": 256, "kind": "rocm", "max_shared_memory_per_block": 65536, "tag": "", "mcpu": "gfx1100", "keys": ["rocm", "gpu"]}
  --opt             flashinfer=0;cublas_gemm=0;cudagraph=0
  --system-lib-prefix ""
  --output          dist/libs/shisa-7b-v1-q4f16_1-rocm.so
  --overrides       context_window_size=None;sliding_window_size=None;prefill_chunk_size=None;attention_sink_size=None;max_batch_size=None;tensor_parallel_shards=None
[2024-01-08 23:19:58] INFO compile.py:131: Creating model from: MistralConfig(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_hidden_layers=32, rms_norm_eps=1e-05, vocab_size=120128, position_embedding_base=10000.0, num_key_value_heads=8, head_dim=128, sliding_window_size=4096, prefill_chunk_size=4096, attention_sink_size=4, tensor_parallel_shards=1, kwargs={})
[2024-01-08 23:19:58] INFO compile.py:141: Exporting the model to TVM Unity compiler
[2024-01-08 23:19:59] INFO compile.py:147: Running optimizations using TVM Unity
[2024-01-08 23:19:59] INFO compile.py:160: Registering metadata: {'model_type': 'mistral', 'quantization': 'q4f16_1', 'context_window_size': -1, 'sliding_window_size': 4096, 'attention_sink_size': 4, 'prefill_chunk_size': 4096, 'tensor_parallel_shards': 1, 'kv_cache_bytes': 536870912}
[2024-01-08 23:19:59] INFO pipeline.py:35: Running TVM Relax graph-level optimizations
[2024-01-08 23:20:00] INFO pipeline.py:35: Lowering to TVM TIR kernels
[2024-01-08 23:20:01] INFO pipeline.py:35: Running TVM TIR-level optimizations
[2024-01-08 23:20:03] INFO pipeline.py:35: Running TVM Dlight low-level optimizations
[2024-01-08 23:20:04] INFO pipeline.py:35: Lowering to VM bytecode
Segmentation fault (core dumped)

```

### vLLM
vLLM has ROCm support and support for specific hardware (which includes gfx1100 now).
- https://docs.vllm.ai/en/stable/getting_started/amd-installation.html
Note: there is a Triton/FA bug:
- https://github.com/vllm-project/vllm/issues/4514
You may be able to work around this with the latest version of PyTorch and Triton (w/ aotriton support) - TBC
## Training
In Feb 2024 I wrote up some notes:
- https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/
### unsloth (NOT WORKING)
Unsloth https://github.com/unslothai/unsloth depends on:
- PyTorch
- Triton
- xformers or flash attention
- bitsandbytes

In theory we have everything we need, and it will startup, however, even after you comment out the `libcuda_dirs()` calls it will die: 
```
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

# You'll need to manually edit site-packages/unsloth/__init__.py
# comment out
# libcuda_dirs()
```
## Libraries and Frameworks
These are probably going to be most useful if you are a developer or training
### PyTorch
PyTorch supports ROCm natively and without code changes (`torch.cuda` just uses ROCm instead). It just needs to be instealled with the ROCm platform:
- https://pytorch.org/get-started/locally/

NOTE: if you want aotriton/FA support you will need PyTorch 2.5.0+ so you may need to install the Preview (Nightly) build instead of Stable (2024-09)
### Triton
Triton also has native ROCm support and you probably can install it and get everything working.
- https://github.com/triton-lang/triton
There is however a ROCm fork where some fixes get upstreamed from:
- https://github.com/ROCm/triton
### bitsandbytes
In 2024-08 and official multi-backend-refactor branch had ROCm support
- https://github.com/bitsandbytes-foundation/bitsandbytes/tree/multi-backend-refactor
As of the end of 2024-09 it appears ROCm support has been folded into the main branch:
- https://github.com/bitsandbytes-foundation/bitsandbytes

You can see some previous discussion here:
- https://github.com/TimDettmers/bitsandbytes/issues/107
- https://github.com/TimDettmers/bitsandbytes/pull/756
- https://github.com/TimDettmers/bitsandbytes/discussions/990
- https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6/tree/rocm
### xformers (NOT WORKING)
There is a ROCm fork but it does not work w/ RDNA3:
- https://github.com/ROCm/xformers/issues/9
	- Depends on CK which does not have RDNA3 support:
		- https://github.com/ROCm/composable_kernel/issues/1171
		- https://github.com/ROCm/composable_kernel/issues/1434

```
pip install -U xformers --index-url https://download.pytorch.org/whl/rocm6.1
pip install amdsmi
```


2024-02-17: The ROCM/xformers fork defaults to a `main` branch, which compiles, but is basically upstream. All the work is done on branches (`develop` seems to be the main one), which sadly ... doesn't compile due to mismatching header files from Composable Kernels.

Note: vLLM has it's own 0.0.23 with a patch to install, but still dies w/ RDNA3
```
# xformers
git clone https://github.com/ROCm/xformers
cd xformers
git fetch
git branch -a
git checkout develop
git submodule update --init --recursive
python setup.py install
python -c 'import xformers; print(xformers.__version__)'
```


CK
FlashInfer
Attention-Gym
Liger
torchtune

Notes:
- W7900
- Discord
- HN
- Reddit

### Flash Attention 2 (SORT OF WORKING)
This seems to work for inference (it only supports batched forward pass, not backward pass) - see the GH issue for more info. You won't be able to train with this.

Also, this is a fork of 2.0.4 so it does not support Mistral's Sliding Window Attention

See:
- https://github.com/ROCm/flash-attention
	- howiejayz/navi_support
- https://github.com/ROCm/flash-attention/issues/27

Install:
```
git clone https://github.com/ROCm/flash-attention
git fetch
git branch -a
git checkout howiejay/navi_support
python setup.py install
```
### TensorFlow (SHOULD WORK?)
Untested, but recent reports are that it should work:
- https://www.reddit.com/r/ROCm/comments/1ahkay9/tensorflow_on_gfx1101_navi32_7800_xt/
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/tensorflow-install.html
```shell
mamba create -n tf python=3.10
sudo apt install rocm-libs rccl
pip install protobuf=3.19.0
pip install tensorflow-rocm
python3 -c 'import tensorflow' 2> /dev/null && echo 'Success' || echo 'Failure'
```
* Try out: https://cprimozic.net/notes/posts/machine-learning-benchmarks-on-the-7900-xtx/
* Can run script, says it's using ROCm Fusion, but runs on CPU?
```shell
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
Life is short, putting for for later
## vLLM (NOT WORKING)
vLLM supports ROCm starting w/ v0.2.4, but only on MI200 cards...
https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-rocm

2024-02-17: failed to get it working on RDNA3, dumps out matrix errors

RDNA3 support should be merged in: https://github.com/vllm-project/vllm/pull/2768
Now let's continue:
```bash
#needs it's own xformers
pip install xformers==0.0.23 --no-deps
bash patch_xformers.rocm.sh

pip install -r requirements-rocm.txt

export GPU_ARCHS=gfx1100
python setup.py install # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation

# 2024-02-10
git clone https://github.com/hongxiayang/vllm.git vllm.navi3x_rocm6
cd vllm.navi3x_rocm6
export GPU_ARCHS=gfx1100
git fetch
git checkout navi3x_rocm6
pip install -e .
# See: https://github.com/vllm-project/vllm/pull/2768



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
# Windows
I don't use Windows for AI/ML, so this doc is going to be rather sporadically updated (if at all).

## llama.cpp

For an easy time, go to [llama.cpp's release page](https://github.com/ggerganov/llama.cpp/releases) and download:
- "hip" version if your GPU is supported (gfx1100, gfx1101, gfx1030, etc)
- "vulkan" or "openblas" version as a fallback if not

Modern versions of llama.cpp should automatically load layers into GPU memory but you can specify something like `-ngl 99` to force it if necessary.

```shell
.\main.exe -m model.bin -ngl 99
```
### Compile for ROCm
This was last update 2023-09-03 so things might change, but here's how I was able to do my own compile in Windows.
#### Requirements
* You'll need [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/) installed. Install it with the basic C++ environment.
* Follow AMD's directions and [install the ROCm software for Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/index.html).
* You'll need `git` if you want to pull the latest from the repo (you can either get the [official Windows installer](https://git-scm.com/download/win) or use a package manager like [Chocolatey](https://chocolatey.org/) to `choco install git`) - note, as an alternative, you could just download the Source code.zip from the [https://github.com/ggerganov/llama.cpp/releases/](https://github.com/ggerganov/llama.cpp/releases/)
#### Instructions
First, launch "x64 Native Tools Command Prompt" from the Windows Menu (you can hit the Windows key and just start typing x64 and it should pop up).
```shell
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
```shell
# You can do this in the GUI search for "environment variable" as well
setx /M PATH "C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"

# Or for session
set PATH="C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"
```

If you set just the global you may need to start a new shell before running this in the `llama.cpp` checkout. You can double check it'S working by outputing the path `echo %PATH%` or just running `hipInfo` or another exe in the ROCm bin folder.

NOTE: If your PATH is wonky for some reason you may get missing .dll errors. You can either fix that, or if all else fails, copy the missing files from `"C:\Program Files\AMD\ROCm\5.5\bin` into the `build/bin` folder since life is too short.
#### Results
Here's my `llama-bench` results running a llama2-7b q4_0 and q4_K_M:
```shell
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

# Misc Resources
Here's a ROCm fork of DeepSpeed (2023-09):
- https://github.com/ascent-tek/rocm_containers/blob/main/README_DeepSpeed.md

2023-07 [Casey Primozic](https://cprimozic.net/) did some testing/benchmarking of the 7900 XTX (TensorFlow, TinyGrad):
- https://cprimozic.net/notes/posts/machine-learning-benchmarks-on-the-7900-xtx/

I have a document that updated from April-June 2024 focused on W7900 (RDNA3 gfx1100 workstation version of the 7900 XTX) but I'm folding all up to date info back to this doc:
* [[W7900 Pervasive Computing Project]]
