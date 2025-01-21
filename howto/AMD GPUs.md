As of August 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is available for Linux or [Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html). It's best to check the latest docs for information:
* https://rocm.docs.amd.com/en/latest/


General [Mamba](https://github.com/conda-forge/miniforge) workflow tip:
```
# Create a baseml so you don't have to keep reinstalling stuff!
mamba create -n baseml python=3.12
mamba activate baseml
# I mostly just use latest stable ROCm pytorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# you can add uv, transformers, huggingface_hub, cmake, ninja, anything else you use everywhere

# From now on you can easily clone your env:
mamba create -n comfyui --clone baseml
mamba activate comfyui
```


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
Compatible iGPUs include the Radeon 780M (gfx1103) on Phoenix and Hawk Point 7X40 and 8X40 APUs and Radeon 890M (gfx1150) on Strix Point (Ryzen AI) APUs. You typically need to apply a `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable to make sure that these are using the right kernels. See also:
- https://github.com/lamikr/rocm_sdk_builder - make a custom ROCm build for your GPU
- https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU - for Windows users, there are pre-built ROCmlibs for many officially unsupported architectures here

Performance 65W 7940HS w/ 64GB of DDR5-5600 (83GB/s theoretical memory bandwidth): [https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589)
* On small (7B) models that fit within the UMA VRAM, ROCm performance is very similar to my M2 MBA's Metal performance. Inference is barely faster than CLBlast/CPU though (~10% faster).
* On a big (70B) model that doesn't fit into allocated VRAM, the ROCm inferences slower than CPU w/ -ngl 0 (CLBlast crashes), and CPU perf is about as expected - about 1.3 t/s inferencing a Q4_K_M. Besides being slower, the ROCm version also caused amdgpu exceptions that killed Wayland 2/3 times (I'm running Linux 6.5.4, ROCm 5.6.1, mesa 23.1.8).

Note: BIOS allows me to set up to 8GB for VRAM in BIOS (UMA_SPECIFIED GART), ROCm does not support GTT (about 35GB/64GB if it did support it, which is not enough for a 70B Q4_0, not that you'd want to at those speeds).

Vulkan drivers can use GTT memory dynamically, but w/ MLC LLM, Vulkan version is 35% slower than CPU-only llama.cpp. Also, the max GART+GTT is still too small for 70B models.
* It may be possible to unlock more UMA/GART memory: [https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056](https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056)
* There is custom allocator that may allow PyTorch to use GTT memory (only useful for PyTorch inferencing obviously): [https://github.com/pomoke/torch-apu-helper](https://github.com/pomoke/torch-apu-helper)
* A writeup of someone playing around w/ ROCm and SD on an older APU: [https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/](https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/)
I was a bit curious at how performance looks like in 2024-09 - using the same model so you can compare how much a year's difference in development makes:
```
# ROCm
❯ ./llama-bench -m /data/ai/models/llm/gguf/meta-llama-2-7b-q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon 780M, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |        262.87 ± 1.23 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |         19.57 ± 0.02 |

build: faac0bae (3841)

# CPU
❯ ./llama-bench -m /data/ai/models/llm/gguf/meta-llama-2-7b-q4_0.gguf
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |       8 |         pp512 |         61.84 ± 0.72 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |       8 |         tg128 |         14.42 ± 0.02 |

build: faac0bae (3841)
```
- On my system, OpenBLAS was slower than the regular CPU version
## AMD NPU (RyzenAI)
The AMD NPU, starting with the 10 TOPS version in the 7X40 (Phoenix Point), 16 TOPS version in the 8X40 (Hawk Point) and 50 TOPS in the Ryzen AI 3XX (Strix Point) are variants of the Xilinx Vitis platform, which AMD has labeled "Ryzen AI." It has it's own drivers and software stack (separate from ROCm). Maybe it'll get folded in one day? Who knows.
- https://ryzenai.docs.amd.com/en/latest/
- https://github.com/amd/RyzenAI-SW (includes a list of software projects)

I won't be spending too much time on this since my 7940HS that I have is 10 TOPS, which is pretty useless, but here are some links and resources:
- [LLMs on RyzenAI with Pytorch](https://github.com/amd/RyzenAI-SW/blob/main/example/transformers/models/llm/docs/README.md)
- [RyzenAI-SW llama.cpp fork](https://github.com/amd/RyzenAI-SW/tree/main/example/transformers/ext/llama.cpp)
	- For upstream, see: https://github.com/ggerganov/llama.cpp/issues/1499
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
## Legacy
Here's a guide on getting PyTorch 2.1.1 working with Polaris (gfx803): https://github.com/nikos230/Run-Pytorch-with-AMD-Radeon-GPU

## Why is ROCm Hardware Support So Limited?
AMD has no intermediate IR, so must compile packages for every single ISA
- 2025: https://github.com/ROCm/ROCm/discussions/4276
- https://github.com/ROCm/ROCm/issues/4224
- https://github.com/ROCm/ROCm/issues/3985
- 2023: https://github.com/ROCm/ROCm/discussions/3893
- 2023: https://github.com/ROCm/ROCm/discussions/4032
- 2023: https://github.com/ROCm/ROCm/discussions/3893
- SPIRV
	- https://github.com/llvm/llvm-project/pull/89796
	- https://github.com/ROCm/SPIRV-LLVM-Translator
	- https://github.com/ROCm/ROCm/issues/3985
- gfxX-generic
	- https://llvm.org/docs/AMDGPUUsage.html#amdgpu-generic-processor-table
AMD developers don't have hardware internally for testing against:
- https://github.com/ROCm/composable_kernel/issues/1020#issuecomment-1896740852

# RDNA3 (navi3x) on Linux
I have several gfx1100 RDNA3 cards, so this will be the the most detailed section of my guide. Some of this may be applicable to different generation GPUs, likely won't be fully tested.

## Driver and ROCm Setup
### Arch Linux
Arch Linux setup is fairly straightforward (can be easier than the official install!) but is community supported by [rocm-arch](https://github.com/rocm-arch/rocm-arch). If you're running an Arch system already, this should be fine, but if you're running a system dedicated to ML, then you should probably prefer Ubuntu LTS for official support.

Install ROCm:
```shell
# More up to date - 6.3.0
paru -S opencl-amd-dev
# These are  at 6.2.2
# yay -S rocm-hip-sdk rocm-ml-sdk rocm-opencl-sdk

# third party monitoring
paru -S amdgpu_top radeontop
```
Install conda (mamba)
```shell
paru -S mambaforge
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

As of 2024-12, on RDNA3, for `bs=1` (single user interactive) inferencing, your best option is probably either llama.cpp for the most compatibility and good speed, or for maximum speed, mlc-llm (but it has limited quantization options and may require quantizing your own models). I ran some speed tests using vLLM's `benchmark_serving.py` for "real world" testing. You can see all the repro details on this page [[vLLM on RDNA3]] but here's the results table w/ Llama 3.1 8B on a W7900:

| Metric                          | vLLM FP16 | vLLM INT8 | vLLM Q5_K_M | llama.cpp Q5_K_M | ExLlamaV2 5.0bpw | MLC q4f16_1 | llama.cpp Q4_K_M |
| ------------------------------- | --------- | --------- | ----------- | ---------------- | ---------------- | ----------- | ---------------- |
| Weights (GB)                    | 14.99     | 8.49      | 5.33        | 5.33             | 5.5              | 4.21        | 4.30             |
| Benchmark duration (s)          | 311.26    | 367.50    | 125.00      | 249.14           | 347.96           | 145.30      | 221.96           |
| Total input tokens              | 6449      | 6449      | 6449        | 6449             | 6449             | 6449        | 6449             |
| Total generated tokens          | 6544      | 6552      | 6183        | 16365            | 16216            | 13484       | 15215            |
| Request throughput (req/s)      | 0.10      | 0.09      | 0.26        | 0.13             | 0.09             | 0.22        | 0.14             |
| Output token throughput (tok/s) | 21.02     | 17.83     | 49.46       | 65.69            | 46.60            | **92.80**   | 68.55            |
| Total Token throughput (tok/s)  | 41.74     | 35.38     | 101.06      | 91.57            | 65.14            | **137.19**  | 97.60            |
| Mean TTFT (ms)                  | 159.58    | 232.78    | 327.56      | **114.67**       | 160.39           | 301.46      | **110.42**       |
| Median TTFT (ms)                | 111.76    | 162.86    | 128.24      | **85.94**        | 148.70           | 176.25      | **74.94**        |
| P99 TTFT (ms)                   | 358.99    | 477.17    | 2911.16     | 362.63           | **303.35**       | 821.72      | 353.58           |
| Mean TPOT (ms)                  | 48.34     | 55.95     | 18.97       | 14.81            | 19.31            | **10.05**   | 14.14            |
| Median TPOT (ms)                | 46.94     | 55.21     | 18.56       | 14.77            | 18.47            | **9.62**    | 14.02            |
| P99 TPOT (ms)                   | 78.78     | 73.44     | 28.75       | 15.88            | 27.35            | **15.46**   | **15.27**        |
| Mean ITL (ms)                   | 46.99     | 55.20     | 18.60       | 15.03            | 21.18            | **10.10**   | 14.38            |
| Median ITL (ms)                 | 46.99     | 55.20     | 18.63       | 14.96            | 19.80            | **9.91**    | 14.43            |
| P99 ITL (ms)                    | 48.35     | 56.56     | 19.43       | 16.47            | 38.79            | **12.68**   | 15.75            |
- vLLM FP8 does not run on RDNA3 
- vLLM bitsandbytes quantization does not run w/ ROCm (multifactor-backend bnb installed) 
- llama.cpp ROCm backend b4276 (HEAD)
- ExLlamaV2 0.2.6 (HEAD)
- MLC nightly 0.18.dev249
### llama.cpp
llama.cpp has ROCm support built-in now (2023-08):
```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release -j
```
* https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#hipblas
* You can use `LLAMA_HIP_UMA=1` for unified memory for APUs but it'll be slower if you don't use it
* `uname -a` , `dkms status` and `apt list | grep rocm | grep '\[installed\]'` to get version numbers of kernel and libs.
* Be sure to set `-j` and make sure you are using all your threads when compiling. You might want `ccache` as well to speed up recompiles
* If you can't get ROCm working, Vulkan is a universal/easy option, but gains and should still give decent gains over CPU inference
* There is a maintained/rebased fork here that may be faster than upstream for RDNA3 and longer context: https://github.com/hjc4869/llama.cpp

2024-09 Update: llama.cpp ROCm inference speeds basically haven't changed all year so I haven't gone and done updates. CUDA is a bit faster w/ FA and Graph support, so has an even bigger lead. There's been some discussion/code with optimizations, but so far those haven't been merged:
- https://github.com/ggerganov/llama.cpp/pull/7011
- https://github.com/ggerganov/llama.cpp/pull/8082

I was curious in just how much performance might be available for optimizations, here's an analysis of 4090 vs 3090 vs 7900 XTX as of 2024-10-04: https://chatgpt.com/share/66ff502b-72fc-8012-95b4-902be6738665

Testing on a W7900 with `llama-bench` on a recent build, I found that the hjc4869 fork is significantly faster than upstream, but that it still runs faster without the current (2024-11) Flash Attention implementation in llama.cpp that with. w/o FA, the pp4096+tg128 speed is 945.03 tok/s vs 799.37 tok/s w/ FA, so about 18.2% faster. With FA, you do save some memory. It maxes out at 6.48 GB vs 6.93 GB (~7%), this will increase as context increases so it may be worth the speed tradeoff.  Note, that vs upstream:
- w/o FA is 945.03 tok/s vs 792.57 tok/s - 19.2% faster
- w/ FA is 799.37 toks/s vs 574.96 tok/s - 39.0% faster
- Note: hjc4869 w/ FA is faster than upstream w/o FA
To replicate, you can run something like:
```
./llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf -p 0 -n 0 -pg 512,128 -pg 1024,128 -pg 2048,128 -pg 4096,128 -fa 0
```

#### Speculative Decode Testing
llama.cpp now has speculative decoding built in the openai-compatible api server.

Running the server:
```
~/ai/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.6 -ngl 99 -ngld 99 -c 32000 -sp
```
- Model (Q4_K_M): https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- Draft Model (Q8_0): https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

vLLM Benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8080' --host localhost --port
 8080 --endpoint='/v1/chat/completions' --model "llama3.1" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Testing with b4277 on a W7900:

| Metric                          | llama.cpp Q4_K_M | llama.cpp w/ +1B DM |
| ------------------------------- | ---------------- | ------------------- |
| Weights (GB)                    | 4.30             | 4.30+1.78           |
| Benchmark duration (s)          | 221.96           | 189.27              |
| Total input tokens              | 6449             | 6449                |
| Total generated tokens          | 15215            | 16319               |
| Request throughput (req/s)      | 0.14             | **0.17**            |
| Output token throughput (tok/s) | 68.55            | **86.22**           |
| Total Token throughput (tok/s)  | 97.60            | **120.30**          |
| Mean TTFT (ms)                  | 110.42           | 110.12              |
| Median TTFT (ms)                | 74.94            | 74.16               |
| P99 TTFT (ms)                   | 353.58           | 352.52              |
| Mean TPOT (ms)                  | 14.14            | **12.99**           |
| Median TPOT (ms)                | 14.02            | **11.87**           |
| P99 TPOT (ms)                   | **15.27**        | 32.88               |
| Mean ITL (ms)                   | 14.38            | **11.40**           |
| Median ITL (ms)                 | 14.43            | **0.04**            |
| P99 ITL (ms)                    | **15.75**        | 155.27              |
- Although sometimes a slightly higher ITL, on average it's lower and gets +25% boost (depends heavily on how closely the draft model predicts the output)
- 3B draft model runs slower than w/o speculative decoding, so not worth it

How well does this work with a larger model (Llama 3.1 70B Q4_K_M)?

| Metric                          | 70B Q4_K_M | 70B + 1B Q8_0 DM | vLLM      |
| ------------------------------- | ---------- | ---------------- | --------- |
| Weights (GB)                    | 39.59      | 39.59+1.22       | 39.59     |
| Memory Usage (MB)               | 46032      | 45345            | 45034     |
| Context                         | 32768      | 20000            | 8000      |
| Load Time                       | 7s         | 8s               | 42m       |
| Benchmark duration (s)          | 1512.45    | 949.35           | 2093.97   |
| Total input tokens              | 6449       | 6449             | 6449      |
| Total generated tokens          | 16417      | 16355            | 6494      |
| Request throughput (req/s)      | 0.02       | **0.03**         | 0.02      |
| Output token throughput (tok/s) | 10.85      | **17.23**        | 3.10      |
| Total Token throughput (tok/s)  | 15.12      | **24.02**        | 6.18      |
| Mean TTFT (ms)                  | 946.74     | 944.03           | 37479.16  |
| Median TTFT (ms)                | 521.19     | 524.84           | 19630.62  |
| P99 TTFT (ms)                   | 3021.41    | 3011.51          | 117896.08 |
| Mean TPOT (ms)                  | 89.51      | **63.83**        | 144.42    |
| Median TPOT (ms)                | 89.14      | **59.99**        | 138.79    |
| P99 TPOT (ms)                   | **94.38**  | 130.05           | 248.71    |
| Mean ITL (ms)                   | 90.46      | **56.31**        | 138.37    |
| Median ITL (ms)                 | 90.58      | **0.15**         | 138.82    |
| P99 ITL (ms)                    | **97.32**  | 322.35           | 141.87    |
- 3B draft model is slower than 1B (Using [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF))
- For 48GB w/ FA, 8-bit kvcache, can get up to 20K context before OOM (w/o the draft model you can get to 32K) `~/ai/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.8 -ngl 99 -ngld 99 -c 20000 -sp -ctk q8_0 -ctv q8_0 -fa`
- vLLM 0.6.4.post2.dev258+gf13cf9ad included as a comparison. Note, that `--gpu_memory_utilization=0.99 --max_model_len 8192` will be close to your max there, also, it will take almost 45 minutes to load (including 1443s and 0.17GB for graph capture, 2449s for engine init) 

```
~/ai/llama.cpp-hjc4869/build/bin/llama-bench -m /models/gguf/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 |  1 |         pp512 |        297.53 ± 0.11 |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 |  1 |         tg128 |         11.35 ± 0.05 |

build: b6af36a5 (4277)
```

### 2024-01-08 Testing
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
### MLC
### Setup
```shell
mamba create -n mlc python=3.11
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62
python -c "import mlc_llm; print(mlc_llm.__version__)"
python -c "import tvm; print(tvm.__file__)"

# Test
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```
* https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
* https://llm.mlc.ai/docs/get_started/quick_start.html

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

The easiest and most reliable way to run vLLM w/ RDNA3 is building the docker image:
```
# dependencies
paru -S docker docker-compose docker-buildx

# build
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 sudo docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .

# check image
sudo docker images

# run (mount your model nad hf folders)
sudo docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /models:/app/model \
   -v /home/lhl/.cache/huggingface:/root/.cache/huggingface \
   docker.io/library/vllm-rocm \
   bash
```
- Note: this docker image does not support hipBLASLt for `gfx1100` and falls back to hipBLAS

If you want to install from source:
```
# We want the nightly PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# May need to copy amd_smi folder locally if you have permission issues installing
pip install /opt/rocm/share/amd_smi

# Dependencies
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

# Undocumented
pip install setuptools_scm
# requires newer cmake than Ubuntu 24.04 LTS provides
mamba install cmake -y

# Build vLLM for RDNA3
PYTORCH_ROCM_ARCH="gfx1100" python setup.py develop

# Test
vllm serve facebook/opt-125m
```
- https://docs.vllm.ai/en/stable/getting_started/amd-installation.html
- There are often faster AOTriton releases that you can link into the torch libs: https://github.com/ROCm/aotriton/releases/
- You can also build your own https://github.com/ROCm/hipBLASLt for your architecture and link in the torch libs as well
### CTranslate2
This is most notably required for faster-whisper (and whisperX)
- [Feature request: AMD GPU support with oneDNN AMD support #1072](https://github.com/OpenNMT/CTranslate2/issues/1072) - the most detailed discussion for AMD support in the CTranslate2 repo
- https://github.com/arlo-phoenix/CTranslate2-rocm - arlo-phoenix created a [hipified fork](https://github.com/OpenNMT/CTranslate2/issues/1072#issuecomment-2271843277) that can run whisperX. Performance appears about [60% faster than whisper.cpp](https://github.com/OpenNMT/CTranslate2/issues/1072#issuecomment-2267170398)
- [CTranslate2: Efficient Inference with Transformer Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html) - 2024-10-24 recent ROCm Blogs post on how upstream might work?

## Training
In Feb 2024 I wrote up some notes:
- https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/

In June 2024 I did a trainer performance shootoff of torchtune vs axolotl (trl) vs unsloth with a 3090, 4090, and W7900:
- https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx

I noticed that AMD has added a lot of simple tutorials in the ROCm docs:
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/index.html
- https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/single-gpu-fine-tuning-and-inference.html
- https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html

You might also want to read this December 2024 article [SemiAnalysis: MI300X vs H100 vs H200 Benchmark Part 1: Training – CUDA Moat Still Alive](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) which focuses more on large-scale pre-training but adds some color on issues you might encounter with more advanced training. For MI300X in particular, you might also want to search for ROCm issues, [like this one](https://github.com/ROCm/ROCm/issues/4021) that took several months to resolve.

(Interestingly, single GPU training, RDNA3 doesn't exhibit the same problems. Multi-GPU )
### axolotl
This has been my preferred trainer for a while: https://github.com/axolotl-ai-cloud/axolotl
It leverages [trl](https://github.com/huggingface/trl) and layers a bunch of optimizations, yaml configs, etc.
### lightning
I haven't used https://github.com/Lightning-AI/pytorch-lightning but here's the Lightning example from: https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#pytorch-lightning-example

Here what the W7900 looked like (after 1 epoch):
```$ CUDA_VISIBLE_DEVICES=0 python test-lightning.py
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/lightning/pytorch/loops/utilities.py:72: `max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit, set `max_epochs=-1`.
/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/lightning/pytorch/trainer/configuration_validator.py:68: You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.
You are using a CUDA device ('AMD Radeon PRO W7900') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type       | Params | Mode
-----------------------------------------------
0 | encoder | Sequential | 100 K  | train
1 | decoder | Sequential | 101 K  | train
-----------------------------------------------
202 K     Trainable params
0         Non-trainable params
202 K     Total params
0.810     Total estimated model params size (MB)
8         Modules in train mode
0         Modules in eval mode
/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
Epoch 0:   0%|                                                  | 0/55000 [00:00<?, ?it/s]/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/torch/nn/modules/linear.py:125: UserWarning: Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas (Triggered internally at ../aten/src/ATen/Context.cpp:296.)
  return F.linear(input, self.weight, self.bias)
Epoch 1:   6%|█▋                          | 3375/55000 [00:07<01:53, 456.73it/s, v_num=10]
```

See also: https://lightning.ai/docs/pytorch/stable/starter/introduction.html

### torchtune
There was an issue w/ hipblaslt in PyTorch when I was trying to get it working that required manual futzing w/ compiles and `.so` files, but since PyTorch will auto-fallback now it should run w/o hassle, but here's the related issue:
- https://github.com/pytorch/torchtune/discussions/1108

Simple test run:
```
pip install torchao torchtune
tune download meta-llama/Llama-2-7b-chat-hf --output-dir /tmp/Llama-2-7b-hf --hf-token $(cat ~/.cache/huggingface/token)
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 tune run lora_finetune_single_device --config llama2/7B_qlora_single_device

# on a W7900 this should take about 6GB of VRAM and about 15h estimated time 
```
### unsloth (NOT WORKING)
Unsloth https://github.com/unslothai/unsloth depends on:
- PyTorch
- Triton
- xformers or flash attention
- bitsandbytes

As of 2024-09, there is a working upstream xformers library (see below), however it's sadly missing support for this function in the ROCm backend:
```
NotImplementedError: Could not run 'xformers::efficient_attention_forward_ck' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'xformers::efficient_attention_forward_ck' is only available for these backends: [CPU, PrivateUse3, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
```
## Libraries and Frameworks
These are probably going to be most useful if you are a developer or training.

AMD's ROCm docs has a list as well, however the docs don't necessarily apply to RDNA3 (since it's [AMD CK](https://github.com/ROCm/composable_kernel) focused, which has [no RDNA3 kernels](https://github.com/ROCm/composable_kernel/issues/1171)! *\*sad trombone\**)
- https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html
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

Uh, actually not quite... I still had to build my own from the multi-backend branch:
```
git clone --depth 1 -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git bnb && cd bnb
pip install -r requirements-dev.txt

# If you don't do this it won't find the version and build will fail!
git fetch --tags

# We just want gfx1100
cmake -DCOMPUTE_BACKEND=hip -S . -DBNB_ROCM_ARCH="gfx1100"
make
pip install .

# this has to be a bug, tries to use rocm62.so no matter what
ln -s bitsandbytes/libbitsandbytes_rocm62.so bitsandbytes/libbitsandbytes_rocm61.so

# test
cd ..
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```
- https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend

You can see some previous discussion here:
- https://github.com/TimDettmers/bitsandbytes/issues/107
- https://github.com/TimDettmers/bitsandbytes/pull/756
- https://github.com/TimDettmers/bitsandbytes/discussions/990
- https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6/tree/rocm
### xformers
The upstream xformers has added experimental ROCm support. Here's how I was able to get it working:
```
# install
pip install -U xformers --index-url https://download.pytorch.org/whl/rocm6.1

# requires https://github.com/ROCm/amdsmi
# you will get an error if you follow the README instructions
apt install amd-smi-lib

# you need to copy since you don't have root instructions to write
cp -r /opt/rocm/share/amd_smi ~/amd_smi
cd ~/amd_smi
pip install .
```

If everything worked, then you should have a working xformers:
```
$ python -m xformers.info
xFormers 0.0.28.post1
memory_efficient_attention.ckF:                    available
memory_efficient_attention.ckB:                    available
memory_efficient_attention.ck_decoderF:            available
memory_efficient_attention.ck_splitKF:             available
memory_efficient_attention.cutlassF:               unavailable
memory_efficient_attention.cutlassB:               unavailable
memory_efficient_attention.fa2F@0.0.0:             unavailable
memory_efficient_attention.fa2B@0.0.0:             unavailable
memory_efficient_attention.fa3F@0.0.0:             unavailable
memory_efficient_attention.fa3B@0.0.0:             unavailable
memory_efficient_attention.triton_splitKF:         available
indexing.scaled_index_addF:                        available
indexing.scaled_index_addB:                        available
indexing.index_select:                             available
sequence_parallel_fused.write_values:              available
sequence_parallel_fused.wait_values:               available
sequence_parallel_fused.cuda_memset_32b_async:     available
sp24.sparse24_sparsify_both_ways:                  available
sp24.sparse24_apply:                               available
sp24.sparse24_apply_dense_output:                  available
sp24._sparse24_gemm:                               available
sp24._cslt_sparse_mm@0.0.0:                        available
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               True
pytorch.version:                                   2.4.1+rocm6.1
pytorch.cuda:                                      available
gpu.compute_capability:                            11.0
gpu.name:                                          AMD Radeon PRO W7900
dcgm_profiler:                                     unavailable
build.info:                                        available
build.cuda_version:                                None
build.hip_version:                                 6.1.40093-bd86f1708
build.python_version:                              3.11.10
build.torch_version:                               2.4.1+rocm6.1
build.env.TORCH_CUDA_ARCH_LIST:                    
build.env.PYTORCH_ROCM_ARCH:                       None
build.env.XFORMERS_BUILD_TYPE:                     Release
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              -allow-unsupported-compiler
build.env.XFORMERS_PACKAGE_FROM:                   wheel-v0.0.28.post1
```

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


Notes:
- Discord
- HN
- Reddit

### Flash Attention 2
This issue in the [ROCm/aotriton](https://github.com/ROCm/aotriton) project: [Memory Efficient Flash Attention for gfx1100 (7900xtx)](https://github.com/ROCm/aotriton/issues/16) is probably the best place to read the story on Flash Attention. As of 2024-09, this support has now been upstreamed to PyTorch 2.5.0+ (you may need to use the nightly if the stable version is not there yet). ([original pull]([https://github.com/pytorch/pytorch/pull/134498](https://github.com/pytorch/pytorch/pull/134498)), [merged pull]([https://github.com/pytorch/pytorch/pull/135869](https://github.com/pytorch/pytorch/pull/135869)))

You might also need to use the environment variable:
```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Using the `examples/benchmark.py` from [pytorch-labs/attention-gym](https://github.com/pytorch-labs/attention-gym) we are able to test this:
```
$ TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python attention-gym/examples/benchmark.py 
Using the default sparsity block size: 128
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                       Causal Mask                                       ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Correctness check passed ✅
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |        150.677 |             14.59 |        764.289 |              7.19 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        363.346 |              6.15 |       1946.23  |              2.87 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        245.548 |              9.1  |        428.728 |             13.02 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
BlockMask(shape=(1, 1, 8192, 8192), sparsity=49.22%, 
(0, 0)
░░                              
██░░                            
████░░                          
██████░░                        
████████░░                      
██████████░░                    
████████████░░                  
██████████████░░                
████████████████░░              
██████████████████░░            
████████████████████░░          
██████████████████████░░        
████████████████████████░░      
██████████████████████████░░    
████████████████████████████░░  
██████████████████████████████░░
)
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                        Alibi Mod                                        ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |        155.3   |             14.16 |        798.569 |              6.88 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        375.784 |             11.7  |       2022.57  |              5.44 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        561.904 |              7.83 |        740.779 |             14.84 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
None
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                   Sliding Window 1024                                   ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Traceback (most recent call last):
  File "/home/lhl/xformers/attention-gym/examples/benchmark.py", line 256, in <module>
    main(**vars(args))
  File "/home/lhl/xformers/attention-gym/examples/benchmark.py", line 234, in main
    available_examples[ex]()
  File "/home/lhl/xformers/attention-gym/examples/benchmark.py", line 216, in <lambda>
    "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024)),
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lhl/xformers/attention-gym/examples/benchmark.py", line 140, in test_mask
    torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)
  File "/home/lhl/miniforge3/envs/xformers/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1530, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 116391936 / 134217728 (86.7%)
Greatest absolute difference: nan at index (0, 0, 1088, 0) (up to 0.1 allowed)
Greatest relative difference: nan at index (0, 0, 1088, 0) (up to 0.01 allowed)
```

It looks like Liger has been doing some independent work as well with Triton kernels that seem to provide a big speedup as well, so maybe worth taking a look at this at some point: [https://github.com/linkedin/Liger-Kernel/pull/275](https://github.com/linkedin/Liger-Kernel/pull/275)

Working Flash Attention is one of the longest running issues for RDNA3. Here are some issues to peruse for more context:
* [https://github.com/vllm-project/vllm/issues/4514](https://github.com/vllm-project/vllm/issues/4514)
* [https://github.com/ROCm/flash-attention/issues/27](https://github.com/ROCm/flash-attention/issues/27)
* [https://github.com/linkedin/Liger-Kernel/issues/126](https://github.com/linkedin/Liger-Kernel/issues/126)
* [https://github.com/pytorch/pytorch/issues/112997](https://github.com/pytorch/pytorch/issues/112997)

**NOTE**: ROCm support was merged into the official FA2 implementation in 2024-08 but does not support RDNA3: https://github.com/Dao-AILab/flash-attention/pull/1010

AMD's CK Flash Attention also arbitrarily keeps dropping MI100 support, but it could work: https://github.com/ROCm/flash-attention/issues/24
MI100 detour:
- https://www.reddit.com/user/TNT3530/comments/1akazn8/amd_instinct_mi100_benchmarks_across_multiple_llm/
- https://news.ycombinator.com/item?id=41727921

### TensorFlow (SHOULD WORK?)
I don't really use TensorFlow, so this is untested, but recent reports are that it should work:
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
Life is short, putting for for later...
## vLLM
vLLM has official RDNA3 (gfx1100 at least) support
https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-rocm


```bash
# Assume you have the latest PyTorch installed already
# Also you should install xformers

# You need to have write permissions
cp -r /opt/rocm/share/amd_smi ~/amd_smi
cd ~/amd_smi
pip install .

cd $VLLM
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

PYTORCH_ROCM_ARCH="gfx1100" python setup.py develop
export GPU_ARCHS=gfx1100
python setup.py develop
```

This seems to run but with some caveats:
- Triton flash should be used by default, but you can use `VLLM_USE_TRITON_FLASH_ATTN=0` if you need to work around this
- Basically no quantization works for AMD. FP8 is only for MI300+
	- https://docs.vllm.ai/en/latest/quantization/supported_hardware.html

## ComfyUI

I didn't have any problems installing ComfyUI from the source instructions, seemed like a pretty well behaved app and I was able to just run python main.py. I did do a bit of tuning and this seemed to work fastes for me (after an initial slower first-run):

```
DISABLE_ADDMM_CUDA_LT=1 PYTORCH_TUNABLEOP_ENABLED=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py
```

(Also there is a recent regression w/ pytorch/hipblaslt: https://github.com/ROCm/hipBLASLt/issues/1243 ; I'm using the latest PyTorch nightly atm to see if it actually fixes things, but ... questionable)

I'm not an SD expert by any means, but on my W7900 (gfx1100, similar to 7900 XTX) when `--use-split-cross-attention` it ends up being about 10% slower than without (and doesn't change memory usage for me ~ 12GB).

I don't know how standard benchmarks are done, but with an SDXL-based checkpoint, I get about 3.14 it/s - it takes ~ 10.0-10.1s to generate a 1024x1024 image w/ 30 steps and uni_pc_bh2 sampler (dpmpp samplers render splotchy/wonky for me) which seems OK? (I'll be seeing how Flux does soon, last time I did much image generation poking around was about 1y agao). In any case it runs about 2X faster than a similar setup on a 7900 XTX on Windows w/ the latest Adrenalin 24.12 + WSL2.
# Windows
I don't use Windows for AI/ML, so this doc is going to be rather sporadically updated (if at all).

## WSL
The Adrenalin Edition 24.12.1 (2024-12) drivers add official support for WSL
- https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-24-12-1.html
- Previously: https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-24-10-21-01-WSL-2.html
The best place to track compatibility will be the official docs:
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/limitations.html#wsl-specific-issues
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/limitations.html#windows-subsystem-for-linux-wsl
Be sure to follow the docs carefully as there are limitations:
- As of 2024-12 Ubuntu 22.04 (not 24.04) is required and will install ROCm 6.2.3
- `rocm-smi` does not work (but you can confirm if your GPU shows up with `rocminfo`)
- "Microsoft does not currently support mGPU setup in WSL."
	- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/mgpu.html#windows-subsystem-for-linux-wsl-support

```
# Install just about everything you might need for compute
amdgpu-install -y --usecase=wsl,rocm,rocmdevtools,openclsdk,hiplibsdk --no-dkms
```
## PyTorch
```
# Install latest Stable PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Replace w/ WSL compatible runtime
location=`pip show torch | grep Location | awk -F ": " '{print $2}'`
cd ${location}/torch/lib/
mv libhsa-runtime64.so libhsa-runtime64.so.torch
ln -s /opt/rocm/lib/libhsa-runtime64.so

# Confirm
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
PyTorch version: 2.5.1+rocm6.2, CUDA available: True, CUDA device count: 1, Device name: AMD Radeon RX 7900 XTX

python -m torch.utils.collect_env
<frozen runpy>:128: RuntimeWarning: 'torch.utils.collect_env' found in sys.modules after import of package 'torch.utils', but prior to execution of 'torch.utils.collect_env'; this may result in unpredictable behaviour
Collecting environment information...
PyTorch version: 2.5.1+rocm6.2
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.35

Python version: 3.12.8 | packaged by conda-forge | (main, Dec  5 2024, 14:24:40) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Radeon RX 7900 XTX (gfx1100)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.2.41133
MIOpen runtime version: 3.2.0
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               16
On-line CPU(s) list:                  0-15
Vendor ID:                            AuthenticAMD
Model name:                           AMD Ryzen 7 5800X3D 8-Core Processor
CPU family:                           25
Model:                                33
Thread(s) per core:                   2
Core(s) per socket:                   8
Socket(s):                            1
Stepping:                             2
BogoMIPS:                             6787.22
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr arat umip vaes vpclmulqdq rdpid
Hypervisor vendor:                    Microsoft
Virtualization type:                  full
L1d cache:                            256 KiB (8 instances)
L1i cache:                            256 KiB (8 instances)
L2 cache:                             4 MiB (8 instances)
L3 cache:                             96 MiB (1 instance)
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Mitigation; safe RET, no microcode
Vulnerability Spec store bypass:      Vulnerable
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] pytorch-triton-rocm==3.1.0
[pip3] torch==2.5.1+rocm6.2
[pip3] torchaudio==2.5.1+rocm6.2
[pip3] torchsde==0.2.6
[pip3] torchvision==0.20.1+rocm6.2
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] pytorch-triton-rocm       3.1.0                    pypi_0    pypi
[conda] torch                     2.5.1+rocm6.2            pypi_0    pypi
[conda] torchaudio                2.5.1+rocm6.2            pypi_0    pypi
[conda] torchsde                  0.2.6                    pypi_0    pypi
[conda] torchvision               0.20.1+rocm6.2           pypi_0    pypi
```
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/limitations.html#running-pytorch-in-virtual-environments
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html#pytorch-rocm-support-matrix
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html
- https://github.com/ROCm/ROCm/issues/3571#issuecomment-2332183716
### Docker
There's a Docker image you can use as well that seems to work OOTB (but uses a hella-old PyTorch)
```
apt install docker.io

docker pull rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2

sudo docker run -it \
--cap-add=SYS_PTRACE  \
--security-opt seccomp=unconfined \
--ipc=host \
--shm-size 8G \
--device=/dev/dxg -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1  \
rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2

root@3c4a2ad600ac:/var/lib/jenkins# python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
PyTorch version: 2.1.2+git53da8f8, CUDA available: True, CUDA device count: 1, Device name: AMD Radeon RX 7900 XTX
```
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html#install-methods
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
cmake.exe .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGGML_HIPBLAS=on  -DCMAKE_C_COMPILER="clang.exe" -DCMAKE_CXX_COMPILER="clang++.exe" -DAMDGPU_TARGETS="gfx1100" -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"

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

# Low Level Programming
For general reference documentation, see:
- [AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/) - includes reference docs for CDNA, RDNA, and earlier generations
- 
For the best general overview of GPU theoretical (and tested) performance for AI/ML I've found, see Stas Bekman's article on [Accelerators](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator) from his amazingly useful [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering). You can use the latest version of his [mamf-finder](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) tool for testing FLOPS as well.
## RDNA3
In order to hit theoretical FLOPS, you must use the dual-issue wave32 VOPD pipeline optimally.
- There are no separate tensor cores
- WMMA has intrinsics FP16, BF16, IU8, IU4
For the best summary, see this article: [AMD GPUOpen: How to accelerate AI applications on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/) (2023-01-10)

For those looking to get more into the nitty gritty, here is reference:
- ["RDNA3" Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf) - Official reference PDF (2023-08-15, 600 pages)
- ["RDNA3.5" Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf) - Official reference PDF (2024-07-23, 650 pages)

For unofficial documentation/reverse engineering, see:
- https://github.com/tinygrad/7900xtx
- https://github.com/tinygrad/tinygrad/blob/master/docs/developer/am.md
- https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/am/amdev.py


2023-01-08 [Chips and Cheese: Microbenchmarking AMD’s RDNA 3 Graphics Architecture]https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture) - test memory bandwidth, WGP/SM throughput for various operations; compares vs RDNA2 (6900XT) and Ada (4090)

2023-07 [Casey Primozic](https://cprimozic.net/) did some testing/benchmarking of the 7900 XTX (TensorFlow, TinyGrad):
- https://cprimozic.net/notes/posts/machine-learning-benchmarks-on-the-7900-xtx/

# Misc Resources
Here's a ROCm fork of DeepSpeed (2023-09):
- https://github.com/ascent-tek/rocm_containers/blob/main/README_DeepSpeed.md

I have a document that updated from April-June 2024 focused on W7900 (RDNA3 gfx1100 workstation version of the 7900 XTX) but I'm folding all up to date info back to this doc:
* [[W7900 Pervasive Computing Project]]
