For the latest Strix Halo / AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`) support, check out:
- 2025-05-02 https://github.com/ROCm/TheRock/discussions/244
- 2025-05-02 https://github.com/ROCm/ROCm/issues/4499
	- https://github.com/ROCm/ROCm/issues/4566
# Testing Checklist
- [ ] ROCm
- [x] llama.cpp
	- [x] HIP (ROCm)
	- [x] Vulkan
	- [x] RPC
	- [ ] Speculative Decoding
		- [ ] 70B
		- [ ] 25-32 Dense
	- [x] wmma
- [x] mamf-finder
- [x] PyTorch
	- [x] hipBLASlt
	- [ ] FA2
- [ ] vLLM
- [ ] SGLang
- [ ] trl
- [ ] Axolotl
- [ ] torchtune
# System Info
```
❯ lsb_release -a
LSB Version:    n/a
Distributor ID: Fedora
Description:    Fedora Linux 43 (Workstation Edition Prerelease)
Release:        43
Codename:       n/a

❯ uname -a
Linux cluster1 6.15.0-0.rc3.20250422gita33b5a08cbbd.29.fc43.x86_64

❯ python -c "import torch; print(f'PyTorch version: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nDevice count: {torch.cuda.device_count()}')"
PyTorch version: 2.5.0a0
CUDA available: True
Device count: 1

❯ python env-info.py
=== System Information ===
Os Info: Fedora Linux 43 (Workstation Edition Prerelease)
Kernel: Linux cluster1 6.15.0-0.rc3.20250422gita33b5a08cbbd.29.fc43.x86_64
Memory Info: Total Memory: 120554 MB

=== GPU Information ===
CUDA: Not found
ROCm: ROCM-SMI version: 3.0.0+unknown
ROCM-SMI-LIB version: 7.3.0
PyTorch CUDA Available: True
PyTorch CUDA Version: N/A
PyTorch HIP Version: 6.3.42134-0

GPU Count: 1
GPU 0: AMD Radeon Graphics

=== Package Versions ===
triton: 3.3.0
torch: 2.5.0a

❯ hipconfig -l
/usr/lib64/rocm/llvm/bin

❯ hipconfig -R
/usr
```

## PyTorch Setup

Despite the first Ryzen AI Max+ processor [launching February 25, 2025](https://www.asus.com/us/news/s02topwrxdtvtura/) with the Asus ROG Flow Z13, as of May 2025 there is still ROCm support ([ROCm #4499](https://github.com/ROCm/ROCm/issues/4499)). In theory it may be possible to build a custom ROCm with `gfx1151` support, but in practice setting this up locally is non-trivial:
- https://github.com/ROCm/TheRock/discussions/244
- https://github.com/lamikr/rocm_sdk_builder

As I prefer to use [Mamba envs](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl) and there is no supported PyTorch build beyond the system PyTorch, I do a slightly (very) janky workaround and symlink the system PyTorch from my venv `site-packages`:
```
torch -> /usr/lib64/python3.13/site-packages/torch
torch-2.5.0a0+gitunknown-py3.13.egg-info -> /usr/lib64/python3.13/site-packages/torch-2.5.0a0+gitunknown-py3.13.egg-info
torchgen -> /usr/lib64/python3.13/site-packages/torchgen
```

While it's not ideal, it beats trying to compile ROCm and PyTorch for an unsupported architecture.

### Docker on Fedora
We can use scottt's Docker image: https://github.com/ROCm/TheRock/discussions/244
```
# Grab image
podman pull docker.io/scottt/therock:pytorch-vision-dev-f41

# 
podman run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --privileged \
  docker.io/scottt/therock:pytorch-vision-dev-f41
```
- I wasn't able to successfully build from source
- toolbox doesn't seem to work properly, but running in podman with the right flags did
- See: https://claude.ai/chat/e9fc7ffd-734b-411b-8bb4-f4028d6c7576
# Peak Performance

RDNA3 has a theoretical 512 FP16 FLOPS/clock/CU.

A Ryzen AI Max 395's Radeon 8060S has 40 CUs at a max clock of 2.9GHz shoul have a peak 59.392 FP16 TFLOPS:
```
512 * 40 * 2.9e9 / 1e12 = 59.392 FP16 TFLOPS
```
- https://chatgpt.com/share/68152629-33a8-8012-817b-62b4fe6bc010
- https://gpuopen.com/learn/wmma_on_rdna3/
- https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
- https://cprimozic.net/notes/posts/machine-learning-benchmarks-on-the-7900-xtx/

This assumes you are using optimized libraries like [rocWMMA](https://github.com/ROCm/rocWMMA) (requires ROCm 6.4) or [hipBLASLt](https://github.com/ROCm/hipBLASLt) otherwise your peak TFLOPS will likely be half of that.

Currently, my test system's results are much lower, however.

There is no official ROCm build for `gfx1151` so I am benchmarking using a custom Fedora `gfx1151` build of PyTorch (2.5) on ROCm 6.3 which only has the [rocBLAS](https://github.com/ROCm/rocBLAS) TensileLibraries available for `gfx1151`.

## rocm_bandwidth_test
We get about 84GB/s from CPU to GPU and within the GPU, 212GB/s
```
git clone https://github.com/ROCm/rocm_bandwidth_test
cd rocm_bandwidth_test
cmake -B build && cmake --build build


❯ build/rocm-bandwidth-test
....................
          RocmBandwidthTest Version: 2.6.0

          Launch Command is: build/rocm-bandwidth-test (rocm_bandwidth -a + rocm_bandwidth -A)


          Device: 0,  AMD Eng Sample: 100-000001243-50_Y
          Device: 1,  AMD Radeon Graphics,  GPU-XX,  c2:0.0

          Inter-Device Access

          D/D       0         1

          0         1         1

          1         1         1


          Inter-Device Numa Distance

          D/D       0         1

          0         0         20

          1         20        0


          Unidirectional copy peak bandwidth GB/s

          D/D       0           1

          0         N/A         84.364

          1         84.147      212.419


          Bidirectional copy peak bandwidth GB/s

          D/D       0           1

          0         N/A         83.489

          1         83.489      N/A
```
## mamf-finder
Using my [mamf-finder](https://github.com/shisa-ai/mamf-finder) repo to test, it takes about just under 35 hours (!) to test with mamf-finder:
```
Warming up the accelerator for 30 secs ... /home/lhl/mamf-finder/mamf-finder/./mamf-finder.py:252: UserWarning: Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas (Triggered internally at /builddir/build/BUILD/python-torch-2.5.1-build/pytorch-v2.5.1/aten/src/ATen/
Context.cpp:296.)
  torch.mm(A, B, out=C)
accelerator warmup finished

Tried  3375 shapes => the best outcomes were:
mean:   5.0 TFLOPS @ 4096x9216x1024 (MxNxK)
median: 5.0 TFLOPS @ 12288x3072x1024 (MxNxK)
max:    5.1 TFLOPS @ 11264x3072x1024 (MxNxK)

Elapsed time: 1 day, 10:40:32
```

As you can see, the max performance is 5.1 BF16 TFLOPS. At the 2.8GHz clock I'm getting, that's an **8.9% efficiency** (57.344 max theoretical).
### in Docker
We get *much* better results using the scottt docker image:
```
[root@4b8fdc8ee74c mamf-finder]# ./test-node.sh
Starting multi-GPU test at Sat May  3 06:17:41 UTC 2025
Waiting for all GPU tests to complete...
Starting tests for GPU 0
Running bfloat16 test on GPU 0
/share/libdrm/amdgpu.ids: No such file or directory

Benchmark started on 2025-05-03 06:17:43

** Command line:
/usr/bin/python ./mamf-finder.py --dtype bfloat16 --m_range 0 16384 1024 --n_range 0 16384 1024 --k_range 0 16384 1024 --output_file=./gpu0-bfloat16-2025-05-03-06-17-41.txt

** Dtype: torch.bfloat16

** Platform/Device info:
Linux 4b8fdc8ee74c 6.15.0-0.rc3.20250422gita33b5a08cbbd.29.fc43.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Apr 22 15:25:32 UTC 2025 x86_64
_CudaDeviceProperties(name='AMD Radeon Graphics', major=11, minor=5, gcnArchName='gfx1151', total_memory=104906MB, multi_processor_count=20, uuid=58580000-0000-0000-0000-000000000000, L2_cache_size=2MB)

** Critical software versions:
torch=2.6.0a0+git90b83a9
hip=6.4.43480-9f04e2822, cuda=None

** Additional notes:
benchmark version: 2


--------------------------------------------------------------------------------


Warming up the accelerator for 30 secs ... accelerator warmup finished
   2916 |   25.6(mean)   25.6(median)   25.8(max) @ 13312x15360x6144     | best:   34.9(mean)   34.9(median)   36.9(max) TFLOPSPS

Tried  3375 shapes => the best outcomes were:
mean:   35.1 TFLOPS @ 15360x3072x1024 (MxNxK)
median: 35.1 TFLOPS @ 15360x3072x1024 (MxNxK)
max:    36.9 TFLOPS @ 6144x3072x3072 (MxNxK)

Elapsed time: 6:04:34
```

At 2.8GHz clock and a max 36.9 TFLOPS that is a much more respectable 64.4% efficiency.
## attention-gym
Install
```
git clone https://github.com/pytorch-labs/attention-gym
cd attention-gym
pip install -e ".[dev]"
pip install -e ".[viz]"
```

Performance bug?
https://github.com/ROCm/MIOpen/pull/3685

```
# TORCH_BLAS_PREFER_HIPBLASLT=0 
HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt/library TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python examples/benchmark.py
```

## aotriton
We need to build and install aotriton:
```
mkdir -p /share/libdrm
cp /opt/rocm/lib/rocm_sysdeps/share/libdrm/amdgpu.ids /share/libdrm/
dnf install gcc gcc-c++ make cmake
dnf install python3-devel
export HIP_PLATFORM=amd
export GPU_TARGETS=gfx1151
git clone https://github.com/ROCm/aotriton
cd aotriton
git submodule sync && git submodule update --init --recursive --force
mkdir build && cd build

#build
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -DAOTRITON_TARGET_ARCH=gfx1151 -G Ninja
ninja install

# ln -s /home/lhl/aotriton/build/install_dir/lib/pyaotriton.cpython-313-x86_64-linux-gnu.so /usr/local/lib/python3.13/site-packages/

export LD_LIBRARY_PATH=/opt/rocm/lib:/home/lhl/aotriton/install_dir/lib:/opt/rocm/lib:

python -c 'import pyaotriton'
```
- takes about 1h wall time to build (27h CPU)

# llama.cpp
## Efficiency
2025-05-03: Currently, the Vulkan backend seems significantly faster than the HIP/ROCm backend on every single `llama-bench` tested model.

2025-05-12: In long context, if you can get HIP to build with rocWMMA then token generation stays high performance w/ FA while it drops significantly with Vulkan.

Interestingly, Vulkan w/ or w/o FA seems to use roughly the same reported memory, which doesn't seem right at all, but that's what the numbers seem to say.

The HIP version performs far below what you'd expect in terms of tok/TFLOP efficiency for prompt processing even vs other RDNA3 architectures: https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/
- `gfx1103` Radeon 780M iGPU gets 14.51 tok/TFLOP. At that efficiency you'd expect the about 850 tok/s that the Vulkan backend delivers. The HIP backends deliver about 350 tok/s, about 40% of the efficiency.
- `gfx1100` Radeon 7900 XTX gets 25.12 tok/TFLOP. At that efficiency you'd expect almost 1500 tok/s, almost double what the Vulkan backend delivers, and >4X what the current HIP backend delivers.

Testing a similar system with Linux 6.14 vs 6.15 showed a 15% performance difference so it's possible future driver updates will improve/fix Strix Halo's ROCm/HIP compute efficiency problems. 

Memory bandwidth efficiency seems better. At 50 tok/s with a 3.56 GB quant, that's about 180 GB/s. This is close to the `rocm_bandwidth_test` results of a peak 212 GB/s of transfer. For HIP and Vulkan we are seeing 70.8-73.3% MBW bandwidth (vs 256 GB/s theoretical peak), which is actually quite good an inline with previously tested RDNA3 APUs.

How bad is the perf? Testing with the standard [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) (Q4_0), the HIP backend barely outperforms the CPU backend (!!!) for prompt processing. Interestingly, despite MBW being theoretically the same, the CPU tg is much worse:

| Run         | pp512 (t/s)       | tg128 (t/s)      | Max Mem (MiB) |
| ----------- | ----------------- | ---------------- | ------------- |
| CPU         | 294.64 ± 0.58     | 28.94 ± 0.04     |               |
| CPU + FA    | 294.36 ± 3.13     | 29.42 ± 0.03     |               |
| Vulkan      | 881.71 ± 1.71     | 52.22 ± 0.05     | **3923**      |
| Vulkan + FA | **884.20 ± 6.23** | **52.73 ± 0.07** | **3923**      |
| HIP         | 348.96 ± 0.31     | 48.72 ± 0.01     | 4219          |
| HIP + FA    | 331.96 ± 0.41     | 45.78 ± 0.02     | 4245          |
| WMMA        | 322.63 ± 1.34     | 48.40 ± 0.02     | 4218          |
| WMMA + FA   | 343.91 ± 0.60     | 50.88 ± 0.01     | 4218          |

## Building

### Vulkan
```
git clone https://github.com/ggml-org/llama.cpp llama.cpp-vulkan
cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON && cmake --build build --config Release -j 32
```
- takes about 1.5 minutes to build

## Qwen 3 MoE
Currently there is a bug where batch size has to be below 360 to prevent a crash. 256 has been tested as the best performer (multiple of 64):
```
❯ llama.cpp-vulkan/build/bin/llama-bench -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf -b 256
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |     256 |           pp512 |        144.36 ± 0.54 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |     256 |           tg128 |         74.76 ± 0.07 |

build: d24d5928 (5255)
```
- https://www.reddit.com/r/LocalLLaMA/comments/1kd5rua/comment/mq8n7sc/

UPDATE: This no longer crashes, but `-b 256` still performs better than on Vulkan, prompt processing is almost 2X faster:
```
❯ build/bin/llama-bench -b 256 -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coo
pmat
| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |     256 |           pp512 |        116.69 ± 0.22 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |     256 |           tg128 |         74.77 ± 0.12 |

build: 43dfd741 (5338)

❯ build/bin/llama-bench -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coo
pmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |           pp512 |         69.31 ± 0.10 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | Vulkan,RPC |  99 |           tg128 |         74.90 ± 0.10 |

build: 43dfd741 (5338)
```

For the HIP backend `-b 256` slows things down though, so this is like a Vulkan only optimization.
## Flash Attention

Measuring memory usage with `rocm-smi`:
```
initial=$(rocm-smi --showmeminfo vram --csv | awk -F, 'NR==2{print int($3/1048576)}'); max=$initial; while sleep 1; do cur=$(rocm-smi --showmeminfo vram --csv | awk -F, 'NR==2{print int($3/1048576)}'); (( cur > max )) && max=$cur; printf "\r%s  used=%4d MiB  Δ=%4d MiB  peak=%4d MiB  Δpeak=%4d MiB " "$(date +%T)" "$cur" "$((cur-initial))" "$max" "$((max-initial))"; done
```

And here's an APU friendly version (measures GTT) using `amdgpu_top`:
```
initial=$(amdgpu_top -d | awk '/^[[:space:]]*GTT/{print int($4)}'); max=$initial; while sleep 1; do cur=$(amdgpu_top -d | awk '/^[[:space:]]*GTT/{print int($4)}'); (( cur > max )) && max=$cur; printf "\r%s  used=%4d MiB  Δ=%4d MiB  peak=%4d MiB  Δpeak=%4d MiB " "$(date +%T)" "$cur" "$((cur-initial))" "$max" "$((max-initial))"; done
```

We compile the latest HEAD `b5343` and test as usual with [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) (Q4_0).
#### pp512/tg128
At the standard `pp512`/`tg128` tests, we see that as tested before, the Vulkan continues to stomp over the HIP backend, and that WMMA makes basically no difference:

| Run         | pp512 (t/s)       | tg128 (t/s)      | Max Mem (MiB) |
| ----------- | ----------------- | ---------------- | ------------- |
| Vulkan      | 881.71 ± 1.71     | 52.22 ± 0.05     | **3923**      |
| Vulkan + FA | **884.20 ± 6.23** | **52.73 ± 0.07** | **3923**      |
| HIP         | 348.96 ± 0.31     | 48.72 ± 0.01     | 4219          |
| HIP + FA    | 331.96 ± 0.41     | 45.78 ± 0.02     | 4245          |
| WMMA        | 322.63 ± 1.34     | 48.40 ± 0.02     | 4218          |
| WMMA + FA   | 343.91 ± 0.60     | 50.88 ± 0.01     | 4218          |

#### pp8192/tg8192
But when we switch to longer context, we see something interesting happen. WMMA + FA basically loses no performance at this longer context length!

Vulkan + FA still has better pp but tg is significantly lower. More data points would be better, but seems like Vulkan performance may continue to decrease as context extends while the HIP+rocWMMA backend should perform better.

| Run         | pp8192 (t/s)      | tg8192 (t/s)     | Max Mem (MiB) |
| ----------- | ----------------- | ---------------- | ------------- |
| Normal      | 245.59 ± 0.10     | 12.43 ± 0.00     | 6+10591       |
| Normal + FA | 190.86 ± 0.49     | 30.01 ± 0.00     | 7+8089        |
| WMMA        | 230.10 ± 0.70     | 12.37 ± 0.00     | 6+10590       |
| WMMA + FA   | 368.77 ± 1.22     | **50.97 ± 0.00** | **7+8062**    |
| Vulkan      | 487.69 ± 0.83     | 7.54 ± 0.02      | 7761+1180     |
| Vulkan + FA | **490.18 ± 4.89** | 32.03 ± 0.01     | 7767+1180     |
- You need to have `rocmwmma` installed - Arch has a package or you will need to build it: https://github.com/ROCm/rocWMMA
- You should then rebuild with `-DGGML_HIP_ROCWMMA_FATTN=ON`
### Building rocWMMA Version

#### Fetch a gfx1151-aware rocWMMA
This is if we have an old rocWMMA that does not have `gfx1151` support merged
```bash
git clone https://github.com/ROCm/rocWMMA ~/llama.cpp/rocWMMA   # PR #538 included
```
- The Fedora package is too old and aborts at compile-time

#### Make hipcc prefer the new headers
Since we need to give precedence to the new includes...
```bash
export CPATH=$HOME/llama.cpp/rocWMMA/library/include:$CPATH
# – or –
export HIPCC_COMPILE_FLAGS_APPEND="-I$HOME/llama.cpp/rocWMMA/library/include"
```
- These env-vars are honoured by every hipcc invocation, putting your copy ahead of `/usr/include/rocwmma`. Can be done w/o root.

#### Stage the ROCm CMake Build-Tools locally
This is if say the Fedora install you have doesn't have `rocm-cmake` (grr)
```bash
git clone https://github.com/ROCm/rocm-cmake ~/src/rocm-cmake
cmake -S ~/src/rocm-cmake -B ~/src/rocm-cmake/build \
      -DCMAKE_INSTALL_PREFIX=$HOME/rocm
cmake --install ~/src/rocm-cmake/build

export CMAKE_PREFIX_PATH=$HOME/rocm:$CMAKE_PREFIX_PATH
```
- Provides `ROCmCMakeBuildToolsConfig.cmake`, satisfying `find_package()` without `sudo`.

#### Stub out the legacy MFMA Flash-Attention kernel
This isn't used but causes compile issues, so we zero it out/skip it.
```cpp
// ggml/src/ggml-cuda/fattn-wmma-f16.cu (replacement)
#include "common.cuh"
#include "fattn-common.cuh"

extern "C" __global__ void flash_attn_ext_f16_stub() { /* noop */ }

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx,
                                       ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}
```
-  `gfx1151` lacks MFMA; compiling the original file fails.  The stub keeps the symbol so the project still links
#### Configure and build llama.cpp for gfx1151
```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)"     \
cmake -S . -B build            \
      -DGGML_HIP=ON            \
      -DGGML_RPC=ON            \
      -DAMDGPU_TARGETS=gfx1151 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
## RPC
Build llama.cpp-hip w/ RPC to run multinode:
```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release     && cmake --build build --config Release -- -j 32

cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON && cmake --build build --config Release -j 32
```

When running `llama-cli` by default it will add itself to nodes and you don't have to run a separate RPC instance. When using `llama-bench` it does not and you should run an RPC server on the host machine as well.

Vulkan has less default memory available than ROCm for some reason!

amdgpu_top / rocm_smi shows there is 14/110000 MiB , 108000 is probably pretty safe:
```
lhl@cluster2:~/llama.cpp$ llama.cpp-vulkan/build/bin/rpc-server -p 50052 -H 0.0.0.0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.0.1'
         Never expose the RPC server to an open network!
         This is an experimental feature and is not secure!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

create_backend: using Vulkan backend
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix co
res: KHR_coopmat
Starting RPC server v2.0.0
  endpoint       : 0.0.0.0:50052
  local cache    : n/a
  backend memory : 78794 MB


lhl@cluster2:~/llama.cpp$ llama.cpp-hip/build/bin/rpc-server -p 50052 -H 0.0.0.0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.0.1'
         Never expose the RPC server to an open network!
         This is an experimental feature and is not secure!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

create_backend: using CUDA backend
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
Starting RPC server v2.0.0
  endpoint       : 0.0.0.0:50052
  local cache    : n/a
  backend memory : 104742 MB

# Manually specify memory
lhl@cluster2:~/llama.cpp$ llama.cpp-vulkan/build/bin/rpc-server -p 50052 -H 0.0.0.0 -m 108000

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.0.1'
         Never expose the RPC server to an open network!
         This is an experimental feature and is not secure!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

create_backend: using Vulkan backend
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
Starting RPC server v2.0.0
  endpoint       : 0.0.0.0:50052
  local cache    : n/a
  backend memory : 108000 MB
```

### Llama 4 Maverick
This is a big MoE model we run to test:
https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
Q4_K_XL = 243 GB
```
❯ llama.cpp-vulkan/build/bin/llama-bench -m ~/models/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf --rpc localhost:50052,192.168.128.12:50052,192.168.128.14:50052
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama4 17Bx128E (Maverick) Q4_K - Medium | 216.18 GiB |   400.71 B | Vulkan,RPC |  99 |           pp512 |         57.93 ± 0.96 |
| llama4 17Bx128E (Maverick) Q4_K - Medium | 216.18 GiB |   400.71 B | Vulkan,RPC |  99 |           tg128 |         16.30 ± 0.14 |
```

## TODO: Speculative Decode
https://github.com/ggml-org/llama.cpp/issues/12968
https://github.com/hjc4869/llama.cpp
https://x.com/hjc4869/status/1913562550064799896

RDNA3 gets a sizable performance uplift with speculative decoding on 4bit models (--draft-max 3 --draft-min 3), and you'll most likely get 8-12 t/s for a 70-72B dense model.

Sweep
https://github.com/AUGMXNT/speed-benchmarking/tree/main/llama.cpp-code

# TODO: Voicechat
https://github.com/AUGMXNT/speed-benchmarking


# Building PyTorch
We want hipBLASLt for general performance and AOTriton for FA2. We should also be able to build with a gfx1151 compatible CK, but that's probably not so useful
## Compile

### hipBLASLt
We are in an env where we have hipBLASLt already but if you're building
```
git clone https://github.com/ROCm/hipBLASLt
cd hipBLASLt
python3 -m pip install -r tensilelite/requirements.txt
# You may need to comment out the PyYAML install
# Also if the paths are wrong...
sudo ln -s /opt/rocm/lib/llvm/bin/amdclang++ /opt/rocm/bin/amdclang++
sudo ln -s /opt/rocm/lib/llvm/bin/amdclang     /opt/rocm/bin/amdclang
export HIP_PLATFORM=amd
export HIPBLASLT_ENABLE_MARKER=0
./install.sh -idc -a gfx1151

# Test if it's working
./hipblaslt-test
```

### aotriton
See the `aotriton` section above, this gets built to `/home/lhl/aotriton/build/install_dir` which you can just point to or you can download the latest release for your version of ROCm: https://github.com/ROCm/aotriton/releases
### Composable Kernel (CK)
```
git clone https://github.com/ROCm/composable_kernel.git
mkdir composable_kernel/build
cd composable_kernel/build

cmake \
        -D CMAKE_PREFIX_PATH=/opt/rocm \
        -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -D CMAKE_BUILD_TYPE=Release \
        -D GPU_TARGETS="gfx1151" \
        -D HIP_PLATFORM=amd \
        ..

# About 15 minutes
time make -j

time make -j install
```

### PyTorch
```
# Enable ROCm (HIP) build and disable CUDA
export USE_ROCM=1
export USE_CUDA=0

# DISABLE KINETO
export USE_KINETO=OFF

# still needed for ROCM_ROCTX_LIB
dnf install roctracer-devel
ln -s /opt/rocm/lib/librocprofiler-sdk-roctx.so /opt/rocm/lib/libroctx64.so

# Will complain about tracing which we're not building...
export BUILD_TEST=OFF

# Needed
dnf install libdrm-devel

# for benchmark.h? - or export BUILD_TEST=OFF
dnf install google-benchmark-devel

# Enable AOTriton integration (FlashAttention kernels) - flag changed w/ 2.8?
export USE_AOTRITON=1
export BUILD_AOTRITON=1

# Specify target GPU architectures for ROCm (limit to gfx1151 for Strix Halo)
export PYTORCH_ROCM_ARCH="gfx1151"

# Point to pre-installed AOTriton (adjust the path to your AOTriton install dir)
export AOTRITON_INSTALLED_PREFIX="/home/lhl/aotriton/build/install_dir"

# Add ROCm and custom library paths to CMake search path
export CMAKE_PREFIX_PATH="/opt/rocm:${CMAKE_PREFIX_PATH}"

# Ensure ROCm libs (and any custom build libs) are in the runtime library path
export LD_LIBRARY_PATH="/opt/rocm/lib:${AOTRITON_INSTALLED_PREFIX}/lib:${LD_LIBRARY_PATH}"

export CXXFLAGS="$CXXFLAGS -Wno-unused-function -Wno-error=unused-function -Wno-error=deprecated-declarations -Wno-error=switch -Wno-error=unused-local-typedefs  -Wno-error=calloc-transposed-args -Wno-array-bound -Wno-error=array-bound"
# export CXXFLAGS="$CXXFLAGS -Wno-error"
# export CCFLAGS="$CFLAGS -Wno-error"
# export HIPCC_FLAGS="$HIPCC_FLAGS -Wno-error"   # for hipcc-compiled kernels


We need to add
defined(__gfx1151__) || 
to
third_party/composable_kernel/include/ck/ck.hpp

# Before we start compiling we need to hipify:
python tools/amd_build/build_amd.py

# see below for rocm-cmake

# see below for rocm-core


# If using CI, modify for STATIC benchmarks OFF
time .ci/pytorch/build.sh

# or just try to directly run:
# cmake3 --build . --target install --config Release

# To get things working/installed properly...
python setup.py develop && python -c "import torch"


# Does this work?
python -c 'import torch,os; print(torch.version.hip, torch.cuda.get_device_name(0))'

# python - <<'PY'
import torch
print("HIP runtime:", torch.version.hip)
print("Device:", torch.cuda.get_device_name(0))
PY
HIP runtime: 6.4.43480-9f04e2822
Device: AMD Radeon Graphics
```
#### rocm-cmake
```
git clone https://github.com/ROCm/rocm-cmake ~/src/rocm-cmake
cmake -S ~/src/rocm-cmake -B ~/src/rocm-cmake/build \
      -DCMAKE_INSTALL_PREFIX=$HOME/rocm
cmake --install ~/src/rocm-cmake/build

export CMAKE_PREFIX_PATH=$HOME/rocm:$CMAKE_PREFIX_PATH
```
#### rocm-core
```
git clone https://github.com/ROCm/rocm-core.git
mkdir -p rocm-core/build
cmake -S rocm-core -B rocm-core/build \
      -DCMAKE_INSTALL_PREFIX=$HOME/rocm \
      -DROCM_VERSION=6.4.0          # match the HIP version you’re using
cmake --build rocm-core/build -j$(nproc)
cmake --install rocm-core/build
```
#### Testing PyTorch
```
# python ../env-info.py
=== System Information ===
Kernel: Linux 2a571ed8a21f 6.15.0-0.rc3.20250422gita33b5a08cbbd.29.fc43.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Apr 22 15:25:32 UTC 2025 x86_64 GNU/Linux
Cpu Info: CPU: AMD Eng Sample (x32)
Memory Info: Total Memory: 120554 MB

=== GPU Information ===
CUDA: Not found
ROCm: ROCM-SMI version: 3.0.0+c865ebb
ROCM-SMI-LIB version: 7.5.0
PyTorch CUDA Available: True
PyTorch CUDA Version: N/A
PyTorch HIP Version: 6.4.43480-9f04e2822

GPU Count: 1
GPU 0: AMD Radeon Graphics

=== Package Versions ===
triton: 3.3.0
torch: 2.8.0a0+git8511d21
torchao: Not installed
transformers: Not installed
flash_attn: Not installed
xformers: Not installed
deepspeed: Not installed
accelerate: Not installed
bitsandbytes: Not installed
axolotl: Not installed
torchtune: Not installed
```

```
# python 02-test-aotriton.py
Triton version: 3.3.0
Driver info: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__firstlineno__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__
', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__static_attributes__', '__str__', '__subclasshook__', '__weakref__', 'active', 'defau
lt', 'reset_active', 'set_active']
Cannot import pyaotriton: No module named 'pyaotriton'
Using device: cuda
Kernel executed successfully: True
LD_LIBRARY_PATH: /opt/rocm/lib:/home/lhl/torch/aotriton/lib:/opt/rocm/lib:
[root@2a571ed8a21f fa]# python 03-test_aotriton_pytorch.py
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
ROCm version: 6.4.43480-9f04e2822
```

```
]# python 03-test_aotriton_pytorch.py
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
ROCm version: 6.4.43480-9f04e2822

Environment variables:
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: Not set
LD_LIBRARY_PATH: /opt/rocm/lib:/home/lhl/torch/aotriton/lib:/opt/rocm/lib:
PYTORCH_ROCM_ARCH: gfx1151
Could not import pyaotriton

Testing scaled_dot_product_attention...
Success! Result shape: torch.Size([1, 1, 128, 64])
```

## Docker Files
- We run our commands from the rocm-TheRock repo root otherwise relative paths are broken


Initial env setup:
```
mamba activate
mamba install cmake ninja patchelf
pip install uv
uv pip install meson

git clone https://github.com/scottt/rocm-TheRock
python ./build_tools/fetch_sources.py
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1151
```
### rocm-dev
We need to build `rocm-dev` first:
```
❯ sh build-rocm-docker.sh
# podman build --build-arg FEDORA_VER=41 -t rocm-dev:41 -f dockerfiles/pytorch-dev/rocm_fedora.Dockerfile .
...

--> 34544a2de4e0
[3/3] STEP 5/5: RUN printf "export PATH=/opt/rocm/bin:$PATH\n" > /etc/profile.d/rocm.sh
[3/3] COMMIT rocm-dev:41
--> 758b36e33cae
Successfully tagged localhost/rocm-dev:41
758b36e33cae4706e6a7030b6ae2604d6655da5c4a6305bfada0ca04339a5f98

lhl in 🌐 cluster1 in rocm-TheRock on  gfx1151 [?] via △ v4.0.2 took 1h1m10s
```

Tag it for pytorch-dev:
```
podman tag localhost/rocm-dev:41 rocm-dev-f41:latest
```

Test it:
```
podman run -it --rm rocm-dev-f41:latest bash
```
### pytorch-dev
```
❯ git diff dockerfiles/
diff --git a/dockerfiles/pytorch-dev/pytorch_dev_fedora.Dockerfile b/dockerfiles/pytorch-dev/pytorch_dev_fedora.Dockerfile
index 462af8c..46e58c2 100644
--- a/dockerfiles/pytorch-dev/pytorch_dev_fedora.Dockerfile
+++ b/dockerfiles/pytorch-dev/pytorch_dev_fedora.Dockerfile
@@ -1,6 +1,11 @@
+# https://github.com/scottt/rocm-TheRock/blob/gfx1151/dockerfiles/pytorch-dev/pytorch_dev_fedora.Dockerfile
+
 ARG FEDORA_VER=41
 FROM rocm-dev-f${FEDORA_VER} AS build

+ENV AMDGPU_TARGETS=gfx1151
+ENV AOTRITON_BUILD_FROM_SOURCE=1
+
 # pytorch-fetch
 RUN --mount=type=cache,id=pytorch-f${FEDORA_VER},target=/therock \
        mkdir -p /therock/pytorch
@@ -9,6 +14,7 @@ RUN --mount=type=cache,id=pytorch-f${FEDORA_VER},target=/therock \
        --mount=type=bind,target=/therock/src,rw \
        python3 /therock/src/external-builds/pytorch/ptbuild.py \
                checkout \
+                --pytorch-ref v2.7.0 \
                --repo /therock/pytorch \
                --depth 1 \
                --jobs 10 \
@@ -24,6 +30,7 @@ RUN --mount=type=cache,id=pytorch-f${FEDORA_VER},target=/therock \
        --mount=type=bind,target=/therock/src,rw \
        python3 /therock/src/external-builds/pytorch/ptbuild.py \
                checkout \
+                --pytorch-ref v2.7.0 \
                --repo /therock/pytorch  \
                --depth 1  \
                --jobs 10
```



```
❯ sh build-pytorch-dev.sh

```

## Working Notes
- https://chatgpt.com/c/681ae923-5244-8012-81a0-ffb56917533b
	- Debugging CK issues when trying to build PyTorch from source
- https://chatgpt.com/c/681d85c5-8700-8012-8d85-8381b5868fa0
	- rocm-dev tagging
	- pytorch cmake w/ roctx , roctracer , kineto
	- pytorch make - werror
- https://chatgpt.com/c/6820b7a7-1d24-8012-8412-e8187098684e
	- llama.cpp + rocwmma
- https://chatgpt.com/c/6821ef0f-dec8-8012-b220-565b7ca7d517
	- going through compiling PyTorch directly

```
cluster1
checkout rock
make changes to rocm, add build scripts, have run scripts
```

Compare to: https://github.com/ROCm/TheRock/blob/main/dockerfiles/pytorch-dev/pytorch_dev_ubuntu_24.04.Dockerfile#L20

https://github.com/ROCm/TheRock/discussions/244

CK off PR: https://github.com/pytorch/pytorch/pull/152951
-DUSE_ROCM_CK_GEMM=ON 

Or use ROCm ≥ 6.5 where CK includes RDNA 3 defines – see ROCm issue #4499 for progress. https://github.com/ROCm/ROCm/issues/4499
https://github.com/ROCm/composable_kernel/issues/775

# TODO: PyTorch Dependent

- [ ] vLLM
- [ ] SGLang
- [ ] torchtune
# Other Reviews

David Huang maintains a llama.cpp fork which has AMD-specific optimizations before it is upstreamed: https://github.com/hjc4869/llama.cpp and did some sweeps/testing on a 60W Ryzen AI Max+ 395 ([HP ZBook Ultra G1a](https://www.hp.com/us-en/workstations/zbook-ultra.html)) on a `gfx1100` GPU_TARGET:
- https://blog.hjc.im/strix-halo-local-llm.html

Jack Stone (Chinese YouTube Hardware Reviewer) did a review of the [GMK EVO-X2 MiniPC](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc) and had a section running some tests w/ LM Studio on Windows - a lot less technical information, but still maybe of interest:
- https://youtu.be/UXjg6Iew9lg?t=238