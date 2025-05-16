For the latest Strix Halo / AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`) support, check out:
- 2025-05-02 https://github.com/ROCm/TheRock/discussions/244
- 2025-05-02 https://github.com/ROCm/ROCm/issues/4499
	- https://github.com/ROCm/ROCm/issues/4566
# Testing Checklist
- [x] ROCm
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

A Ryzen AI Max 395's Radeon 8060S has 40 CUs at a max clock of 2.9GHz should have a peak 59.392 FP16 TFLOPS:
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

# make sure pyaotriton linked
ln -s /home/lhl/aotriton/build/install_dir/lib/pyaotriton.cpython-313-x86_64-linux-gnu.so /usr/local/lib/python3.13/site-packages/

# make sure c lib is linked
export LD_LIBRARY_PATH=/opt/rocm/lib:/home/lhl/aotriton/build/install_dir/lib:/opt/rocm/lib:

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

## Improving Performance

### rocBLAS w/ hipBLASLt
First we test:
```
rocblas-bench -f gemm_ex -m 1024 -n 1024 -k 1024 \
              --a_type f16_r --b_type f16_r --c_type f16_r \
              --d_type f16_r --compute_type f16_r | grep -i BLASLT

export ROCBLAS_USE_HIPBLASLT=1

rocblas-bench -f gemm_ex -m 1024 -n 1024 -k 1024 \
              --a_type f16_r --b_type f16_r --c_type f16_r \
              --d_type f16_r --compute_type f16_r | grep -i BLASLT
```

If results deon't 
```
git clone https://github.com/ROCm/rocBLAS
cd rocBLAS

# May need to edit install.sh to `elevate_if_not_root dnf install -y ${package_dependencies} --skip-unavailable`
# Also if the paths are wrong...
sudo ln -s /opt/rocm/lib/llvm/bin/amdclang++ /opt/rocm/bin/amdclang++
sudo ln -s /opt/rocm/lib/llvm/bin/amdclang     /opt/rocm/bin/amdclang

dnf install libdrm-devel -y
ldconfig -p | grep -E 'libdrm(_amdgpu)?\.so$'

# -c gives still gives errors, so we just skip the client for now
HIP_PLATFORM=amd ./install.sh -id -j$(nproc) -a gfx1151
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
Driver info: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__firstlineno__', '__format__', '__ge__', '__getattribute__', '__get
state__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__r
epr__', '__setattr__', '__sizeof__', '__static_attributes__', '__str__', '__subclasshook__', '__weakref__', 'active', 'default', 'reset_active', 'set_activ
e']
PyAOTriton imported successfully!
PyAOTriton contents: ['DType', 'HipMemory', 'Stream', 'T0', 'T1', 'T2', 'T4', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '
get_name_suffix', 'hipDeviceSynchronize', 'hipErrorAlreadyAcquired', 'hipErrorAlreadyMapped', 'hipErrorArrayIsMapped', 'hipErrorAssert', 'hipErrorCapturedE
vent', 'hipErrorContextAlreadyCurrent', 'hipErrorContextAlreadyInUse', 'hipErrorContextIsDestroyed', 'hipErrorCooperativeLaunchTooLarge', 'hipErrorDeinitia
lized', 'hipErrorECCNotCorrectable', 'hipErrorFileNotFound', 'hipErrorGraphExecUpdateFailure', 'hipErrorHostMemoryAlreadyRegistered', 'hipErrorHostMemoryNo
tRegistered', 'hipErrorIllegalAddress', 'hipErrorIllegalState', 'hipErrorInitializationError', 'hipErrorInsufficientDriver', 'hipErrorInvalidConfiguration'
, 'hipErrorInvalidContext', 'hipErrorInvalidDevice', 'hipErrorInvalidDeviceFunction', 'hipErrorInvalidDevicePointer', 'hipErrorInvalidGraphicsContext', 'hi
pErrorInvalidHandle', 'hipErrorInvalidImage', 'hipErrorInvalidKernelFile', 'hipErrorInvalidMemcpyDirection', 'hipErrorInvalidPitchValue', 'hipErrorInvalidR
esourceHandle', 'hipErrorInvalidSource', 'hipErrorInvalidSymbol', 'hipErrorInvalidValue', 'hipErrorLaunchFailure', 'hipErrorLaunchOutOfResources', 'hipErro
rLaunchTimeOut', 'hipErrorMapBufferObjectFailed', 'hipErrorMapFailed', 'hipErrorMemoryAllocation', 'hipErrorMissingConfiguration', 'hipErrorNoBinaryForGpu'
, 'hipErrorNoDevice', 'hipErrorNotFound', 'hipErrorNotInitialized', 'hipErrorNotMapped', 'hipErrorNotMappedAsArray', 'hipErrorNotMappedAsPointer', 'hipErro
rNotReady', 'hipErrorNotSupported', 'hipErrorOperatingSystem', 'hipErrorOutOfMemory', 'hipErrorPeerAccessAlreadyEnabled', 'hipErrorPeerAccessNotEnabled', '
hipErrorPeerAccessUnsupported', 'hipErrorPriorLaunchFailure', 'hipErrorProfilerAlreadyStarted', 'hipErrorProfilerAlreadyStopped', 'hipErrorProfilerDisabled
', 'hipErrorProfilerNotInitialized', 'hipErrorRuntimeMemory', 'hipErrorRuntimeOther', 'hipErrorSetOnActiveProcess', 'hipErrorSharedObjectInitFailed', 'hipE
rrorSharedObjectSymbolNotFound', 'hipErrorStreamCaptureImplicit', 'hipErrorStreamCaptureInvalidated', 'hipErrorStreamCaptureIsolation', 'hipErrorStreamCapt
ureMerge', 'hipErrorStreamCaptureUnjoined', 'hipErrorStreamCaptureUnmatched', 'hipErrorStreamCaptureUnsupported', 'hipErrorStreamCaptureWrongThread', 'hipE
rrorTbd', 'hipErrorUnknown', 'hipErrorUnmapFailed', 'hipErrorUnsupportedLimit', 'hipError_t', 'hipSuccess', 'kBFloat16', 'kFloat16', 'kFloat32', 'kInt16',
'kInt32', 'kInt64', 'kInt8', 'kUInt16', 'kUInt32', 'kUInt64', 'kUInt8', 'kUnknown', 'v2']
Using device: cuda
Kernel executed successfully: True
LD_LIBRARY_PATH: /opt/rocm/lib:/home/lhl/aotriton/build/install_dir/lib:/opt/rocm/lib:
```

```
# python 03-test_aotriton_pytorch.py
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
ROCm version: 6.4.43480-9f04e2822

Environment variables:
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: Not set
LD_LIBRARY_PATH: /opt/rocm/lib:/home/lhl/aotriton/build/install_dir/lib:/opt/rocm/lib:
PYTORCH_ROCM_ARCH: gfx1151

pyaotriton imported successfully
torch.ops.aotriton is available
Registered aten ops: 834

Testing scaled_dot_product_attention...
Success! Result shape: torch.Size([1, 1, 128, 64])
```

```
# python 04-test_attention_small.py
Testing with sizes: batch=1, heads=1, seq_len=128, head_dim=64
Basic attention success! Result shape: torch.Size([1, 1, 128, 64])
AOTriton attention success! Result shape: torch.Size([1, 1, 128, 64])
```

```
# python 05-attention-bench.py
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                  AOTriton Status Check                                  ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
ROCm version: 6.4.43480-9f04e2822
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: 1
pyaotriton imported successfully
torch.ops.aotriton is available

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                           Testing Tiny: B=1, H=1, S=128, D=64                           ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Estimated memory per QKV tensor: 0.00 GB
Total QKV memory: 0.00 GB
+--------------+----------------+-------------------+----------------+-------------------+
| Operation    |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+==============+================+===================+================+===================+
| Causal FA2   |         0.0886 |              0.02 |         0.125  |              0.04 |
+--------------+----------------+-------------------+----------------+-------------------+
| Regular SDPA |         0.0689 |              0.03 |         0.1241 |              0.04 |
+--------------+----------------+-------------------+----------------+-------------------+

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                           Testing Small: B=2, H=4, S=512, D=64                          ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Estimated memory per QKV tensor: 0.00 GB
Total QKV memory: 0.00 GB
+--------------+----------------+-------------------+----------------+-------------------+
| Operation    |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+==============+================+===================+================+===================+
| Causal FA2   |         0.5034 |              0.53 |         0.6336 |              1.06 |
+--------------+----------------+-------------------+----------------+-------------------+
| Regular SDPA |         0.4589 |              0.58 |         0.6298 |              1.07 |
+--------------+----------------+-------------------+----------------+-------------------+

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                          Testing Medium: B=4, H=8, S=1024, D=64                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Estimated memory per QKV tensor: 0.00 GB
Total QKV memory: 0.01 GB
+--------------+----------------+-------------------+----------------+-------------------+
| Operation    |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+==============+================+===================+================+===================+
| Causal FA2   |        16.2504 |              0.26 |        16.0349 |              0.67 |
+--------------+----------------+-------------------+----------------+-------------------+
| Regular SDPA |        15.5982 |              0.28 |        16.0953 |              0.67 |
+--------------+----------------+-------------------+----------------+-------------------+

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                          Testing Large: B=8, H=16, S=2048, D=64                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Estimated memory per QKV tensor: 0.03 GB
Total QKV memory: 0.09 GB
+--------------+----------------+-------------------+----------------+-------------------+
| Operation    |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+==============+================+===================+================+===================+
| Causal FA2   |        151.853 |              0.45 |        131.531 |              1.31 |
+--------------+----------------+-------------------+----------------+-------------------+
| Regular SDPA |        120.143 |              0.57 |        131.255 |              1.31 |
+--------------+----------------+-------------------+----------------+-------------------+

╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                         Testing XLarge: B=16, H=16, S=4096, D=64                        ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Estimated memory per QKV tensor: 0.12 GB
Total QKV memory: 0.38 GB
Memory access fault by GPU node-1 (Agent handle: 0x55b017570c40) on address 0x7fcd499e6000. Reason: Page not present or supervisor privilege.
Aborted (core dumped)

```

```
# python 06-check-hipblaslt.py
=== Environment ===
LD_LIBRARY_PATH: /opt/rocm/lib:/home/lhl/aotriton/build/install_dir/lib:/opt/rocm/lib:
PYTORCH_ROCM_ARCH: gfx1151
HSA_OVERRIDE_GFX_VERSION: Not set

=== PyTorch Info ===
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
ROCm version: 6.4.43480-9f04e2822

=== GPU Info ===
GPU: AMD Radeon Graphics
GPU Architecture: ['gfx1151']
rocminfo: Name:                    gfx1151
rocminfo: Name:                    amdgcn-amd-amdhsa--gfx1151
rocminfo: Name:                    amdgcn-amd-amdhsa--gfx11-generic

=== hipBLASLt Check ===
hipBLASLt linking: libhipblaslt.so.0 => /opt/rocm/lib/libhipblaslt.so.0 (0x00007f14e2e05000)
hipBLASLt linking: libhipblaslt-d.so.0 => /opt/rocm/lib/libhipblaslt-d.so.0 (0x00007f14dbdcb000)

hipBLASLt directory contents:
  library/
    TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.dat
    TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1151.dat
    TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx1151.dat
    TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1151.dat
    TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.dat
    ... and 43 more files
```

```
# python 08-test-hipblaslt-perf.py
Environment check:
PYTORCH_ROCM_ARCH: gfx1151
HIPBLASLT_TENSILE_LIBPATH: /opt/rocm/lib/hipblaslt/library
TORCH_BLAS_PREFER_HIPBLASLT: 1
Testing GEMM performance...
GEMM 4096x4096x4096: 21.613 ms, 6.36 TFLOPS

Testing Attention performance...
Attention 8x16x2048x64: 150.233 ms, 0.91 TFLOPS

WARNING: GEMM performance is low. hipBLASLt may not be properly configured.
Check that:
1. The correct architecture kernels are in /opt/rocm/lib/hipblaslt/library
2. HIPBLASLT_TENSILE_LIBPATH is set correctly
3. Your GPU architecture matches the available kernels

### They are...
# /opt/rocm/lib/hipblaslt/library/TensileLibrary_lazy_gfx1151.dat
# /opt/rocm/lib/hipblaslt/library/extop_gfx1151.co
```

```
# python 09-test-attention-backend.py
=== Environment ===
PYTORCH_ROCM_ARCH: gfx1151
HIPBLASLT_TENSILE_LIBPATH: /opt/rocm/lib/hipblaslt/library
TORCH_BLAS_PREFER_HIPBLASLT: 1
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: 1

=== Backend Check ===
PyTorch version: 2.8.0a0+git8511d21
CUDA available: True
Current device: 0
Device name: AMD Radeon Graphics

=== AOTriton Check ===
pyaotriton imported successfully
torch.ops.aotriton is available
AOTriton ops: ['__doc__', '__loader__', '__name__', '__package__', '__spec__', '_dir', 'name']

=== SDPA Backends ===
Flash SDPA enabled: True
Memory efficient SDPA enabled: True
Math SDPA enabled: True

=== Testing Attention Variants ===

1. Standard SDPA (no causal):
Standard SDPA: 118.740 ms, 1.16 TFLOPS

2. Causal SDPA:
Causal SDPA: 149.004 ms, 0.92 TFLOPS

3. SDPA with attn_mask:
SDPA with mask: 149.673 ms, 0.92 TFLOPS

4. Force Flash Attention backend:
/usr/lib64/python3.13/contextlib.py:109: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: Memory efficient kernel not used because: (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:859.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: Memory Efficient attention has been runtime disabled. (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/sdp_utils_cpp.h:550.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: Flash attention kernel not used because: (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:861.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: Flash attention was not compiled for current AMD GPU architecture. Attempting to run on architecture gfx1151 (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:241.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: CuDNN attention kernel not used because: (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:863.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py:95: UserWarning: Torch was not compiled with cuDNN attention. (Triggered internally at /home/lhl/torch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:618.)
  lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
Traceback (most recent call last):
  File "/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py", line 93, in <module>
    test_attention_variant(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Flash Attention",
        ^^^^^^^^^^^^^^^^^^
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py", line 58, in test_attention_variant
    _ = func(q, k, v)
  File "/home/lhl/strix-halo-testing/flash-attention/09-test-attention-backend.py", line 95, in <lambda>
    lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: No available kernel. Aborting execution.

```

```
# python 10-test_aotriton_direct.py
=== AOTriton Direct Test ===
AOTriton module: <module 'pyaotriton' from '/usr/local/lib/python3.13/site-packages/pyaotriton.cpython-313-x86_64-linux-gnu.so'>
AOTriton directory: ['DType', 'HipMemory', 'Stream', 'T0', 'T1', 'T2', 'T4', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'get_name_suffix', 'hipDeviceSynchronize', 'hipErrorAlreadyAcquired', 'hipErrorAlreadyMapped', 'hipErrorArrayIsMapped', 'hipErrorAssert', 'hipErrorCapturedEvent', 'hipErrorContextAlreadyCurrent', 'hipErrorContextAlreadyInUse', 'hipErrorContextIsDestroyed', 'hipErrorCooperativeLaunchTooLarge', 'hipErrorDeinitialized', 'hipErrorECCNotCorrectable', 'hipErrorFileNotFound', 'hipErrorGraphExecUpdateFailure', 'hipErrorHostMemoryAlreadyRegistered', 'hipErrorHostMemoryNotRegistered', 'hipErrorIllegalAddress', 'hipErrorIllegalState', 'hipErrorInitializationError', 'hipErrorInsufficientDriver', 'hipErrorInvalidConfiguration', 'hipErrorInvalidContext', 'hipErrorInvalidDevice', 'hipErrorInvalidDeviceFunction', 'hipErrorInvalidDevicePointer', 'hipErrorInvalidGraphicsContext', 'hipErrorInvalidHandle', 'hipErrorInvalidImage', 'hipErrorInvalidKernelFile', 'hipErrorInvalidMemcpyDirection', 'hipErrorInvalidPitchValue', 'hipErrorInvalidResourceHandle', 'hipErrorInvalidSource', 'hipErrorInvalidSymbol', 'hipErrorInvalidValue', 'hipErrorLaunchFailure', 'hipErrorLaunchOutOfResources', 'hipErrorLaunchTimeOut', 'hipErrorMapBufferObjectFailed', 'hipErrorMapFailed', 'hipErrorMemoryAllocation', 'hipErrorMissingConfiguration', 'hipErrorNoBinaryForGpu', 'hipErrorNoDevice', 'hipErrorNotFound', 'hipErrorNotInitialized', 'hipErrorNotMapped', 'hipErrorNotMappedAsArray', 'hipErrorNotMappedAsPointer', 'hipErrorNotReady', 'hipErrorNotSupported', 'hipErrorOperatingSystem', 'hipErrorOutOfMemory', 'hipErrorPeerAccessAlreadyEnabled', 'hipErrorPeerAccessNotEnabled', 'hipErrorPeerAccessUnsupported', 'hipErrorPriorLaunchFailure', 'hipErrorProfilerAlreadyStarted', 'hipErrorProfilerAlreadyStopped', 'hipErrorProfilerDisabled', 'hipErrorProfilerNotInitialized', 'hipErrorRuntimeMemory', 'hipErrorRuntimeOther', 'hipErrorSetOnActiveProcess', 'hipErrorSharedObjectInitFailed', 'hipErrorSharedObjectSymbolNotFound', 'hipErrorStreamCaptureImplicit', 'hipErrorStreamCaptureInvalidated', 'hipErrorStreamCaptureIsolation', 'hipErrorStreamCaptureMerge', 'hipErrorStreamCaptureUnjoined', 'hipErrorStreamCaptureUnmatched', 'hipErrorStreamCaptureUnsupported', 'hipErrorStreamCaptureWrongThread', 'hipErrorTbd', 'hipErrorUnknown', 'hipErrorUnmapFailed', 'hipErrorUnsupportedLimit', 'hipError_t', 'hipSuccess', 'kBFloat16', 'kFloat16', 'kFloat32', 'kInt16', 'kInt32', 'kInt64', 'kInt8', 'kUInt16', 'kUInt32', 'kUInt64', 'kUInt8', 'kUnknown', 'v2']

AOTriton v2 available: ['CppTune', 'CppTuneSpecialKernelIndex', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'flash', 'kDefault', 'kSkipGPUCall']

=== Torch Ops ===
torch.ops.aotriton: ['__doc__', '__loader__', '__name__', '__package__', '__spec__', '_dir', 'name']
No flash_attention found in aotriton ops

=== Testing AOTriton Functions ===
Found AOTriton flash attention functions!
Direct AOTriton call failed: attn_fwd(): incompatible function arguments. The following argument types are supported:
    1. (q: pyaotriton.T4, k: pyaotriton.T4, v: pyaotriton.T4, b: pyaotriton.T4, sm_scale: float, softmax_lse: pyaotriton.T2, out: pyaotriton.T4, dropout_p: float, philox_seed: pyaotriton.T0, philox_offset1: pyaotriton.T0, philox_offset2: int, philox_seed_output: pyaotriton.T0, philox_offset_output: pyaotriton.T0, encoded_softmax: pyaotriton.T4, is_causal: bool, atomic_for_causal: pyaotriton.T0, stream: pyaotriton.Stream = None, extargs: pyaotriton.v2.flash.FwdExtraArguments = <pyaotriton.v2.flash.FwdExtraArguments object at 0x7f1e90857470>) -> pyaotriton.hipError_t

Invoked with: tensor([[[[ 8.7842e-01,  2.9517e-01, -1.3975e+00,  ..., -5.7520e-01,
            7.1777e-01, -5.8887e-01],
          [-3.4741e-01, -1.2568e+00,  8.9941e-01,  ...,  4.9902e-01,
           -4.6289e-01, -1.2549e+00],
          [-1.1465e+00,  2.9297e-01, -4.3457e-02,  ...,  3.3594e-01,
           -1.1025e+00,  6.3553e-03],
          ...,
          [ 1.4199e+00, -2.2656e-01, -9.6582e-01,  ...,  3.2617e-01,
            1.8823e-01, -1.8184e+00],
          [ 1.1436e+00,  8.3191e-02, -8.3740e-01,  ...,  9.8022e-02,
           -1.7744e+00, -4.8975e-01],
          [-7.4365e-01,  2.8174e-01,  3.2153e-01,  ..., -1.7080e+00,
            9.6436e-02,  2.3965e+00]],

         [[-4.2773e-01, -2.1094e+00, -1.8096e+00,  ...,  1.0994e-02,
            1.0537e+00,  4.5361e-01],
          [ 2.0176e+00, -1.3223e+00,  6.1719e-01,  ...,  1.4807e-01,
           -1.4102e+00,  1.3213e+00],
          [ 2.0801e-01,  7.6562e-01, -3.2593e-02,  ..., -7.0654e-01,
            4.9561e-01,  2.1074e+00],
          ...,
          [-1.0483e-02,  8.0908e-01, -5.3857e-01,  ...,  8.7256e-01,
           -1.1055e+00, -1.9568e-01],
          [ 8.2812e-01, -3.1641e-01,  1.1484e+00,  ..., -1.0352e+00,
            7.2266e-01,  1.4404e+00],
          [-6.9287e-01, -7.0801e-01,  2.2925e-01,  ...,  1.6025e+00,
           -4.2773e-01, -1.7559e+00]],

         [[ 8.8672e-01, -3.6475e-01, -7.0508e-01,  ...,  4.6167e-01,
            1.6514e+00,  2.6538e-01],
          [-5.7422e-01,  1.0586e+00,  1.8730e+00,  ...,  3.6072e-02,
            2.4688e+00, -2.4688e+00],
          [-7.0459e-01, -9.9609e-01, -8.4033e-01,  ..., -3.7109e-02,
            6.6528e-02,  9.9512e-01],
          ...,
          [ 2.2754e-01, -1.9375e+00,  6.6699e-01,  ..., -1.4450e-02,
            1.2119e+00,  7.3926e-01],
          [-4.3262e-01, -2.5269e-01,  3.5797e-02,  ...,  1.3557e-02,
           -1.0098e+00,  1.3408e+00],
          [ 4.4604e-01, -4.1992e-01,  4.1821e-01,  ...,  2.3413e-01,
           -5.5957e-01, -1.1133e+00]],

         [[-1.1090e-01,  6.1230e-01, -6.0107e-01,  ..., -1.9795e+00,
            2.4780e-02,  1.0785e-01],
          [-9.0674e-01, -1.4087e-01, -1.0869e+00,  ...,  5.8740e-01,
           -1.1719e+00, -2.7979e-01],
          [ 1.1426e+00, -8.3350e-01,  1.6377e+00,  ..., -1.2280e-01,
            5.8008e-01,  9.3359e-01],
          ...,
          [ 7.6807e-01,  1.9608e-03,  3.0493e-01,  ...,  1.7773e-01,
            1.2383e+00,  2.2090e+00],
          [-1.1162e+00,  4.3408e-01,  1.3879e-01,  ...,  1.3301e+00,
            1.6143e+00,  1.6689e+00],
          [ 8.2959e-01, -1.4392e-01, -2.4707e-01,  ...,  5.9375e-01,
            7.0605e-01, -7.5537e-01]]],


        [[[ 5.5957e-01,  4.5825e-01, -3.5815e-01,  ..., -2.3206e-01,
           -1.0293e+00,  7.8125e-01],
          [-1.6687e-01,  7.3193e-01,  1.3896e+00,  ..., -1.1719e+00,
           -7.9883e-01, -1.1494e+00],
          [ 9.3689e-02, -7.2559e-01,  4.5532e-01,  ..., -1.9805e+00,
           -8.4424e-01,  6.2012e-01],
          ...,
          [-5.9662e-03, -1.2627e+00,  4.3286e-01,  ..., -4.6021e-01,
            5.0928e-01,  6.2256e-01],
          [-5.1416e-01, -3.2861e-01, -3.0688e-01,  ...,  8.9600e-01,
           -9.2236e-01, -2.7002e-01],
          [-1.0566e+00, -3.1812e-01,  9.8633e-02,  ..., -7.6562e-01,
            1.7852e+00, -7.9541e-01]],

         [[-1.4434e+00,  1.0430e+00, -1.0938e+00,  ...,  1.1660e+00,
            2.2422e+00,  4.8462e-01],
          [ 1.5576e+00, -8.8037e-01,  4.7461e-01,  ..., -1.4004e+00,
            2.4988e-01, -1.2832e+00],
          [ 5.4248e-01,  1.8347e-01, -7.0264e-01,  ..., -5.2148e-01,
           -3.9502e-01,  5.5371e-01],
          ...,
          [ 1.4580e+00,  1.0361e+00,  6.7041e-01,  ...,  2.2888e-01,
           -1.0117e+00, -1.8037e+00],
          [-7.1240e-01, -7.4609e-01,  1.0928e+00,  ..., -1.4316e+00,
           -2.5635e-01,  1.6465e+00],
          [ 1.5051e-01,  3.1274e-01,  2.0996e+00,  ...,  3.6652e-02,
            6.4062e-01,  2.1211e+00]],

         [[-1.3623e-01, -1.4346e+00, -1.9153e-01,  ...,  1.0479e+00,
           -7.0312e-02, -1.6904e+00],
          [ 1.3721e+00,  1.9165e-02, -4.4751e-01,  ..., -1.4023e+00,
           -1.8967e-02,  9.0332e-01],
          [ 7.9297e-01,  1.4658e+00, -3.4155e-01,  ..., -3.9819e-01,
            7.0801e-01, -1.2256e+00],
          ...,
          [-5.3076e-01, -1.3525e+00,  4.8096e-01,  ..., -3.5352e-01,
           -1.6345e-01,  4.5679e-01],
          [ 2.7246e-01,  4.0112e-01, -1.6904e+00,  ..., -2.9028e-01,
            1.5449e+00,  1.3977e-01],
          [-1.2422e+00,  7.9004e-01, -5.9961e-01,  ..., -2.0527e+00,
            1.8237e-01, -2.5220e-01]],

         [[-3.9185e-01, -2.2888e-01, -1.1895e+00,  ...,  7.0215e-01,
           -3.1396e-01, -3.5522e-01],
          [-7.6367e-01,  4.6582e-01, -8.5889e-01,  ...,  1.1104e+00,
           -4.3896e-01, -6.6064e-01],
          [ 1.0566e+00,  7.0618e-02,  3.1665e-01,  ..., -2.7319e-01,
            8.6133e-01,  1.0664e+00],
          ...,
          [ 1.2998e+00, -8.3691e-01,  1.7021e+00,  ...,  7.0166e-01,
           -1.6367e+00, -7.6465e-01],
          [-5.7764e-01, -7.0020e-01,  1.7432e-01,  ...,  1.4697e+00,
           -4.1016e-01,  1.0039e+00],
          [ 6.3623e-01,  8.3203e-01, -1.3464e-01,  ...,  1.3125e+00,
            9.5947e-01,  4.6582e-01]]]], device='cuda:0', dtype=torch.float16), tensor([[[[-1.7578e-01,  2.1699e+00,  7.4854e-01,  ...,  5.2051e-01,
           -4.5752e-01, -4.1040e-01],
          [ 7.8516e-01, -1.7310e-01,  6.5869e-01,  ..., -1.4473e+00,
           -6.5723e-01,  4.9341e-01],
          [ 1.4783e-01, -1.1660e+00,  1.4868e-01,  ...,  5.3174e-01,
           -8.9209e-01, -9.1162e-01],
          ...,
          [ 1.1729e+00,  7.5439e-01,  1.3047e+00,  ..., -5.7251e-02,
            5.2881e-01, -1.4259e-02],
          [-6.9971e-01,  1.8286e-01, -1.0681e-01,  ...,  1.9883e+00,
            1.4336e+00, -2.6050e-01],
          [ 5.5127e-01,  1.2772e-02,  7.4658e-01,  ..., -6.1182e-01,
            8.0322e-01, -1.7549e+00]],

         [[ 1.0479e+00,  1.2666e+00, -2.5366e-01,  ..., -1.7224e-01,
           -9.3213e-01,  1.0771e+00],
          [-1.9756e+00,  9.9268e-01,  8.6133e-01,  ..., -8.7061e-01,
           -8.6230e-01, -2.1328e+00],
          [ 2.9468e-01, -2.9834e-01, -1.2578e+00,  ..., -9.5557e-01,
            8.8965e-01, -5.5225e-01],
          ...,
          [-2.1133e+00, -7.0459e-01,  5.7910e-01,  ...,  1.0146e+00,
           -1.3398e+00, -1.6431e-01],
          [-1.5439e+00,  1.6592e+00,  1.0566e+00,  ..., -4.8047e-01,
           -5.4639e-01,  5.9473e-01],
          [ 7.8674e-02,  2.7578e+00, -1.1768e+00,  ...,  3.1860e-01,
            4.1351e-02,  2.5122e-01]],

         [[-1.6963e+00, -3.3301e-01,  3.6719e-01,  ..., -5.6348e-01,
            1.2158e+00,  8.4814e-01],
          [-1.0625e+00, -9.1406e-01,  6.3574e-01,  ...,  5.6006e-01,
           -1.5352e+00,  2.1621e+00],
          [-1.4854e+00,  6.2842e-01, -4.4189e-01,  ..., -7.7783e-01,
            9.9658e-01, -8.3691e-01],
          ...,
          [ 2.4890e-01, -1.5156e+00,  6.2500e-02,  ...,  8.9111e-02,
           -1.2373e+00,  3.7659e-02],
          [-6.6162e-01,  7.4268e-01,  1.8970e-01,  ..., -1.2646e+00,
           -9.0674e-01, -4.1138e-01],
          [-1.3770e+00,  2.2188e+00, -3.4106e-01,  ..., -6.9531e-01,
           -1.3320e+00,  9.1736e-02]],

         [[ 5.6934e-01,  1.3562e-01,  1.3818e-01,  ...,  3.2056e-01,
           -4.4385e-01,  1.8145e+00],
          [ 2.7075e-01,  9.7803e-01,  1.1487e-01,  ...,  1.5508e+00,
           -2.4341e-01, -5.5127e-01],
          [-4.8975e-01,  8.2959e-01, -9.2822e-01,  ..., -1.0527e+00,
           -1.1338e+00, -4.3335e-01],
          ...,
          [-1.7207e+00, -1.1602e+00, -9.7998e-01,  ..., -9.1504e-01,
           -3.8794e-01, -1.0205e-01],
          [ 5.9570e-01, -2.0081e-01,  8.7598e-01,  ...,  5.4736e-01,
            1.0400e+00, -1.6797e+00],
          [-1.3467e+00,  5.8887e-01,  1.9668e+00,  ...,  8.3447e-01,
           -1.0645e+00, -4.0698e-01]]],


        [[[ 4.5972e-01,  1.7676e+00,  9.1309e-01,  ..., -6.0010e-01,
            1.4675e-04, -4.4458e-01],
          [-1.0820e+00, -7.3096e-01, -4.6777e-01,  ...,  1.5605e+00,
           -6.9971e-01, -2.2876e-01],
          [-6.4648e-01,  9.4385e-01,  1.1292e-01,  ...,  2.1057e-01,
           -6.1328e-01,  2.9766e+00],
          ...,
          [ 5.8887e-01,  7.1338e-01,  1.4434e+00,  ...,  1.0967e+00,
           -1.2139e+00,  7.0557e-01],
          [ 1.1865e-01, -8.7354e-01, -6.0822e-02,  ...,  1.7981e-01,
           -2.3730e-01,  7.0166e-01],
          [ 8.0762e-01, -5.6689e-01, -1.7444e-01,  ...,  2.6636e-01,
           -6.2598e-01, -6.0449e-01]],

         [[ 3.7158e-01,  1.1383e-01, -3.1323e-01,  ...,  1.1396e+00,
           -1.9421e-01,  2.6221e-01],
          [-6.1914e-01, -1.7334e+00,  7.9346e-01,  ...,  8.4473e-01,
            5.6006e-01,  3.4497e-01],
          [ 3.9038e-01, -8.4229e-01, -1.3887e+00,  ...,  1.3994e+00,
            7.3682e-01,  8.6328e-01],
          ...,
          [ 7.3145e-01, -3.2520e-01,  3.9819e-01,  ..., -9.9854e-01,
            5.2948e-02,  1.0664e+00],
          [-6.7920e-01,  5.7812e-01, -1.1426e+00,  ...,  2.2676e+00,
            9.6338e-01, -1.5613e-01],
          [-6.2451e-01, -4.3408e-01,  1.9043e-01,  ..., -6.8506e-01,
            6.0254e-01, -8.7769e-02]],

         [[ 8.0029e-01,  1.4062e+00, -1.1074e+00,  ..., -1.1953e+00,
           -2.3132e-01, -1.0968e-01],
          [-1.3757e-01,  7.9199e-01, -2.6343e-01,  ...,  2.0947e-01,
            1.0297e-01, -1.9055e-01],
          [ 1.6533e+00,  4.4849e-01, -8.2642e-02,  ..., -5.7080e-01,
           -1.4355e+00, -7.6318e-01],
          ...,
          [-3.0060e-02,  5.6445e-01,  1.0264e+00,  ...,  1.1499e-01,
           -2.6025e-01,  1.6768e+00],
          [-6.1914e-01, -5.6201e-01,  1.2959e+00,  ..., -5.4102e-01,
           -7.8027e-01,  1.1582e+00],
          [-7.7490e-01, -2.6831e-01,  1.3232e+00,  ...,  3.0396e-01,
            6.5918e-01, -6.2500e-01]],

         [[-7.7881e-01, -1.1006e+00, -1.1699e+00,  ...,  1.6514e+00,
           -1.1064e+00,  7.5635e-01],
          [-7.3584e-01,  2.8784e-01,  6.8848e-01,  ...,  1.1689e+00,
            1.2148e+00,  2.2113e-04],
          [-9.3408e-01,  2.4841e-01,  8.1738e-01,  ...,  3.8940e-01,
           -1.0420e+00,  2.8052e-01],
          ...,
          [ 3.6084e-01, -2.6147e-01,  9.4971e-01,  ..., -5.0146e-01,
           -2.0195e+00,  4.0527e-01],
          [ 7.2949e-01,  1.1533e+00, -2.4475e-01,  ...,  1.2031e+00,
            6.4160e-01,  8.6768e-01],
          [ 1.1504e+00, -5.8350e-02, -1.0166e+00,  ..., -2.5073e-01,
           -2.9834e-01, -2.4551e+00]]]], device='cuda:0', dtype=torch.float16), tensor([[[[-1.6504,  0.7422,  0.1641,  ...,  2.1914,  1.8896, -2.5234],
          [ 0.7729,  0.2445, -0.5029,  ...,  2.5215,  1.0293, -1.5850],
          [-0.2517, -1.3340,  0.1346,  ..., -0.3223, -1.6475,  0.7090],
          ...,
          [-0.9243, -1.0488,  1.1738,  ..., -0.4255, -1.8105,  0.2074],
          [ 1.6553,  0.5649, -0.2566,  ..., -0.7725,  0.5547,  1.0850],
          [ 0.8730, -0.1299,  0.5757,  ..., -0.5127, -0.7134, -0.0875]],

         [[ 2.5957, -0.6538, -1.0029,  ..., -1.1846,  0.3379,  1.1982],
          [-0.1344, -1.3779, -0.4507,  ..., -2.1953,  0.7969, -0.6162],
          [ 0.4790,  0.1487,  0.4922,  ...,  0.4131, -1.1133,  1.8477],
          ...,
          [-0.9316, -0.4924, -0.7290,  ...,  0.0494,  0.6860,  3.3613],
          [ 0.4219, -0.3130, -0.8701,  ..., -1.7520,  0.8677,  0.9678],
          [-0.7227, -0.6245, -1.9268,  ..., -0.0322, -0.8569, -1.7148]],

         [[-0.2056, -0.3342, -0.2400,  ...,  0.0228,  1.9404, -1.2969],
          [ 0.4846,  0.0338, -0.6626,  ...,  0.5244,  1.4736, -0.3696],
          [-0.9194,  0.3132,  0.7432,  ..., -0.0458,  1.2246, -0.5054],
          ...,
          [-1.0459,  0.7080, -1.1611,  ...,  0.5557,  0.9355, -0.0980],
          [ 1.0459,  0.7290,  1.0410,  ...,  1.6309,  1.3203,  1.1426],
          [ 0.6577, -0.9009,  0.0431,  ...,  0.4414,  0.7822, -1.0596]],

         [[ 0.0640,  1.2666, -0.0655,  ..., -2.4824,  0.1224, -1.5479],
          [ 0.0494,  2.7891,  0.0605,  ..., -0.2932,  0.2380, -0.0081],
          [-1.5420, -0.7515, -0.6421,  ...,  0.9141,  0.7910,  1.0400],
          ...,
          [ 0.0831,  0.4316,  0.2251,  ..., -0.2517, -1.1299, -0.5386],
          [-0.0611,  0.0428, -0.3496,  ..., -0.2465, -0.4312,  0.0789],
          [ 0.2905,  1.0977, -0.7837,  ..., -2.0547, -1.4258, -0.0647]]],


        [[[ 0.3408, -2.1738,  0.1917,  ...,  0.0419,  0.7139, -0.0323],
          [ 1.1445, -1.4463, -1.6777,  ..., -0.3076,  0.6929, -2.5039],
          [ 0.0489, -0.7905, -0.7964,  ..., -0.3015, -0.5513,  1.1885],
          ...,
          [ 0.3604,  0.9102,  1.0771,  ..., -0.9785,  0.6860, -0.3896],
          [-0.6543,  0.8101, -0.1576,  ..., -1.7715,  0.9971, -2.1387],
          [-0.6553,  1.9863, -0.8545,  ..., -0.7451,  0.2363,  1.3330]],

         [[ 2.1836, -0.3901, -0.4858,  ..., -0.5479, -0.6724,  1.6904],
          [ 0.3455, -1.0078, -2.2500,  ..., -0.1362,  0.4175, -2.5391],
          [-0.2849, -0.9790, -0.1232,  ...,  0.7690, -0.0687,  0.6992],
          ...,
          [ 1.5488, -0.9888, -0.7573,  ..., -0.6802,  1.7705,  0.5068],
          [-0.1108,  1.5000,  0.1373,  ..., -0.7476, -0.0363, -1.0791],
          [ 0.2421,  1.4014, -0.7021,  ...,  0.3547, -0.5068, -0.7393]],

         [[ 0.2507,  0.5610, -0.3145,  ...,  0.1754,  0.6875, -1.1465],
          [ 2.3809, -0.4871, -0.3687,  ...,  0.0426,  0.3660,  0.5264],
          [ 0.5962,  0.0942,  0.3777,  ...,  1.1758,  1.0400, -0.9888],
          ...,
          [-0.0923,  1.1201, -0.6548,  ...,  1.0742,  2.3008, -0.0726],
          [ 0.5532,  0.7227,  0.7153,  ..., -0.6240,  0.0735, -0.2072],
          [-0.0040,  1.7100,  0.3816,  ..., -0.8926, -1.1201,  0.3584]],

         [[ 0.6475,  0.0412, -0.1018,  ..., -1.2344, -0.8154,  1.0732],
          [-1.5967,  0.4148, -0.4739,  ...,  1.2676,  0.1183,  0.3384],
          [ 0.6641,  0.4724,  0.0482,  ...,  2.4902,  0.8418,  0.8423],
          ...,
          [ 0.3525,  0.5088,  1.2031,  ..., -0.1844,  0.7842,  0.9062],
          [-0.1414, -0.3110,  0.0466,  ...,  1.2637, -0.1755, -0.0563],
          [ 1.0723, -2.1602,  0.7920,  ...,  1.0166,  0.1201,  1.6133]]]],
       device='cuda:0', dtype=torch.float16), None, 0.0, True, None

=== Checking PyTorch Backend Registration ===
No AOTriton attention ops found in PyTorch registry

=== Relevant Environment Variables ===
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: 1
PYTORCH_ROCM_ARCH: gfx1151
HSA_OVERRIDE_GFX_VERSION: Not set
ROCM_ARCH: Not set
HIP_VISIBLE_DEVICES: Not set
PYTORCH_ROCM_AOTRITON_PREFER_DEFAULT: Not set

=== Debugging PyTorch SDPA Selection ===
SDPA execution completed

=== Architecture Detection ===
Device properties: _CudaDeviceProperties(name='AMD Radeon Graphics', major=11, minor=5, gcnArchName='gfx1151', total_memory=104906MB, multi_processor_count=20, uuid=58580000-0000-0000-0000-000000000000, pci_bus_id=194, pci_device_id=0, pci_domain_id=0, L2_cache_size=2MB)
rocminfo: Name:                    gfx1151
rocminfo: Name:                    amdgcn-amd-amdhsa--gfx1151
rocminfo: Name:                    amdgcn-amd-amdhsa--gfx11-generic
```

```
# python 11-test_manual_aotriton.py
=== Manual AOTriton Test ===
AOTriton functions:
v2 module: ['CppTune', 'CppTuneSpecialKernelIndex', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'flash', 'kDefault', 'kSkipGPUCall']
flash module: ['BwdExtraArguments', 'FusedBwdExtraArguments', 'FwdExtraArguments', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'attn_bw
d', 'attn_bwd_compact_varlen', 'attn_bwd_fused', 'attn_fwd', 'attn_fwd_compact_varlen', 'check_gpu', 'debug_fill_dropout_rng', 'debug_fill_dropout_rng_tens
or', 'debug_simulate_encoded_softmax']
Successfully imported attention functions!

Testing AOTriton attention...
AOTriton call failed: no signature found for builtin <built-in method attn_fwd of PyCapsule object at 0x7fa509606840>

=== PyTorch Integration Check ===
torch.ops.aotriton exists
Available ops: ['__doc__', '__loader__', '__name__', '__package__', '__spec__', '_dir', 'name']

=== Manual Registration Check ===
```

```
# python 14-simple_aotriton_test.py
=== Simple AOTriton Test ===
Error creating AOTriton types: __init__(): incompatible constructor arguments. The following argument types are supported:
    1. pyaotriton.HipMemory()

Invoked with: kwargs: ptr=140181988442112, size=2048

=== Checking for compiled kernels ===
Error checking GPU: check_gpu(): incompatible function arguments. The following argument types are supported:
    1. (stream: pyaotriton.Stream) -> pyaotriton.hipError_t

Invoked with:

=== Environment Check ===
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