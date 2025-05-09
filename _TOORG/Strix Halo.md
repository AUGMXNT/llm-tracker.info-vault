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
	- [ ] wmma
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
## mamf-finder
Using my [mamf-finder](https://github.com/shisa-ai/mamf-finder) repo to test, it takes about just under 35 hours to test with mamf-finder:

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

Performance bug?
https://github.com/ROCm/MIOpen/pull/3685

```
# TORCH_BLAS_PREFER_HIPBLASLT=0 
HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt/library TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python examples/benchmark.py
```

## aotriton
We need to build and install aotriton
```
mkdir -p /share/libdrm
cp /home/lhl/amdgpu.ids /share/libdrm/
dnf install gcc gcc-c++ make cmake
dnf install python3-devel
export HIP_PLATFORM=amd
export GPU_TRAGETS=gfx1151
git clone https://github.com/ROCm/aotriton
cd aotriton
git submodule sync && git submodule update --init --recursive --force
mkdir build && cd build

cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -G Ninja
ninja install

# ln -s /home/lhl/aotriton/build/install_dir/lib/pyaotriton.cpython-313-x86_64-linux-gnu.so /usr/local/lib/python3.13/site-packages/

python -c 'import pyaotriton'
```


# llama.cpp
## Vulkan vs HIP
2025-05-03: Currently, the Vulkan backend is significantly faster than the HIP/ROCm backend on every single `llama-bench` tested model.

## HIP Instability
Besides poor performance, I get frequent GPU hangs / core dumps with `6.15.0-0.rc3.20250422gita33b5a08cbbd.29.fc43.x86_64`:

```
May 02 13:01:34 cluster1 systemd-coredump[2426713]: [🡕] Process 2426590 (llama-bench) of user 1002 dumped core.

Module [dso] without build-id.
Module [dso] without build-id.
Module [dso] without build-id.
Module [dso] without build-id.
Module [dso] without build-id.
Module [dso] without build-id.
Module libdrm.so.2 from rpm libdrm-2.4.124-2.fc42.x86_64
Module libelf.so.1 from rpm elfutils-0.192-8.fc42.x86_64
Module libdrm_amdgpu.so.1 from rpm libdrm-2.4.124-2.fc42.x86_64
Module libkeyutils.so.1 without build-id.
Module libkrb5support.so.0 without build-id.
Module libcom_err.so.3 without build-id.
Module libk5crypto.so.3 without build-id.
Module libkrb5.so.3 without build-id.
Module libnuma.so.1 from rpm numactl-2.0.19-2.fc42.x86_64
Module libz.so.1 without build-id.
Module libzstd.so.1 without build-id.
Module libgssapi_krb5.so.2 without build-id.
Module libcrypto.so.3 without build-id.
Module libssl.so.3 without build-id.
Module libssh2.so.1 without build-id.
Module libnghttp2.so.14 without build-id.
Module libgcc_s.so.1 without build-id.
Module libstdc++.so.6 without build-id.
Module libcurl.so.4 without build-id.

Stack trace of thread 2426591:
#0  0x00007f5b0b9b5bec __pthread_kill_implementation (libc.so.6 + 0x73bec)
#1  0x00007f5b0b95babe raise (libc.so.6 + 0x19abe)
#2  0x00007f5b0b9436d0 abort (libc.so.6 + 0x16d0)
#3  0x00007f5af2612a06 _ZN4rocr4core7Runtime18HwExceptionHandlerElPv.cold (libhsa-runtime64.so.1 + 0x12a06)
#4  0x00007f5af267dfab _ZN4rocr4core7Runtime15AsyncEventsLoopEPv (libhsa-runtime64.so.1 + 0x7dfab)
#5  0x00007f5af261be9c _ZN4rocr2os16ThreadTrampolineEPv (libhsa-runtime64.so.1 + 0x1be9c)
#6  0x00007f5b0b9b3c84 start_thread (libc.so.6 + 0x71c84)
#7  0x00007f5b0ba3612c __clone3 (libc.so.6 + 0xf412c)

Stack trace of thread 2426660:
#0  0x00007f5b0ba31eed ioctl (libc.so.6 + 0xefeed)
#1  0x00007f5af26f43d0 hsakmt_ioctl (libhsa-runtime64.so.1 + 0xf43d0)
#2  0x00007f5af26f4b90 hsaKmtWaitOnMultipleEvents_Ext.part.0 (libhsa-runtime64.so.1 + 0xf4b90)
#3  0x00007f5af267e64d _ZN4rocr4core7Runtime15AsyncEventsLoopEPv (libhsa-runtime64.so.1 + 0x7e64d)
#4  0x00007f5af261be9c _ZN4rocr2os16ThreadTrampolineEPv (libhsa-runtime64.so.1 + 0x1be9c)
#5  0x00007f5b0b9b3c84 start_thread (libc.so.6 + 0x71c84)
#6  0x00007f5b0ba3612c __clone3 (libc.so.6 + 0xf412c)

Stack trace of thread 2426590:
#0  0x00007f5b0b9c4b87 free (libc.so.6 + 0x82b87)
#1  0x00007f5af54f189a _ZNSt6vectorISt10unique_ptrIA_cSt14default_deleteIS1_EESaIS4_EE12emplace_backIJS4_EEERS4_DpOT_ (libamd_comgr.so.2 + 0x2af189a)
#2  0x00007f5af54f1675 _ZN4llvm7msgpack8Document9addStringENS_9StringRefE (libamd_comgr.so.2 + 0x2af1675)
#3  0x00007f5af54ee655 _ZN5COMGR8metadata14getIsaMetadataEN4llvm9StringRefERNS1_7msgpack8DocumentE (libamd_comgr.so.2 + 0x2aee655)
#4  0x00007f5af54cf222 amd_comgr_get_isa_metadata (libamd_comgr.so.2 + 0x2acf222)
#5  0x00007f5b02c13dc3 _ZN3amd6device6Kernel20SetAvailableSgprVgprEv (libamdhip64.so.6 + 0x413dc3)
#6  0x00007f5b02c87e57 _ZN3amd3roc15LightningKernel8postLoadEv (libamdhip64.so.6 + 0x487e57)
#7  0x00007f5b02c84365 _ZN3amd3roc16LightningProgram10setKernelsEPvmimNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE (libamdhip64.so.6 + 0x484365)
#8  0x00007f5b02c033f2 _ZN3amd6device7Program6loadLCEv (libamdhip64.so.6 + 0x4033f2)
#9  0x00007f5b02c4e12a _ZN3amd7Program4loadERKSt6vectorIPNS_6DeviceESaIS3_EE (libamdhip64.so.6 + 0x44e12a)
#10 0x00007f5b029835dd _ZN3hip13FatBinaryInfo12BuildProgramEi (libamdhip64.so.6 + 0x1835dd)
#11 0x00007f5b0298822e _ZN3hip8Function11getStatFuncEPP18ihipModuleSymbol_ti (libamdhip64.so.6 + 0x18822e)
#12 0x00007f5b02925ca0 _ZN3hip6StatCO11getStatFuncEPP18ihipModuleSymbol_tPKvi (libamdhip64.so.6 + 0x125ca0)
#13 0x00007f5b02b227bb _ZN3hip16ihipLaunchKernelEPKv4dim3S2_PPvmP12ihipStream_tP11ihipEvent_tS8_i (libamdhip64.so.6 + 0x3227bb)
#14 0x00007f5b02af7c0a _ZN3hip22hipLaunchKernel_commonEPKv4dim3S2_PPvmP12ihipStream_t (libamdhip64.so.6 + 0x2f7c0a)
#15 0x00007f5b02af8278 _ZN3hip15hipLaunchKernelEPKv4dim3S2_PPvmP12ihipStream_t (libamdhip64.so.6 + 0x2f8278)
#16 0x00007f5b0c0b1f85 _Z13ggml_cuda_cpyR25ggml_backend_cuda_contextPK11ggml_tensorPS1_b (libggml-hip.so + 0xb1f85)
#17 0x00007f5b0c0daffd _ZL31ggml_backend_cuda_graph_computeP12ggml_backendP11ggml_cgraph (libggml-hip.so + 0xdaffd)
#18 0x00007f5b0bf2e110 ggml_backend_sched_graph_compute_async (libggml-base.so + 0x18110)
#19 0x00007f5b0e643ea0 _ZN13llama_context13graph_computeEP11ggml_cgraphb (libllama.so + 0x2aea0)
#20 0x00007f5b0e648a58 _ZN13llama_context6decodeER11llama_batch (libllama.so + 0x2fa58)
#21 0x00007f5b0e649c7e llama_decode (libllama.so + 0x30c7e)
#22 0x0000000000408f5b _ZL11test_promptP13llama_contextiii (/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench + 0x8f5b)
#23 0x0000000000406d7d main (/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench + 0x6d7d)
#24 0x00007f5b0b9455b5 __libc_start_call_main (libc.so.6 + 0x35b5)
#25 0x00007f5b0b945668 __libc_start_main@@GLIBC_2.34 (libc.so.6 + 0x3668)
#26 0x0000000000407ee5 _start (/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench + 0x7ee5)
ELF object binary architecture: AMD x86-64
```


### Qwen 3 MoE
Currently there is a bug where batch size has to be below 360 to prevent a crash. 256 has been tested as the best performer:
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

## RPC
Build llama.cpp-hip w/ RPC
```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release     && cmake --build build --config Release -- -j 32

cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON && cmake --build build --config Release -j 32
```

When running `llama-cli` by default it will add it to nodes. When using `llama-bench` it does not and you should run an RPC server locally.


Vulkan has less default memory available than ROCm!

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

## Speculative Decode
https://github.com/ggml-org/llama.cpp/issues/12968
https://github.com/hjc4869/llama.cpp
https://x.com/hjc4869/status/1913562550064799896

RDNA3 gets a sizable performance uplift with speculative decoding on 4bit models (--draft-max 3 --draft-min 3), and you'll most likely get 8-12 t/s for a 70-72B dense model.

Sweep
https://github.com/AUGMXNT/speed-benchmarking/tree/main/llama.cpp-code

# Voicechat
https://github.com/AUGMXNT/speed-benchmarking


# Building PyTorch
With AOTriton and FA2

```
mamba activate
mamba install cmake ninja patchelf
pip install uv
uv pip install meson

git clone https://github.com/scottt/rocm-TheRock
python ./build_tools/fetch_sources.py
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1151
```


## Docker Files
- We run our commands from the rocm-TheRock repo root otherwise relative paths are broken

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