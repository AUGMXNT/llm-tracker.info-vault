For the latest Strix Halo / AMD RyzenAI Max with Radeon 8060S (`gfx1151`) support, check out:
- 2025-05-02 https://github.com/ROCm/TheRock/discussions/244
- 2025-05-02 https://github.com/ROCm/ROCm/issues/4499
	- https://github.com/ROCm/ROCm/issues/4566

# Checklist
- [ ] ROCm
- [x] llama.cpp
	- [x] HIP (ROCm)
	- [x] Vulkan
	- [x] RPC
- [x] mamf-finder
- [x] PyTorch
	- [ ] hipBLASlt
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


# llama.cpp

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