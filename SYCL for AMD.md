# Install

## Intel oneAPI Base Toolkit
Download offline installer
```
# from "Installation from the Command Line" Section
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh

# will install in /opt/intel (otherwise ~/intel if not sudo)
sudo sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --silent --cli --eula accept
```
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline

## oneAPI for AMD
```
wget https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd
sudo sh oneapi-for-amd-gpus-2025.0.0-rocm-6.1.0-linux.sh
```
- https://developer.codeplay.com/products/oneapi/amd/download

## oneMKL for AMD
```
# HIPTARGET
export HIPTARGET=gfx1100

# oneMKL
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
# Find your HIPTARGET with rocminfo, under the key 'Name:'
cmake -B buildWithrocBLAS -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_ROCBLAS_BACKEND=ON -DHIP_TARGETS=${HIPTARGET} -DTARGET_DOMAINS=blas
cmake --build buildWithrocBLAS --config Release
```
- https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md#linux
- Note: there is a typo in the llama.cpp docs and you need `-DHIP_TARGETS` not `-DHIPTARGETS`

If it works you should see something like:
```
source /opt/intel/oneapi/setvars.sh
sycl-ls

[opencl:cpu][opencl:0] Intel(R) OpenCL, AMD EPYC 9274F 24-Core Processor                OpenCL 3.0 (Build 0) [2024.18.10.0.08_160000]
[hip:gpu][hip:0] AMD HIP BACKEND, AMD Radeon Pro W7900 gfx1100 [HIP 60342.13]
```

## llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp llama.cpp-sycl
cd llama.cpp-sycl

# Export relevant ENV variables
export LD_LIBRARY_PATH=~/ai/oneMKL/buildWithrocBLAS/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=~/ai/oneMKL/buildWithrocBLAS/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=~/ai/oneMKL/buildWithrocBLAS/include:$CPLUS_INCLUDE_DIR

# Build LLAMA with rocBLAS acceleration through SYCL

## AMD
# Use FP32, FP16 is not supported
# Find your GGML_SYCL_DEVICE_ARCH with rocminfo, under the key 'Name:'
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=AMD -DGGML_SYCL_DEVICE_ARCH=gfx1100 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build all binary
cmake --build build --config Release -j -v

```



# Benchmarks

## Vulkan
```
# For AMD:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf

# For NVIDIA:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
```

Results
```
🐟 ❯ VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro W7900 (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | warp size: 64 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
ggml_vulkan: Compiling shaders..........................Done!
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         pp512 |       1813.49 ± 7.09 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         tg128 |        112.75 ± 0.81 |

build: ba1cb19c (4327)
```

ROCm
```
🐟 ❯ ~/ai/llama.cpp/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |      2875.28 ± 23.18 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |         99.42 ± 0.08 |

build: ba1cb19c (4327)
```

SYCL
```

```