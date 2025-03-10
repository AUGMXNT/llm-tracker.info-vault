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
- Note: there is a typo in the llama.cpp docs and you need `-DHIP_TARGETS` not `-DHIPTARGETS` (fPR submitted)

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
export 
export MKL_ROCBLAS=~/ai/oneMKL/buildWithrocmBLAS/lib
export LD_LIBRARY_PATH=$MKL_ROCBLAS:$LD_LIBRARY_PATH
export LIBRARY_PATH=$MKL_ROCBLAS:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=$MKL_ROCBLAS:$CPLUS_INCLUDE_DIR

# Build LLAMA with rocBLAS acceleration through SYCL

## AMD
# Use FP32, FP16 is not supported
# Find your GGML_SYCL_DEVICE_ARCH with rocminfo, under the key 'Name:'
export GGML_SYCL_DEVICE_ARCH=gfx1100 # Example architecture
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=AMD -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_SHARED_LINKER_FLAGS="-L$MKL_ROCBLAS -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lonemath_blas_rocblas -Wl,--no-as-needed -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core"

# build all binary
cmake --build build --config Release -j -v

```
- CMake modification required for AMD (PR submitted)
- `-DCMAKE_SHARED_LINKER_FLAGS` required (PR submitted)

Fixes for building successfully:
https://github.com/lhl/llama.cpp-sycl-amd/commit/5cb6209de5379411a28d94ebea3fe1abaac3d26b
https://github.com/ggerganov/llama.cpp/pull/10851

Potentially can build with `-DBUILD_SHARED_LIBS=OFF` so that loading up oneAPI envs isn't necessary? (might still need device env vars though)

# Benchmarks

## SYCL
- `llama-cli` runs ok!
- `llama-bench` segfaults! https://github.com/ggerganov/llama.cpp/issues/10850
- `llama-server` fails w/ memory allocation error with `--ngl`?
```
oneapi::mkl::oneapi::mkl::blas::gemm: cannot allocate memory on host
Exception caught at file:/home/lhl/github/lhl/llama.cpp-sycl-amd/ggml/src/ggml-sycl/ggml-sycl.cpp, line:3356, func:operator()
SYCL error: CHECK_TRY_ERROR(dpct::gemm_batch( *main_stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha, (const char *)src0_as_f16, dpct::library_data_t::real_half, nb01 / nb00, nb02 / nb00, (const char *)src1_f16, dpct::library_data_t::real_half, nb11 / nb10, nb12 / nb10, beta, (char *)dst_t, cu_data_type, ne01, nb2 / nb0, ne12 * ne13, cu_compute_type)): Meet error in this line code!
```


## ROCm
### llama-bench - llama2-7b-q4_0
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


### Llama 3 70B Q4_K_M

#### w/ draft model, q8 kvcache

Server:
```
build/bin/llama-server -m /models/gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.8 -ngl 99 -ngld 99 -c 8000 -cd 8000 -ctk q8_0 -ctv q8_0 -fa
```

Benchmark:
```
python benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.3" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --tokenizer meta-llama/Llama-3.3-70B-Instruct
```

Results:
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  2163.94   
Total input tokens:                      14688     
Total generated tokens:                  35891     
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         16.59     
Total Token throughput (tok/s):          23.37     
---------------Time to First Token----------------
Mean TTFT (ms):                          1202.44   
Median TTFT (ms):                        967.68    
P99 TTFT (ms):                           3677.83   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          66.68     
Median TPOT (ms):                        63.17     
P99 TPOT (ms):                           190.02    
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.10     
Median ITL (ms):                         0.07      
P99 ITL (ms):                            355.21    
==================================================
```
#### w/ draft model (no FA)
```
build/bin/llama-server -m /models/gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.8 -ngl 99 -ngld 99 -c 6000 -cd 6000
```
- 8000 OOMs
- q8 kvcache OOMs forever

```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  2009.77   
Total input tokens:                      14688     
Total generated tokens:                  35455     
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         17.64     
Total Token throughput (tok/s):          24.95     
---------------Time to First Token----------------
Mean TTFT (ms):                          1080.25   
Median TTFT (ms):                        962.51    
P99 TTFT (ms):                           3103.85   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          63.17     
Median TPOT (ms):                        61.55     
P99 TPOT (ms):                           181.66    
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.67     
Median ITL (ms):                         0.17      
P99 ITL (ms):                            312.86    
==================================================
```
#### hjc4869 w/ draft model, q8 kvcache
```
build/bin/llama-server -m /models/gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.8 -ngl 99 -ngld 99 -c 12000 -cd 12000 -ctk q8_0 -ctv q8_0 -fa
```
- seems to be able to fit a lot more context than upstream
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  2095.09   
Total input tokens:                      14688     
Total generated tokens:                  35749     
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         17.06     
Total Token throughput (tok/s):          24.07     
---------------Time to First Token----------------
Mean TTFT (ms):                          1085.26   
Median TTFT (ms):                        963.65    
P99 TTFT (ms):                           3112.18   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          65.42     
Median TPOT (ms):                        63.96     
P99 TPOT (ms):                           189.01    
---------------Inter-token Latency----------------
Mean ITL (ms):                           56.63     
Median ITL (ms):                         0.14      
P99 ITL (ms):                            332.08    
==================================================
```
## Vulkan

How to run:
```
# For AMD:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf

# For NVIDIA:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
```
### llama-bench - llama2-7b-q4_0
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ~/ai/llama.cpp-vulkan/build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro W7900 (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | warp size: 64 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
ggml_vulkan: Compiling shaders..........................Done!
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         pp512 |       1813.49 ± 7.09 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |         tg128 |        112.75 ± 0.81 |

build: ba1cb19c (4327)
```

### Llama 3 70B Q4_K_M

#### w/ draft model
Server
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json build/bin/llama-server -m /models/gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf -md /models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --draft-max 16 --draft-min 1 --draft-p-min 0.8 -ngl 99 -ngld 99 -c 8000 -cd 8000 -fa
```
- 8b kvcache doesn't work

Results
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  5638.62   
Total input tokens:                      14688     
Total generated tokens:                  36721     
Request throughput (req/s):              0.01      
Output token throughput (tok/s):         6.51      
Total Token throughput (tok/s):          9.12      
---------------Time to First Token----------------
Mean TTFT (ms):                          4335.36   
Median TTFT (ms):                        1830.94   
P99 TTFT (ms):                           14857.58  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          157.80    
Median TPOT (ms):                        147.69    
P99 TPOT (ms):                           460.64    
---------------Inter-token Latency----------------
Mean ITL (ms):                           145.59    
Median ITL (ms):                         0.03      
P99 ITL (ms):                            1264.12   
==================================================
```
#### w/ draft model (no FA)
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  3830.46   
Total input tokens:                      14688     
Total generated tokens:                  36303     
Request throughput (req/s):              0.02      
Output token throughput (tok/s):         9.48      
Total Token throughput (tok/s):          13.31     
---------------Time to First Token----------------
Mean TTFT (ms):                          2212.03   
Median TTFT (ms):                        1218.89   
P99 TTFT (ms):                           6633.29   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          118.94    
Median TPOT (ms):                        113.34    
P99 TPOT (ms):                           369.24    
---------------Inter-token Latency----------------
Mean ITL (ms):                           101.43    
Median ITL (ms):                         0.07      
P99 ITL (ms):                            689.22    
==================================================
```

## vLLM
```
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
- If you don't want to build your own you can use the image from the EmbeddedLM article: `ghcr.io/embeddedllm/vllm-rocm:navi-gguf-690c57c`

```
# 6400 tokens max
vllm serve /app/model/gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf --served_model_name llama3.3 --gpu-memory-utilization 0.98 --max-model-len 6000
```

Note, loading times are a bit crazy:
```
# start
INFO 12-16 16:11:46 api_server.py:625] vLLM API server version 0.6.4.post2.dev258+gf13cf9ad

# loaded model - 1m 20s (43.727/44.984 GB)
INFO 12-16 16:13:06 model_runner.py:1094] Loading model weights took 39.6686 GB

# 12m 44s for memory profiling
INFO 12-16 16:25:51 worker.py:237] Memory profiling results: duration=764.60 seconds, total_gpu_memory=44.98GiB, initial_memory_usage=39.91GiB, peak_torch_memory=41.23GiB, memory_usage_post_profile=40.01GiB, non_torch_memory=0.34GiB, kv_cache_size=2.52GiB, gpu_memory_utilization=0.98.
INFO 12-16 16:25:51 gpu_executor.py:76] # GPU blocks: 516, # CPU blocks: 819
INFO 12-16 16:25:51 gpu_executor.py:80] Maximum concurrency for 6000 tokens per request: 1.38x

# 24m 12s graph capture ; 36m 56s total
INFO 12-16 16:50:04 model_runner.py:1523] Graph capturing finished in 1452 secs, took 0.17 GiB
INFO 12-16 16:50:04 llm_engine.py:493] init engine (profile, create kv cache, warmup model) took 2217.68 seconds

# Took 38m 18s for this to load...
INFO 12-16 16:50:04 api_server.py:560] Using supplied chat template:
INFO 12-16 16:50:04 api_server.py:560] None
INFO 12-16 16:50:04 launcher.py:19] Available routes are:
INFO 12-16 16:50:04 launcher.py:27] Route: /openapi.json, Methods: GET, HEAD
INFO 12-16 16:50:04 launcher.py:27] Route: /docs, Methods: GET, HEAD
INFO 12-16 16:50:04 launcher.py:27] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 12-16 16:50:04 launcher.py:27] Route: /redoc, Methods: GET, HEAD
INFO 12-16 16:50:04 launcher.py:27] Route: /health, Methods: GET
INFO 12-16 16:50:04 launcher.py:27] Route: /tokenize, Methods: POST
INFO 12-16 16:50:04 launcher.py:27] Route: /detokenize, Methods: POST
INFO 12-16 16:50:04 launcher.py:27] Route: /v1/models, Methods: GET
INFO 12-16 16:50:04 launcher.py:27] Route: /version, Methods: GET
INFO 12-16 16:50:04 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 12-16 16:50:04 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 12-16 16:50:04 launcher.py:27] Route: /v1/embeddings, Methods: POST
INFO 12-16 16:50:04 launcher.py:27] Route: /v1/score, Methods: POST
INFO:     Started server process [2224]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Tough results:
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  4349.64   
Total input tokens:                      14688     
Total generated tokens:                  12426     
Request throughput (req/s):              0.01      
Output token throughput (tok/s):         2.86      
Total Token throughput (tok/s):          6.23      
---------------Time to First Token----------------
Mean TTFT (ms):                          41185.18  
Median TTFT (ms):                        19905.55  
P99 TTFT (ms):                           120130.56 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          141.39    
Median TPOT (ms):                        138.64    
P99 TPOT (ms):                           216.53    
---------------Inter-token Latency----------------
Mean ITL (ms):                           138.51    
Median ITL (ms):                         138.80    
P99 ITL (ms):                            141.77    
==================================================
```