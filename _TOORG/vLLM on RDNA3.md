For my main AMD GPU docs, see: [[AMD GPUs]]
# vLLM 2025-02-27
## Build Docker
```
sudo DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="rocm/vllm-dev:navi_base" -f Dockerfile.rocm -t vllm-rocm .
```
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=rocm
- Docker - https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=rocm#build-image-from-source

## Run
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
   vllm-rocm \
   bash
```

# vLLM 2024-12-7
See: https://embeddedllm.com/blog/vllm-now-supports-running-gguf-on-amd-radeon-gpu

Build
```
paru -S docker docker-compose docker-buildx

git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 sudo docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
```

You can check on the model here:
```
sudo docker images
```

Run the docker instance (mount your HF and models folder)
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

Note: this docker image does not support hipBLASLt for `gfx1100` and falls back to hipBLAS
# Benchmarks

Testing on an AMD W7900 w/ Docker build: `0.6.4.post2.dev258+gf13cf9ad`

Successful requests:                     32        
Benchmark duration (s):                  347.96    
Total input tokens:                      6449      
Total generated tokens:                  16216     
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         46.60     
Total Token throughput (tok/s):          65.14     
---------------Time to First Token----------------
Mean TTFT (ms):                          160.39    
Median TTFT (ms):                        148.70    
P99 TTFT (ms):                           303.35    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.31     
Median TPOT (ms):                        18.47     
P99 TPOT (ms):                           27.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.18     
Median ITL (ms):                         19.80     
P99 ITL (ms):                            38.79  

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
| Mean TPOT (ms)                  | 48.34     | 55.95     | 18.97       | **14.81**        | 19.31            | 10.05       | 14.14            |
| Median TPOT (ms)                | 46.94     | 55.21     | 18.56       | **14.77**        | 18.47            | 9.62        | 14.02            |
| P99 TPOT (ms)                   | 78.78     | 73.44     | 28.75       | **15.88**        | 27.35            | **15.46**   | **15.27**        |
| Mean ITL (ms)                   | 46.99     | 55.20     | 18.60       | 15.03            | 21.18            | **10.10**   | 14.38            |
| Median ITL (ms)                 | 46.99     | 55.20     | 18.63       | 14.96            | 19.80            | **9.91**    | 14.43            |
| P99 ITL (ms)                    | 48.35     | 56.56     | 19.43       | 16.47            | 38.79            | **12.68**   | 15.75            |
- vLLM FP8 does not run on RDNA3 
- vLLM bitsandbytes quantization does not run w/ ROCm (multifactor-backend bnb installed) 
- llama.cpp ROCm backend b4276 (HEAD)
- ExLlamaV2 0.2.6 (HEAD)
- MLC nightly 0.18.dev249

## System Details
More system info:
```
PyTorch version: 2.6.0.dev20241113+rocm6.2
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.2.0 24292 26466ce804ac523b398608f17388eb6d605a3f09)
CMake version: version 3.26.4
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.12.1-2-cachyos-x86_64-with-glibc2.31
GPU models and configuration: AMD Radeon PRO W7900 (gfx1100)
HIP runtime version: 6.2.41133
MIOpen runtime version: 3.2.0
Is XNNPACK available: True

CPU:
Model name:                           AMD EPYC 9274F 24-Core CPU MHz:                              4299.904
CPU max MHz:                          4303.1250
CPU min MHz:                          1500.0000

Versions of relevant libraries:
[pip3] mypy==1.8.0
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.4
[pip3] optree==0.9.1
[pip3] pynvml==11.5.3
[pip3] pytorch-triton-rocm==3.1.0+cf34004b8a
[pip3] pyzmq==26.2.0
[pip3] torch==2.6.0.dev20241113+rocm6.2
[pip3] torchvision==0.20.0.dev20241113+rocm6.2
[pip3] transformers==4.47.0
[pip3] triton==3.1.0
ROCM Version: 6.2.41133-dd7f95766
vLLM Version: 0.6.4.post2.dev258+gf13cf9ad
PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda
PYTORCH_TEST_WITH_ROCM=1
PYTORCH_ROCM_ARCH=gfx908;gfx90a;gfx942;gfx1100
MAX_JOBS=32
LD_LIBRARY_PATH=/opt/conda/envs/py_3.9/lib/python3.9/site-packages/cv2/../../lib64:/opt/ompi/lib:/opt/rocm/lib:/usr/local/lib::/opt/rocm/lib/:/libtorch/lib:
VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_MODULE_LOADING=LAZY
```

## vLLM
### Full
Run server:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --num-scheduler-step 1 --served_model_name llama3.1-8b
```
- 15GB of weights
- `INFO 12-07 17:39:09 worker.py:237] Memory profiling results: duration=39.11 seconds, total_gpu_memory=44.98GiB, initial_memory_usage=15.23GiB, peak_torch_memory=16.20GiB, memory_usage_post_profile=15.46GiB, non_torch_memory=0.44GiB, kv_cache_size=23.84GiB, gpu_memory_utilization=0.90.`
- `INFO 12-07 17:38:16 config.py:1066] Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.`
- no speed difference w/ `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` vs not

Run benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct`
```

Results:
```
============ Serving Benchmark Result ============
Successful requests:                     32        
Benchmark duration (s):                  311.26    
Total input tokens:                      6449      
Total generated tokens:                  6544      
Request throughput (req/s):              0.10      
Output token throughput (tok/s):         21.02     
Total Token throughput (tok/s):          41.74     
---------------Time to First Token----------------
Mean TTFT (ms):                          159.58    
Median TTFT (ms):                        111.76    
P99 TTFT (ms):                           358.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          48.34     
Median TPOT (ms):                        46.94     
P99 TPOT (ms):                           78.78     
---------------Inter-token Latency----------------
Mean ITL (ms):                           46.99     
Median ITL (ms):                         46.99     
P99 ITL (ms):                            48.35     
==================================================
```

### FP8
Dynamic doesn't work, Wah wah...
```
ERROR 12-07 14:21:53 engine.py:366] RuntimeError: Error in model execution (input dumped to /tmp/err_execute_model_input_20241207-142153.pkl): torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+
```

But let's try some baked ones... https://github.com/vllm-project/vllm/blob/main/docs/source/quantization/fp8.rst
https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8

Serve:
```
root@epyc:/vllm-workspace# vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --num-scheduler-step 1 --served_model_name llama3.1-8b
```

Nope:
```
RuntimeError: Error in model execution (input dumped to /tmp/err_execute_model_input_20241207-154119.pkl): torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+
```

### INT8 (W8A8)
Using: https://github.com/vllm-project/llm-compressor
Took: 55m to convert Llama 3.1 8B on W7900


Serve:
```
vllm serve Llama-3.1-8B-Instruct-INT8 --num-scheduler-step 1 --served_model_name llama3.1-8b
```
- 8.5GB weights

Benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Results:
```
============ Serving Benchmark Result ============
Successful requests:                     32        
Benchmark duration (s):                  367.50    
Total input tokens:                      6449      
Total generated tokens:                  6552      
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         17.83     
Total Token throughput (tok/s):          35.38     
---------------Time to First Token----------------
Mean TTFT (ms):                          232.78    
Median TTFT (ms):                        162.86    
P99 TTFT (ms):                           477.17    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          55.95     
Median TPOT (ms):                        55.21     
P99 TPOT (ms):                           73.44     
---------------Inter-token Latency----------------
Mean ITL (ms):                           55.20     
Median ITL (ms):                         55.20     
P99 ITL (ms):                            56.56     
==================================================
```

### bitsandbytes
Install:
```
# triton.ops needs 3.1.0 not 3.0.0
pip install -U triton
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' --no-deps
```
Serve:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --num-scheduler-step 1 --served_model_name llama3.1-8b -q bitsandbytes --load_format bitsandbytes
```

```
ERROR 12-07 15:51:19 engine.py:366] ValueError: bitsandbytes quantization is currently not supported in rocm.
```


### Q5_K_M
Run Server
```
vllm serve /app/model/gguf/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --num-scheduler-step 1 --served_model_name llama3.1-8b
```
- 5GB of weights

Run benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct`
```

Results:
```
============ Serving Benchmark Result ============
Successful requests:                     32        
Benchmark duration (s):                  125.00    
Total input tokens:                      6449      
Total generated tokens:                  6183      
Request throughput (req/s):              0.26      
Output token throughput (tok/s):         49.46     
Total Token throughput (tok/s):          101.06    
---------------Time to First Token----------------
Mean TTFT (ms):                          327.56    
Median TTFT (ms):                        128.24    
P99 TTFT (ms):                           2911.16   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.97     
Median TPOT (ms):                        18.56     
P99 TPOT (ms):                           28.75     
---------------Inter-token Latency----------------
Mean ITL (ms):                           18.60     
Median ITL (ms):                         18.63     
P99 ITL (ms):                            19.43     
==================================================
```

## llama.cpp Comparion
### llama-bench
```
~/ai/llama.cpp/build/bin/llama-bench -m /models/gguf/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 8B Q5_K - Medium         |   5.33 GiB |     8.03 B | ROCm       |  99 |         pp512 |       2494.45 ± 6.23 |
| llama 8B Q5_K - Medium         |   5.33 GiB |     8.03 B | ROCm       |  99 |         tg128 |         72.42 ± 0.13 |

build: f162d45a (4276)
```
### benchmark_serving.py

Run server:
```
~/ai/llama.cpp/build/bin/llama-server -m /models/gguf/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf -ngl 99 -a 'llama3.1-8b' -sp
```
- `-sp` special token output required

Run benchmark:
```
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
pip install numpy datasets Pillow tqdm transformers

python benchmark_serving.py --backend openai-chat --model "llama3.1:8b-instruct-q5_K_M" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --port 11434 --endpoint /v1/chat/completions


python benchmark_serving.py --backend openai-chat --model 'llama3.1-8b' --base-url 'http://localhost:8080' --host localhost --port 8080 --endpoint='/v1/chat/completions' --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Results (HEAD b4276):
```
============ Serving Benchmark Result ============
Successful requests:                     32        
Benchmark duration (s):                  249.14    
Total input tokens:                      6449      
Total generated tokens:                  16365     
Request throughput (req/s):              0.13      
Output token throughput (tok/s):         65.69     
Total Token throughput (tok/s):          91.57     
---------------Time to First Token----------------
Mean TTFT (ms):                          114.67    
Median TTFT (ms):                        85.94     
P99 TTFT (ms):                           362.63    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.81     
Median TPOT (ms):                        14.77     
P99 TPOT (ms):                           15.88     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.03     
Median ITL (ms):                         14.96     
P99 ITL (ms):                            16.47     
==================================================
```

### Qwen2.5
Qwen2.5 does not have Flash Attention support in vLLM:
```
WARNING 12-07 18:09:13 registry.py:308] Model architecture 'Qwen2ForCausalLM' is partially supported by ROCm: Sliding window attention (SWA) is not yet supported in Triton flash attention. For half-precision SWA support, please use CK flash attention by setting `VLLM_USE_TRITON_FLASH_ATTN=0`
```

 Qwen2.5-Coder-0.5B-Instruct-Q8_0 runs significantly slower vs llama.cpp.  vLLM:
```
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  41.62     
Total input tokens:                      3211      
Total generated tokens:                  3074      
Request throughput (req/s):              0.38      
Output token throughput (tok/s):         73.86     
Total Token throughput (tok/s):          151.00    
---------------Time to First Token----------------
Mean TTFT (ms):                          1796.72   
Median TTFT (ms):                        1899.05   
P99 TTFT (ms):                           2002.22   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.24      
Median TPOT (ms):                        4.26      
P99 TPOT (ms):                           4.49      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.19      
Median ITL (ms):                         4.20      
P99 ITL (ms):                            4.77      
==================================================
```
- look at that TTFT, yikes

llama.cpp:
```
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  38.04     
Total input tokens:                      3211      
Total generated tokens:                  7472      
Request throughput (req/s):              0.42      
Output token throughput (tok/s):         196.41    
Total Token throughput (tok/s):          280.82    
---------------Time to First Token----------------
Mean TTFT (ms):                          18.65     
Median TTFT (ms):                        16.21     
P99 TTFT (ms):                           45.31     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.97      
Median TPOT (ms):                        4.92      
P99 TPOT (ms):                           5.24      
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.06      
Median ITL (ms):                         5.07      
P99 ITL (ms):                            5.42      
==================================================
```

## ExLlamaV2

Tabby Server:
```
python start.py --port 8000 --disable-auth 1 --dummy-model-name llama3.1-8b --model-dir /models/exl2 --model-name turboderp_Llama-3.1-8B-Instruct-exl2-5.0bpw
```
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for better performance
Benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Results:
```
============ Serving Benchmark Result ============
Successful requests:                     32        
Benchmark duration (s):                  347.96    
Total input tokens:                      6449      
Total generated tokens:                  16216     
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         46.60     
Total Token throughput (tok/s):          65.14     
---------------Time to First Token----------------
Mean TTFT (ms):                          160.39    
Median TTFT (ms):                        148.70    
P99 TTFT (ms):                           303.35    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.31     
Median TPOT (ms):                        18.47     
P99 TPOT (ms):                           27.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.18     
Median ITL (ms):                         19.80     
P99 ITL (ms):                            38.79     
==================================================
```

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` throws error:
- `ERROR:    RuntimeError: [_efficient_attention_forward] Unsupported mask type on ROCM, for now`
- See:
	- https://github.com/pytorch/pytorch/issues/138300
	- https://github.com/pytorch/pytorch/pull/134498
	- https://github.com/ROCm/aotriton/pull/39
	- https://github.com/ROCm/flash-attention/issues/79

Use nightly PyTorch:
```
pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```
Grab AOTriton 0.8 Beta https://github.com/ROCm/aotriton/releases/tag/0.8b
```
cd ~/Downloads
wget https://github.com/ROCm/aotriton/releases/download/0.8b/aotriton-0.8b-manylinux_2_28_x86_64-rocm6.2-shared.tar.gz
tar xvfz aotriton-0.8b-manylinux_2_28_x86_64-rocm6.2-shared.tar.gz
```
Replace AOTriton so
```
cd ~/mambaforge/envs/exllamav2/lib/python3.12/site-packages/torch/lib
mv libaotriton_v2.so libaotriton_v2.so.torch
cp ~/Downloads/aotriton/aotriton/lib/libaotriton_v2.so* ./ 
```
Try running TabbyAPI:
```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python start.py --port 8000 --disable-auth 1 --dummy-model-name llama3.1-8b --model-dir /models/exl2 --model-name turboderp_Llama-3.1-8B-Instruct-exl2-5.0bpw

python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

See also: https://github.com/bjj/exllamav2-openai-server
# Future Optimization
https://github.com/mostlygeek/llama-swap/tree/main/examples/benchmark-snakegame