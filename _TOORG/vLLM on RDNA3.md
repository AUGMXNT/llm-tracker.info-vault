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

## Full
Run server:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --num-scheduler-step 1 --served_model_name llama3.1-8b
```

Run benchmark:
```
python benchmark_serving.py --backend openai-chat --base-url 'http://localhost:8000' --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "llama3.1-8b" --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct`
```

Results:
```

```

## Q5_K_M
Run Server
```
vllm serve /app/model/gguf/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --num-scheduler-step 1 --served_model_name llama3.1-8b
```

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

# llama.cpp Comparion
## llama-bench
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
## benchmark_serving.py

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
