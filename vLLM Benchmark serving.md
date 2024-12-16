venv: vllm
```
python benchmark_serving.py --backend openai-chat --base-url 'http://ip-10-1-1-135:8000/' --endpoint='/v1/chat/completions' --model "meta-llama/Llama-3.3-70B-Instruct" --dataset-name sharegpt --num-prompts 32 --max-concurrency 1
```

Concurrency 1, 4, 8, 16, 32

python benchmark_serving.py --backend openai-chat --host ssh-ubitus-a01c8ddb1cf77c36.elb.ap-northeast-1.amazonaws.com --port 8000 --endpoint='/shisa-405b-fp16/v1/chat/completions' --model "hisa-405b-fp16 --dataset-name sharegpt --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 32 --max-concurrency 1 --tokenizer meta-llama/Llama-3.3-70B-Instruct

Compare with TGI
https://github.com/huggingface/hf-rocm-benchmark
https://huggingface.co/docs/text-generation-inference/en/installation_amd
https://huggingface.co/docs/text-generation-inference/en/conceptual/chunking



# MI300X
## Benchmark
```
(vllm-rocm) hotaisle@ENC1-CLS01-SVR14:~/vllm-rocm/benchmarks$ python benchmark_serving.py --backend openai-chat --model meta-llama/Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 128 --tokenizer meta-llama/Llama-3.1-70B-Instruct --port 8000 --endpoint /v1/chat/completions
```
## Results

### tp8
```
# tp8
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048

============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  63.81     
Total input tokens:                      220902    
Total generated tokens:                  191043    
Request throughput (req/s):              16.05     
Output token throughput (tok/s):         2993.85   
Total Token throughput (tok/s):          6455.63   
---------------Time to First Token----------------
Mean TTFT (ms):                          564.63    
Median TTFT (ms):                        363.05    
P99 TTFT (ms):                           2301.28   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.10     
Median TPOT (ms):                        36.02     
P99 TPOT (ms):                           73.00     
---------------Inter-token Latency----------------
Mean ITL (ms):                           35.30     
Median ITL (ms):                         21.43     
P99 ITL (ms):                            276.44    
==================================================
```
- OK
### tp2 + pp4
```
# tp2 pp4
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 2 -pp 4 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048

============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  393.36    
Total input tokens:                      220902    
Total generated tokens:                  191625    
Request throughput (req/s):              2.60      
Output token throughput (tok/s):         487.14    
Total Token throughput (tok/s):          1048.72   
---------------Time to First Token----------------
Mean TTFT (ms):                          2011.53   
Median TTFT (ms):                        2473.93   
P99 TTFT (ms):                           3957.14   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          271.65    
Median TPOT (ms):                        237.66    
P99 TPOT (ms):                           794.14    
---------------Inter-token Latency----------------
Mean ITL (ms):                           2620.64   
Median ITL (ms):                         2629.03   
P99 ITL (ms):                            4034.65   
==================================================
```
- awful
### tp2 x 4
```
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 2 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048

# --max-concurrency = 32 (then multiply throughput x 4)
```

sd 1b
```
# sd 1B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.2-1B-Instruct


```
sd 8b

```
# sd 8B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.1-8B-Instruct
```
