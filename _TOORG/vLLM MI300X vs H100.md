# MI300

```
# Server
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 4 meta-llama/Llama-3.1-70B-Instruct --gpu_memory_utiliz
ation=0.98 --num-scheduler-steps 20

# Client
~/vllm-rocm/benchmarks$ python benchmark_serving.py --backend openai-chat --model meta-llama/Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 256 --tokenizer meta-llama/Llama-3.1-70B-Instruct --port 8000 --endpoint /v1/chat/completions

Maximum request concurrency: 128
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:35<00:00, 10.68it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  95.88
Total input tokens:                      220902
Total generated tokens:                  191594
Request throughput (req/s):              10.68
Output token throughput (tok/s):         1998.36
Total Token throughput (tok/s):          4302.41
---------------Time to First Token----------------
Mean TTFT (ms):                          1156.72
Median TTFT (ms):                        804.28
P99 TTFT (ms):                           4097.30
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.05
Median TPOT (ms):                        51.27
P99 TPOT (ms):                           76.24
---------------Inter-token Latency----------------
Mean ITL (ms):                           50.95
Median ITL (ms):                         33.46
P99 ITL (ms):                            560.38
==================================================

Maximum request concurrency: 256
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:24<00:00, 12.13it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  84.43
Total input tokens:                      220902
Total generated tokens:                  191585
Request throughput (req/s):              12.13
Output token throughput (tok/s):         2269.09
Total Token throughput (tok/s):          4885.40
---------------Time to First Token----------------
Mean TTFT (ms):                          2621.59
Median TTFT (ms):                        1455.22
P99 TTFT (ms):                           7408.13
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          92.84
Median TPOT (ms):                        84.74
P99 TPOT (ms):                           241.76
---------------Inter-token Latency----------------
Mean ITL (ms):                           81.05
Median ITL (ms):                         46.74
P99 ITL (ms):                            1180.43
==================================================


Maximum request concurrency: 512
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:24<00:00, 12.16it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  84.19
Total input tokens:                      220902
Total generated tokens:                  191574
Request throughput (req/s):              12.16
Output token throughput (tok/s):         2275.44
Total Token throughput (tok/s):          4899.22
---------------Time to First Token----------------
Mean TTFT (ms):                          16126.02
Median TTFT (ms):                        19532.95
P99 TTFT (ms):                           26480.88
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          93.01
Median TPOT (ms):                        84.16
P99 TPOT (ms):                           250.88
---------------Inter-token Latency----------------
Mean ITL (ms):                           80.65
Median ITL (ms):                         42.97
P99 ITL (ms):                            992.18
==================================================

```

```
# Server
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 4 meta-llama/Llama-3.1-70B-Instruct --gpu_memory_utiliz
ation=0.98 --num-scheduler-steps 20 --max-num-seqs 4096

# Client
(base) 1 hotaisle@ENC1-CLS01-SVR14:~/vllm-rocm/benchmarks$ python benchmark_serving.py --backend openai-chat --model meta-llama/Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 256 --tokenizer meta-llama/Llama-3.1-70B-Instruct --port 8000 --endpoint /v1/chat/completions


Maximum request concurrency: 128
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:36<00:00, 10.62it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  96.39     
Total input tokens:                      220902    
Total generated tokens:                  192022    
Request throughput (req/s):              10.62     
Output token throughput (tok/s):         1992.04   
Total Token throughput (tok/s):          4283.69   
---------------Time to First Token----------------
Mean TTFT (ms):                          1060.39   
Median TTFT (ms):                        813.15    
P99 TTFT (ms):                           3973.97   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          55.43     
Median TPOT (ms):                        51.67     
P99 TPOT (ms):                           120.38    
---------------Inter-token Latency----------------
Mean ITL (ms):                           51.73     
Median ITL (ms):                         33.44     
P99 ITL (ms):                            557.99    
==================================================

Maximum request concurrency: 256
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:24<00:00, 12.13it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  84.40
Total input tokens:                      220902
Total generated tokens:                  191426
Request throughput (req/s):              12.13
Output token throughput (tok/s):         2268.00
Total Token throughput (tok/s):          4885.23
---------------Time to First Token----------------
Mean TTFT (ms):                          2472.82
Median TTFT (ms):                        1426.22
P99 TTFT (ms):                           7257.70
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          95.16
Median TPOT (ms):                        85.84
P99 TPOT (ms):                           329.28
---------------Inter-token Latency----------------
Mean ITL (ms):                           81.81
Median ITL (ms):                         42.56
P99 ITL (ms):                            979.78
==================================================

Maximum request concurrency: 512
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:15<00:00, 13.65it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  75.04     
Total input tokens:                      220902    
Total generated tokens:                  192098    
Request throughput (req/s):              13.65     
Output token throughput (tok/s):         2560.09   
Total Token throughput (tok/s):          5504.05   
---------------Time to First Token----------------
Mean TTFT (ms):                          6944.99   
Median TTFT (ms):                        3276.44   
P99 TTFT (ms):                           13433.17  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          175.03    
Median TPOT (ms):                        136.59    
P99 TPOT (ms):                           1090.07   
---------------Inter-token Latency----------------
Mean ITL (ms):                           117.71    
Median ITL (ms):                         41.52     
P99 ITL (ms):                            1728.53   
==================================================
```

fp8 -tp 4
6200 - that's it?
```
Maximum request concurrency: 512
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:59<00:00, 17.10it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  59.90
Total input tokens:                      220902
Total generated tokens:                  191650
Request throughput (req/s):              17.10
Output token throughput (tok/s):         3199.72
Total Token throughput (tok/s):          6887.81
---------------Time to First Token----------------
Mean TTFT (ms):                          4064.47
Median TTFT (ms):                        2685.92
P99 TTFT (ms):                           10167.17
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          178.13
Median TPOT (ms):                        121.51
P99 TPOT (ms):                           1183.10
---------------Inter-token Latency----------------
Mean ITL (ms):                           99.00
Median ITL (ms):                         38.87
P99 ITL (ms):                            1103.67
==================================================

```

fp8 -tp 2
```
Maximum request concurrency: 512
100%|████████████████████████████████████████████████████████████████████████| 1024/1024 [01:35<00:00, 10.75it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  95.24     
Total input tokens:                      220902    
Total generated tokens:                  191551    
Request throughput (req/s):              10.75     
Output token throughput (tok/s):         2011.33   
Total Token throughput (tok/s):          4330.86   
---------------Time to First Token----------------
Mean TTFT (ms):                          9565.49   
Median TTFT (ms):                        4524.62   
P99 TTFT (ms):                           17677.60  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          202.14    
Median TPOT (ms):                        159.19    
P99 TPOT (ms):                           980.43    
---------------Inter-token Latency----------------
Mean ITL (ms):                           145.94    
Median ITL (ms):                         55.92     
P99 ITL (ms):                            2121.05   
==================================================

2011.33*4 = 8011.33
```


fp8  -tp 1 - 32K 
```
LLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 1 meta-llama/Llama-3.1-70B-Instruct --gpu_memory_utilization=0.98 --num-scheduler-steps 20 --max-num-seqs 4096 --quantization="fp8" --max-model-len 32768

INFO 12-25 14:00:09 model_runner.py:1094] Loading model weights took 67.6979 GB
INFO 12-25 14:00:28 worker.py:265] Memory profiling results: duration=18.41 seconds, total_gpu_memory=191.98GiB, 
initial_memory_usage=71.40GiB, peak_torch_memory=86.79GiB, memory_usage_post_profile=73.77GiB, non_torch_memory=6
.07GiB, kv_cache_size=95.28GiB, gpu_memory_utilization=0.98.
INFO 12-25 14:00:28 gpu_executor.py:76] # GPU blocks: 19513, # CPU blocks: 819
INFO 12-25 14:00:28 gpu_executor.py:80] Maximum concurrency for 32768 tokens per request: 9.53x


Maximum request concurrency: 512
100%|████████████████████████████████████████████████████████████████████████| 1024/1024 [02:02<00:00,  8.35it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  122.65    
Total input tokens:                      220902    
Total generated tokens:                  191232    
Request throughput (req/s):              8.35      
Output token throughput (tok/s):         1559.19   
Total Token throughput (tok/s):          3360.29   
---------------Time to First Token----------------
Mean TTFT (ms):                          8822.25   
Median TTFT (ms):                        6137.77   
P99 TTFT (ms):                           21440.85  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          352.85    
Median TPOT (ms):                        221.08    
P99 TPOT (ms):                           2422.45   
---------------Inter-token Latency----------------
Mean ITL (ms):                           201.20    
Median ITL (ms):                         92.02     
P99 ITL (ms):                            2922.29   
==================================================


(base) hotaisle@ENC1-CLS01-SVR14:~/vllm-rocm/benchmarks$ python
Python 3.12.6 | packaged by conda-forge | (main, Sep 22 2024, 14:16:49) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 1559*8
12472
```