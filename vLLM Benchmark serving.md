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
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  286.06    
Total input tokens:                      220902    
Total generated tokens:                  192103    
Request throughput (req/s):              3.58      
Output token throughput (tok/s):         671.54    
Total Token throughput (tok/s):          1443.76   
---------------Time to First Token----------------
Mean TTFT (ms):                          420.74    
Median TTFT (ms):                        383.36    
P99 TTFT (ms):                           1625.62   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.82     
Median TPOT (ms):                        43.05     
P99 TPOT (ms):                           58.85     
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.94     
Median ITL (ms):                         33.32     
P99 ITL (ms):                            256.37    
==================================================


286.06/4 = 71.515
14.32
2686.16
5775.04
```

## sd 1b
```
# sd 1B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.2-1B-Instruct --num_speculative_tokens 5

# num_speculative_tokens 5
INFO 12-16 16:46:04 metrics.py:482] Speculative metrics: Draft acceptance rate: 0.797, System efficiency: 0.640, Number of speculative tokens: 5, Number of accepted tokens: 197711, Number of draft tokens: 248165, Number of emitted tokens: 190541.

============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  116.32    
Total input tokens:                      220902    
Total generated tokens:                  192104    
Request throughput (req/s):              8.80      
Output token throughput (tok/s):         1651.57   
Total Token throughput (tok/s):          3550.73   
---------------Time to First Token----------------
Mean TTFT (ms):                          447.49    
Median TTFT (ms):                        229.31    
P99 TTFT (ms):                           2581.31   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          84.12     
Median TPOT (ms):                        77.14     
P99 TPOT (ms):                           227.50    
---------------Inter-token Latency----------------
Mean ITL (ms):                           263.58    
Median ITL (ms):                         188.09    
P99 ITL (ms):                            929.71    
==================================================


# num_speculative_tokens 10

INFO 12-16 17:05:51 spec_decode_worker.py:961] SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=4.51 scoring_time_ms=32.27 verification_time_ms=1.58
INFO 12-16 17:05:54 metrics.py:482] Speculative metrics: Draft acceptance rate: 0.786, System efficiency: 0.443, Number of speculative tokens: 10, Number of accepted tokens: 311498, Number of draft tokens: 396390, Number of emitted tokens: 193355.

============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  141.94    
Total input tokens:                      220902    
Total generated tokens:                  191876    
Request throughput (req/s):              7.21      
Output token throughput (tok/s):         1351.84   
Total Token throughput (tok/s):          2908.18   
---------------Time to First Token----------------
Mean TTFT (ms):                          507.09    
Median TTFT (ms):                        349.68    
P99 TTFT (ms):                           2440.07   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          107.29    
Median TPOT (ms):                        95.27     
P99 TPOT (ms):                           307.86    
---------------Inter-token Latency----------------
Mean ITL (ms):                           401.02    
Median ITL (ms):                         336.03    
P99 ITL (ms):                            1050.55   
==================================================
```
- note, scheduler_steps > 1 not supported w/ speculative decoding
## sd 8b

```
# sd 8B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.1-8B-Instruct --num_speculative_tokens 5

# num_speculative_tokens 5
INFO 12-16 17:15:43 spec_decode_worker.py:961] SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=7.98 scoring_time_ms=36.46 verification_time_ms=1.61
INFO 12-16 17:15:47 metrics.py:482] Speculative metrics: Draft acceptance rate: 0.871, System efficiency: 0.741, Number of speculative tokens: 5, Number of accepted tokens: 186291, Number of draft tokens: 213925, Number of emitted tokens: 190301.

============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  117.36    
Total input tokens:                      220902    
Total generated tokens:                  191582    
Request throughput (req/s):              8.73      
Output token throughput (tok/s):         1632.46   
Total Token throughput (tok/s):          3514.76   
---------------Time to First Token----------------
Mean TTFT (ms):                          591.40    
Median TTFT (ms):                        259.36    
P99 TTFT (ms):                           2951.28   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          81.17     
Median TPOT (ms):                        76.20     
P99 TPOT (ms):                           195.45    
---------------Inter-token Latency----------------
Mean ITL (ms):                           306.46    
Median ITL (ms):                         220.89    
P99 ITL (ms):                            774.41    
==================================================
```
