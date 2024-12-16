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
## Server
```
# Tuned
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048

# sd 1B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.2-1B-Instruct

# sd 8B
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve -tp 8 meta-llama/Llama-3.1-70B-Instruct --num-scheduler-steps 12 --gpu_memory_utilization=0.96 --max-num-seqs 2048 --speculative_model meta-llama/Llama-3.1-8B-Instruct
```

## Bench
```
(vllm-rocm) hotaisle@ENC1-CLS01-SVR14:~/vllm-rocm/benchmarks$ python benchmark_serving.py --backend openai-chat --model meta-llama/Llama-3.1-70B-Instruct --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 128 --tokenizer meta-llama/Llama-3.1-70B-Instruct --port 8000 --endpoint /v1/chat/completions
```
## Results
Tuned
```
```
sd 1b
```
```
sd 8b
