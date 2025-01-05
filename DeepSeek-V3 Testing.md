

# vLLM TP vs PP

## PP=2 TP=8
``
```
$ PP=2 sbatch vllm-server.slurm
...
[2025-01-05 11:32:46] Starting vLLM server with tensor parallel size: 8
[2025-01-05 11:32:46] Full vLLM command:
[2025-01-05 11:32:46] vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --num-scheduler-steps 20 --host 0.0.0.0 --port 8000
[2025-01-05 11:32:46] vLLM logs will be written to: /fsx/ubuntu/meti/experiments/00-enja-translation-eval/logs/278-20250105_113126/vllm_server.log
[2025-01-05 11:32:46] Full vLLM command:
[2025-01-05 11:32:46] vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 --pipeline-parallel-size 2 --gpu-memory-utilization 0.95 --num-scheduler-steps 20 --max-model-len 8192 --host 0.0.0.0 --port 8000 --trust-remote-code
[2025-01-05 11:32:46] vLLM server started with PID: 3427707
[2025-01-05 11:32:46] Waiting for vLLM server to become ready...
INFO 01-05 11:33:24 api_server.py:712] vLLM API server version 0.6.6.post2.dev5+g5ce4627a
INFO 01-05 11:33:24 api_server.py:713] args: Namespace(subparser='serve', model_tag='deepseek-ai/DeepSeek-V3', config='', host='0.0.0.0', port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='deepseek-ai/DeepSeek-V3', task='auto', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=8192, guided_decoding_backend='xgrammar', logits_processor_pattern=None, distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=2, tensor_parallel_size=8, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.95, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=None, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=20, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', generation_config=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, dispatch_function=<function serve at 0x7f3cdb0c6520>)

...

INFO 01-05 12:04:53 distributed_gpu_executor.py:57] # GPU blocks: 2889, # CPU blocks: 528
INFO 01-05 12:04:53 distributed_gpu_executor.py:61] Maximum concurrency for 8192 tokens per request: 5.64x
```
- 34 minutes to load

```
❯ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 64 --seed 42
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=64, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 64
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [14:09<00:00,  1.21it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  849.46
Total input tokens:                      229783
Total generated tokens:                  196323
Request throughput (req/s):              1.21
Output token throughput (tok/s):         231.11
Total Token throughput (tok/s):          501.62
---------------Time to First Token----------------
Mean TTFT (ms):                          2639.01
Median TTFT (ms):                        4076.09
P99 TTFT (ms):                           5234.55
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          319.29
Median TPOT (ms):                        236.82
P99 TPOT (ms):                           1464.94
---------------Inter-token Latency----------------
Mean ITL (ms):                           4421.57
Median ITL (ms):                         4402.53
P99 ITL (ms):                            5306.83
==================================================

❯ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 128 --seed 42
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=128, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 128
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [09:32<00:00,  1.79it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  572.44
Total input tokens:                      229783
Total generated tokens:                  195623
Request throughput (req/s):              1.79
Output token throughput (tok/s):         341.74
Total Token throughput (tok/s):          743.15
---------------Time to First Token----------------
Mean TTFT (ms):                          4602.03
Median TTFT (ms):                        4990.24
P99 TTFT (ms):                           15123.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          435.80
Median TPOT (ms):                        279.78
P99 TPOT (ms):                           2854.01
---------------Inter-token Latency----------------
Mean ITL (ms):                           5147.83
Median ITL (ms):                         4947.29
P99 ITL (ms):                            13139.99
==================================================

❯ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency 256 --seed 42
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=256, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 256
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [08:51<00:00,  1.93it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  531.65
Total input tokens:                      229783
Total generated tokens:                  196498
Request throughput (req/s):              1.93
Output token throughput (tok/s):         369.60
Total Token throughput (tok/s):          801.81
---------------Time to First Token----------------
Mean TTFT (ms):                          36067.72
Median TTFT (ms):                        36935.92
P99 TTFT (ms):                           78867.00
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          476.54
Median TPOT (ms):                        291.63
P99 TPOT (ms):                           3272.67
---------------Inter-token Latency----------------
Mean ITL (ms):                           4960.69
Median ITL (ms):                         4915.31
P99 ITL (ms):                            6029.69
==================================================


❯ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --
model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-
concurrency 512 --seed 42

Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_nam
e='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=512, model='deepseek-ai/DeepSeek-V3', tokenizer=No
ne, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, d
isable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metr
ics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_out
put_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_o
utput_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 512
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [09:05<00:00,  1.88it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  545.97
Total input tokens:                      229783
Total generated tokens:                  196351
Request throughput (req/s):              1.88
Output token throughput (tok/s):         359.64
Total Token throughput (tok/s):          780.51
---------------Time to First Token----------------
Mean TTFT (ms):                          101638.17
Median TTFT (ms):                        136694.69
P99 TTFT (ms):                           182805.42
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          488.89
Median TPOT (ms):                        299.54
P99 TPOT (ms):                           3528.80
---------------Inter-token Latency----------------
Mean ITL (ms):                           5150.19
Median ITL (ms):                         5100.70
P99 ITL (ms):                            6421.38
==================================================
```

## TP16
```
❯ for c in 64 128 256 512; do echo "Running with concurrency $c..." && python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1024 --max-concurrency $c --seed 42; done
Running with concurrency 64...
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=64, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 64
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [11:39<00:00,  1.46it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  699.36
Total input tokens:                      229783
Total generated tokens:                  196723
Request throughput (req/s):              1.46
Output token throughput (tok/s):         281.29
Total Token throughput (tok/s):          609.85
---------------Time to First Token----------------
Mean TTFT (ms):                          2223.85
Median TTFT (ms):                        3500.61
P99 TTFT (ms):                           3836.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          250.66
Median TPOT (ms):                        188.68
P99 TPOT (ms):                           1163.99
---------------Inter-token Latency----------------
Mean ITL (ms):                           3539.84
Median ITL (ms):                         3373.03
P99 ITL (ms):                            4265.99
==================================================
Running with concurrency 128...
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=128, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 128
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [06:46<00:00,  2.52it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  406.81
Total input tokens:                      229783
Total generated tokens:                  196577
Request throughput (req/s):              2.52
Output token throughput (tok/s):         483.22
Total Token throughput (tok/s):          1048.06
---------------Time to First Token----------------
Mean TTFT (ms):                          2340.63
Median TTFT (ms):                        3499.54
P99 TTFT (ms):                           3817.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          257.49
Median TPOT (ms):                        192.60
P99 TPOT (ms):                           1206.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           3592.16
Median ITL (ms):                         3349.91
P99 ITL (ms):                            4366.81
==================================================
Running with concurrency 256...
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=256, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 256
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [05:32<00:00,  3.08it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  332.79
Total input tokens:                      229783
Total generated tokens:                  196895
Request throughput (req/s):              3.08
Output token throughput (tok/s):         591.65
Total Token throughput (tok/s):          1282.12
---------------Time to First Token----------------
Mean TTFT (ms):                          13584.20
Median TTFT (ms):                        16816.48
P99 TTFT (ms):                           38278.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          359.57
Median TPOT (ms):                        223.84
P99 TPOT (ms):                           2436.09
---------------Inter-token Latency----------------
Mean ITL (ms):                           3916.60
Median ITL (ms):                         3904.36
P99 ITL (ms):                            6149.72
==================================================
Running with concurrency 512...
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=512, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1024, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 512
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [05:24<00:00,  3.15it/s]
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  324.76
Total input tokens:                      229783
Total generated tokens:                  196326
Request throughput (req/s):              3.15
Output token throughput (tok/s):         604.53
Total Token throughput (tok/s):          1312.09
---------------Time to First Token----------------
Mean TTFT (ms):                          51245.90
Median TTFT (ms):                        76532.73
P99 TTFT (ms):                           93584.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          355.42
Median TPOT (ms):                        219.52
P99 TPOT (ms):                           2423.14
---------------Inter-token Latency----------------
Mean ITL (ms):                           3868.20
Median ITL (ms):                         3855.88
P99 ITL (ms):                            6106.05
==================================================


```

# llama.cpp

## llama-bench
```
(base) ubuntu@ip-10-1-1-135:~/llama.cpp/DeepSeek-V3-Q5_K_M$ time ~/llama.cpp/llama.cpp/build/bin/llama-gguf-split --merge DeepSeek-V3-Q5_K_M-00001-of-00011.gguf DeepSeek-V3-Q5_K_M.gguf

(base) ubuntu@ip-10-1-1-135:~/llama.cpp/DeepSeek-V3-Q5_K_M$ time ../llama.cpp/build/bin/llama-bench -m DeepSeek-V3-Q5_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 8 CUDA devices:
  Device 0: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 1: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 2: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 3: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 4: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 5: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 6: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
  Device 7: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| deepseek2 671B Q5_K - Medium   | 442.74 GiB |   671.03 B | CUDA       |  99 |         pp512 |        290.28 ± 1.25 |
| deepseek2 671B Q5_K - Medium   | 442.74 GiB |   671.03 B | CUDA       |  99 |         tg128 |         23.63 ± 0.04 |

build: b56f079e (4418)

real    9m18.083s
user    1m18.287s
sys     7m58.478s
```


## benchmark_serving.py
```
# llama.cpp
(vllm) ubuntu@ip-10-1-1-135:~/meti/benchmarks$ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name share
gpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 50 --max-concurrency 1 --seed 42
Namespace(backend='openai-chat', base_url=None, host='localhost', port=8080, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=1, mode
l='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=50, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False,
metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_l
en=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [26:52<00:00, 32.24s/it]
============ Serving Benchmark Result ============
Successful requests:                     50
Benchmark duration (s):                  1612.14
Total input tokens:                      12211
Total generated tokens:                  35857
Request throughput (req/s):              0.03
Output token throughput (tok/s):         22.24
Total Token throughput (tok/s):          29.82
---------------Time to First Token----------------
Mean TTFT (ms):                          1353.39
Median TTFT (ms):                        1121.37
P99 TTFT (ms):                           3898.91
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.01
Median TPOT (ms):                        42.97
P99 TPOT (ms):                           44.10
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.08
Median ITL (ms):                         42.90
P99 ITL (ms):                            46.61
==================================================

# vLLM PP=2 TP=8
❯ python ~/vllm/benchmarks/benchmark_serving.py --backend openai-chat --host ip-10-1-21-143 --port 8000 --endpoint='/v1/chat/completions' --model "deepseek-ai/DeepSeek-V3" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 50 --max-concurrency 1 --seed 42
Namespace(backend='openai-chat', base_url=None, host='ip-10-1-21-143', port=8000, endpoint='/v1/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=1, model='deepseek-ai/DeepSeek-V3', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=50, logprobs=None, request_rate=inf, burstiness=1.0, seed=42, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto')
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [58:56<00:00, 70.73s/it]
============ Serving Benchmark Result ============
Successful requests:                     50
Benchmark duration (s):                  3536.56
Total input tokens:                      12211
Total generated tokens:                  10683
Request throughput (req/s):              0.01
Output token throughput (tok/s):         3.02
Total Token throughput (tok/s):          6.47
---------------Time to First Token----------------
Mean TTFT (ms):                          347.96
Median TTFT (ms):                        341.99
P99 TTFT (ms):                           427.86
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          408.90
Median TPOT (ms):                        339.68
P99 TPOT (ms):                           1127.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           6317.84
Median ITL (ms):                         6349.42
P99 ITL (ms):                            6846.15
==================================================

# vLLM TP=16
============ Serving Benchmark Result ============
Successful requests:                     50
Benchmark duration (s):                  1826.67
Total input tokens:                      12211
Total generated tokens:                  10742
Request throughput (req/s):              0.03
Output token throughput (tok/s):         5.88
Total Token throughput (tok/s):          12.57
---------------Time to First Token----------------
Mean TTFT (ms):                          394.63
Median TTFT (ms):                        176.61
P99 TTFT (ms):                           3931.75
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          207.92
Median TPOT (ms):                        172.19
P99 TPOT (ms):                           597.99
---------------Inter-token Latency----------------
Mean ITL (ms):                           3226.57
Median ITL (ms):                         3219.82
P99 ITL (ms):                            3330.43
==================================================
```