

# llama.cpp

## llama2-7b

### 600W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |     16634.04 ± 97.98 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        278.95 ± 0.23 |

build: 233d773d0 (6413)
```

### 500W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |    16496.11 ± 145.65 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        279.00 ± 0.06 |

build: 233d773d0 (6413)
```

### 450W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |    16060.49 ± 530.00 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        279.04 ± 0.03 |

build: 233d773d0 (6413)
```

### 400W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |   15225.27 ± 1276.77 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        279.04 ± 0.06 |

build: 233d773d0 (6413)
```

### 350W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |   14118.89 ± 2081.13 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        274.81 ± 0.63 |

build: 233d773d0 (6413)
```

### 300W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |   12802.05 ± 2888.77 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        260.42 ± 0.60 |

build: 233d773d0 (6413)
```

### 250W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |   10619.58 ± 3539.91 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        236.88 ± 0.21 |

build: 233d773d0 (6413)
```

### 200W
```
❯ build/bin/llama-bench -m /models/llm/gguf/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           pp512 |    9115.39 ± 4243.14 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CUDA       |  99 |  1 |           tg128 |        189.82 ± 2.81 |

build: 233d773d0 (6413)
```

## 405B
Single card:
```
❯ build/bin/llama-bench -fa 1 -m /models/llm/gguf/shisa-v2-llama3.1-405b-IQ2_XXS-00001-of-00003.gguf -ngl 116
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama ?B IQ2_XXS - 2.0625 bpw  |  99.90 GiB |   405.85 B | CUDA       | 116 |  1 |           pp512 |        222.08 ± 1.50 |
| llama ?B IQ2_XXS - 2.0625 bpw  |  99.90 GiB |   405.85 B | CUDA       | 116 |  1 |           tg128 |          2.68 ± 0.00 |
```


## 70B
```
❯ build/bin/llama-bench -fa 1 -m /models/llm/gguf/shisa-v2-llama3.3-70b.i1-Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | CUDA       |  99 |  1 |           pp512 |      1793.76 ± 13.25 |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | CUDA       |  99 |  1 |           tg128 |         34.05 ± 0.01 |

build: 233d773d0 (6413)
```

## Mistral Nemo 12B
```
❯ build/bin/llama-bench -fa 1 -m /models/llm/gguf/shisa-v2-mistral-nemo-12b.i1-Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | CUDA       |  99 |  1 |           pp512 |      9876.68 ± 41.64 |
| llama 13B Q4_K - Medium        |   6.96 GiB |    12.25 B | CUDA       |  99 |  1 |           tg128 |        163.31 ± 0.05 |

build: 233d773d0 (6413)
```

## gpt-oss-120b
```❯ build/bin/llama-bench -fa 1 -m /models/llm/gguf/gpt-oss-120b-UD-Q8_K_XL-00001-of-00002.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| gpt-oss 120B Q8_0              |  60.03 GiB |   116.83 B | CUDA       |  99 |  1 |           pp512 |     2419.45 ± 513.86 |
| gpt-oss 120B Q8_0              |  60.03 GiB |   116.83 B | CUDA       |  99 |  1 |           tg128 |        193.30 ± 0.99 |

build: 233d773d0 (6413)
```


## Qwen3 30B-A3B
```
❯ build/bin/llama-bench -fa 1 -m /models/llm/gguf/Qwen3-30B-A3B-Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |  1 |           pp512 |      7640.08 ± 25.58 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |  1 |           tg128 |        251.61 ± 0.39 |

build: 233d773d0 (6413)
```


# Attention Gym
```
❯ python examples/benchmark.py
Using the default sparsity block size: 128
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                       Causal Mask                                       ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Correctness check passed ✅
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         6.6376 |            331.3  |        20.3867 |            269.66 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        34.8769 |             64.04 |        99.9404 |             55.87 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        10.0974 |            221.18 |        30.0158 |            186.02 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
BlockMask(shape=(1, 1, 8192, 8192), sparsity=49.22%,
(0, 0)
░░
██░░
████░░
██████░░
████████░░
██████████░░
████████████░░
██████████████░░
████████████████░░
██████████████████░░
████████████████████░░
██████████████████████░░
████████████████████████░░
██████████████████████████░░
████████████████████████████░░
██████████████████████████████░░
)
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                        Alibi Mod                                        ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         6.6637 |            330    |        20.5225 |            267.88 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        35.883  |            122.57 |       101.713  |            108.1  |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        24.2935 |            181.04 |        60.4314 |            181.94 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
None
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                   Sliding Window 1024                                   ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Correctness check passed ✅
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         6.899  |            318.75 |        20.765  |            264.75 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        34.3567 |             16.88 |        99.5174 |             14.57 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |         3.3248 |            174.39 |         9.8839 |            146.66 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
BlockMask(shape=(1, 1, 8192, 8192), sparsity=86.82%,
(0, 0)
░░
██░░
░░██░░
  ░░██░░
    ░░██░░
      ░░██░░
        ░░██░░
          ░░██░░
            ░░██░░
              ░░██░░
                ░░██░░
                  ░░██░░
                    ░░██░░
                      ░░██░░
                        ░░██░░
                          ░░██░░
)
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                Prefix Lm Causal Mask 1024                               ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Correctness check passed ✅
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         6.7311 |            326.69 |        20.6141 |            266.69 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        35.5379 |             63.69 |       100.835  |             56.12 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |         9.8134 |            230.65 |        30.8518 |            183.41 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
BlockMask(shape=(1, 1, 8192, 8192), sparsity=48.54%,
(0, 0)
████
████
████░░
██████░░
████████░░
██████████░░
████████████░░
██████████████░░
████████████████░░
██████████████████░░
████████████████████░░
██████████████████████░░
████████████████████████░░
██████████████████████████░░
████████████████████████████░░
██████████████████████████████░░
)
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                       Doc Mask Mod                                      ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
Correctness check passed ✅
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |       108.282  |            324.93 |       283.532  |            310.23 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |       585.221  |              5.68 |      1636.11   |              5.08 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        30.5244 |            108.84 |        63.6006 |            130.59 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
BlockMask(shape=(1, 1, 32768, 32768), sparsity=95.28%,
(0, 0)
░░
░░░░
  ░░░░
  ░░░░░░
      ░░░░
        ░░░░
        ░░░░░░
            ░░░░
            ░░░░░░
                ░░░░
                  ░░░░
                  ░░░░░░
                      ░░░░
                      ░░░░░░
                          ░░░░
                            ░░░░
                            ░░░░░░
                                ░░░░
                                ░░░░░░
                                    ██░░
)
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                     Tanh Softcap 30                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         6.8936 |            319    |        21.1924 |            259.41 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        37.0007 |            118.86 |       104.819  |            104.9  |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        34.7981 |            126.39 |        75.8538 |            144.95 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
None
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                  Tanh Softcap Approx 30                                 ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |         7.1194 |            308.88 |        21.1144 |            260.37 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        36.8829 |            119.24 |       104.563  |            105.15 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        22.5573 |            194.97 |        61.7585 |            178.03 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
None
```


# vLLM

### Shisa V2 UnPhi 14B
## c=max
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
INFO 09-09 03:24:41 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7ff17daa8040>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=None, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_r
ate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None
, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_
mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                              | 00:50 elapsed, 12105:01:31 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:43<00:00, 22.81it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  43.85
Total input tokens:                      214316
Total generated tokens:                  91942
Request throughput (req/s):              22.81
Output token throughput (tok/s):         2096.79
Total Token throughput (tok/s):          6984.37
---------------Time to First Token----------------
Mean TTFT (ms):                          9718.01
Median TTFT (ms):                        9388.28
P99 TTFT (ms):                           19349.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          305.36
Median TPOT (ms):                        170.61
P99 TPOT (ms):                           729.54
---------------Inter-token Latency----------------
Mean ITL (ms):                           98.99
Median ITL (ms):                         40.35
P99 ITL (ms):                            733.29
==================================================
```

## c=32
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency=32
INFO 09-09 03:27:26 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f20e9a63ba0>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=32, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_rat
e=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None,
ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mo
de='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                | 00:01 elapsed, 450:28:28 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:24<00:00, 11.86it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Maximum request concurrency:             32
Benchmark duration (s):                  84.29
Total input tokens:                      214316
Total generated tokens:                  91928
Request throughput (req/s):              11.86
Output token throughput (tok/s):         1090.59
Total Token throughput (tok/s):          3633.14
---------------Time to First Token----------------
Mean TTFT (ms):                          62.26
Median TTFT (ms):                        49.91
P99 TTFT (ms):                           135.56
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.26
Median TPOT (ms):                        25.61
P99 TPOT (ms):                           38.10
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.17
Median ITL (ms):                         23.01
P99 ITL (ms):                            84.66
==================================================
```

## c=64
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency=64
INFO 09-09 03:29:24 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f761cd87ba0>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=64, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_rat
e=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None,
ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mo
de='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                | 00:01 elapsed, 134:22:41 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 64
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:51<00:00, 19.52it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Maximum request concurrency:             64
Benchmark duration (s):                  51.22
Total input tokens:                      214316
Total generated tokens:                  92278
Request throughput (req/s):              19.52
Output token throughput (tok/s):         1801.67
Total Token throughput (tok/s):          5986.06
---------------Time to First Token----------------
Mean TTFT (ms):                          61.08
Median TTFT (ms):                        57.94
P99 TTFT (ms):                           126.50
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.14
Median TPOT (ms):                        28.16
P99 TPOT (ms):                           30.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.87
Median ITL (ms):                         27.05
P99 ITL (ms):                            32.06
==================================================
```

## c=128
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency=128
INFO 09-09 03:31:06 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7fd36325bba0>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=128, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_ra
te=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None,
 ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_m
ode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                | 00:01 elapsed, 348:02:54 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 128
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:35<00:00, 28.09it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Maximum request concurrency:             128
Benchmark duration (s):                  35.60
Total input tokens:                      214316
Total generated tokens:                  90886
Request throughput (req/s):              28.09
Output token throughput (tok/s):         2553.30
Total Token throughput (tok/s):          8574.17
---------------Time to First Token----------------
Mean TTFT (ms):                          84.11
Median TTFT (ms):                        68.22
P99 TTFT (ms):                           217.02
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          33.08
Median TPOT (ms):                        32.83
P99 TPOT (ms):                           39.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.09
Median ITL (ms):                         32.56
P99 ITL (ms):                            38.51
==================================================
```

## c=256
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency=256
INFO 09-09 03:32:21 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f38e0ee7ba0>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=256, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_ra
te=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None,
 ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_m
ode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                | 00:01 elapsed, 291:13:45 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 256
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:30<00:00, 32.89it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Maximum request concurrency:             256
Benchmark duration (s):                  30.40
Total input tokens:                      214316
Total generated tokens:                  91334
Request throughput (req/s):              32.89
Output token throughput (tok/s):         3004.18
Total Token throughput (tok/s):          10053.51
---------------Time to First Token----------------
Mean TTFT (ms):                          174.58
Median TTFT (ms):                        121.58
P99 TTFT (ms):                           380.03
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          53.93
Median TPOT (ms):                        56.78
P99 TPOT (ms):                           77.49
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.53
Median ITL (ms):                         41.37
P99 ITL (ms):                            67.17
==================================================
```

## c=512
```
❯ vllm bench serve --dataset-name sharegpt --model shisa-ai/shisa-v2-unphi4-14b --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency=512
INFO 09-09 03:33:38 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7fc7f29dbba0>, seed=0, num_prompts=1000, dataset_name='sharegpt', no_stream=False, da
taset_path='ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_le
n=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, ra
ndom_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_outpu
t_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm
', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=512, model='shisa-ai/shisa-v2-unphi4-14b', tokenizer=None, use_beam_search=False, logprobs=None, request_ra
te=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None,
 ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_m
ode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                | 00:01 elapsed, 330:21:19 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 512
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.30it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Maximum request concurrency:             512
Benchmark duration (s):                  26.81
Total input tokens:                      214316
Total generated tokens:                  92521
Request throughput (req/s):              37.30
Output token throughput (tok/s):         3451.19
Total Token throughput (tok/s):          11445.53
---------------Time to First Token----------------
Mean TTFT (ms):                          430.01
Median TTFT (ms):                        380.60
P99 TTFT (ms):                           733.46
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          75.76
Median TPOT (ms):                        69.05
P99 TPOT (ms):                           153.60
---------------Inter-token Latency----------------
Mean ITL (ms):                           47.72
Median ITL (ms):                         40.44
P99 ITL (ms):                            139.67
==================================================
```


# Training

llama-3.2-1b
85-88GB memory
bs=32
3:00-3:20h sft