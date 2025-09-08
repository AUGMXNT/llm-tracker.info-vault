

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
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |  1 |           pp512 |     7432.36 ± 248.16 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |  1 |           tg128 |        251.04 ± 0.43 |

build: 233d773d0 (6413)
```