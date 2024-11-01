I have an [Intel Core Ultra 7 258V](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html) laptop (see [in-progress Linux review](https://github.com/lhl/linuxlaptops/wiki/2024-MSI-Prestige-13-AI--Evo-A2VM)) which has an "Intel® Arc™ Graphics 140V" GPU (Xe2 architecture). Chips and Cheese has the [most in-depth analysis of the iGPU](https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels) which includes architectural and real world comparisons w/ the prior-gen Xe-LPG, as well as RDNA 3.5 (in the AMD Ryzen AI 9 HX 370 w/ Radeon 890M).

The 258V has Vector Engines with 2048-bit XMX units that Intel specs at 64 INT8 TOPS. Each XMX can do INT8 4096 OPS/clock or FP16 2048 OPS/clock, so that would be a max theoretical 32 FP16 TOPS.

https://www.indiekings.com/2024/08/intel-arc-140v-first-taste-of-xe2.html

https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF
https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true



https://github.com/ggerganov/llama.cpp/discussions/4167

https://community.amd.com/t5/ai/accelerating-llama-cpp-performance-in-consumer-llm-applications/ba-p/720311

https://github.com/intel/intel-npu-acceleration-library

# Comparisons
## 7900 XTX
```
$ ./llama-bench -m /models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         pp512 |      2892.62 ± 23.56 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         tg128 |         81.10 ± 0.06 |

build: 1804adb0 (4006)
```
## W7900
```
❯ ./llama-bench -m /models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         pp512 |       2557.94 ± 6.10 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | ROCm       |  99 |         tg128 |         76.29 ± 0.12 |

build: 1804adb0 (4006)
```
## RTX 3090
```
 CUDA_VISIBLE_DEVICES=1 ./llama-bench -m /models/llm/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         pp512 |      5408.04 ± 27.49 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         tg128 |        150.65 ± 0.90 |

build: 1804adb0 (4006)
```
## RTX 4090
```
❯ CUDA_VISIBLE_DEVICES=0 ./llama-bench -m /models/llm/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         pp512 |     12329.83 ± 43.77 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |  1 |         tg128 |        172.31 ± 0.04 |

build: 1804adb0 (4006)
```
