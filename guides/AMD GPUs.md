As of August 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is available for Linux or [Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html).

# Linux
Testing was done with a Radeon VII (16GB HBM2 VRAM, gfx906) on Arch Linux

[Officially Supported GPUs for ROCm 5.6](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html#supported-gpus) are: Radeon VII, Radeon Pro VII, V620, W6800, and MI Instinct MI50, MI100, MI210, MI250, MI250X


## RDNA3 (eg 7900 XT, XTX)
These **are not officially supported w/ ROCm 5.6** (but some support coming in 5.7 in Fall 2023), however you *might* be able to get it work for certain tasks (SD, LLM inferencing):
* [https://github.com/RadeonOpenCompute/ROCm/issues/1880](https://github.com/RadeonOpenCompute/ROCm/issues/1880)
* [https://gist.github.com/BloodBlight/0d36b33d215056395f34db26fb419a63](https://gist.github.com/BloodBlight/0d36b33d215056395f34db26fb419a63)
* [https://github.com/are-we-gfx1100-yet]([https://github.com/are-we-gfx1100-yet)
* [https://cprimozic.net/notes/posts/setting-up-tensorflow-with-rocm-on-7900-xtx/](https://cprimozic.net/notes/posts/setting-up-tensorflow-with-rocm-on-7900-xtx/)
* [https://news.ycombinator.com/item?id=36574179](https://news.ycombinator.com/item?id=36574179)

## AMD APU
* Performance 65W 7940HS w/ 64GB of DDR5-5600 (83GB/s theoretical memory bandwidth): [https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1041125589)
  * On small (7B) models that fit within the UMA VRAM, ROCm performance is very similar to my M2 MBA's Metal performance. Inference is barely faster than CLBlast/CPU though (~10% faster).
  * On a big (70B) model that doesn't fit into allocated VRAM, the ROCm inferences slower than CPU w/ -ngl 0 (CLBlast crashes), and CPU perf is about as expected - about 1.3 t/s inferencing a Q4_K_M. Besides being slower, the ROCm version also caused amdgpu exceptions that killed Wayland 2/3 times (I'm running Linux 6.5.4, ROCm 5.6.1, mesa 23.1.8).

Note BIOS allows me to set up to 8GB for VRAM in BIOS (UMA_SPECIFIED GART), ROCm does not support GTT (about 35GB/64GB if it did support it, which is not enough for a 70B Q4_0, not that you'd want to at those speeds).

Vulkan drivers can use GTT memory dynamically, but w/ MLC LLM, Vulkan version is 35% slower than CPU-only llama.cpp. Also, the max GART+GTT is still too small for 70B models.
* It may be possible to unlock more UMA/GART memory: [https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056](https://winstonhyypia.medium.com/amd-apu-how-to-modify-the-dedicated-gpu-memory-e27b75905056)
* There is custom allocator that may allow PyTorch to use GTT memory (only useful for PyTorch inferencing obviously): [https://github.com/pomoke/torch-apu-helper](https://github.com/pomoke/torch-apu-helper)
* A writeup of someone playing around w/ ROCm and SD on an older APU: [https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/](https://www.gabriel.urdhr.fr/2022/08/28/trying-to-run-stable-diffusion-on-amd-ryzen-5-5600g/)

## Arch Linux Setup
Install ROCm:
```
# all the amd gpu compute stuff
yay -S rocm-hip-sdk rocm-ml-sdk rocm-opencl-sdk

# third party monitoring
yay -S amdgpu_top radeontop
```
Install conda (mamba)
```
yay -S mambaforge
/opt/mambaforge/bin/mamba init fish
```
Create Environment
```
mamba create -n llm
mamba activate llm
```

## OC
We have some previous known good memory timings for our Radeon VII card:
```
sudo sh -c 'echo manual > /sys/class/drm/card0/device/power_dpm_force_performance_level'
sudo sh -c 'echo 8 > /sys/class/drm/card0/device/pp_dpm_sclk'
sudo amdmemorytweak --gpu 0 --ref 7500 --rtp 6 --rrds 3 --faw 12 --ras 19 --rc 30 --rcdrd 11 --rp 11
```

## llama.cpp
Let's first try llama.cpp
```
mkdir ~/llm
cd ~/llm
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
LLAMA_CLBLAST=1 make
```
* [https://www.reddit.com/r/LocalLLaMA/comments/13p8zq2/update_opencl_is_merged_amd_gpus_now_work_with/](https://www.reddit.com/r/LocalLLaMA/comments/13p8zq2/update_opencl_is_merged_amd_gpus_now_work_with/)
* [https://www.reddit.com/r/LocalLLaMA/comments/13m8li2/finally_got_a_model_running_on_my_xtx_using/](https://www.reddit.com/r/LocalLLaMA/comments/13m8li2/finally_got_a_model_running_on_my_xtx_using/)

We're benchmarking with with a recent llama-13b q4_0 fine tune ([Nous Hermes](https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML))

Here are the results from 2023-06-29 [commit 96a712c](https://github.com/ggerganov/llama.cpp/commit/96a712ca1b7f427e3bd7ffc0c70b2105cfc7fbf1)


NOTE: We use `-ngl 99` to ensure all layers are loaded in memory.

```
CUDA_VISIBLE_DEVICES=0 ./main -m ../models/nous-hermes-13b.ggmlv3.q4_0.bin -ngl 99 -n 2048 --ignore-eos

main: build = 762 (96a712c)
main: seed  = 1688035176
ggml_opencl: selecting platform: 'AMD Accelerated Parallel Processing'
ggml_opencl: selecting device: 'gfx906:sramecc+:xnack-'
ggml_opencl: device FP16 support: true
llama.cpp: loading model from ../models/nous-hermes-13b.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32001
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 5120
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 40
llama_model_load_internal: n_layer    = 40
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 13824
llama_model_load_internal: model size = 13B
llama_model_load_internal: ggml ctx size =    0.09 MB
llama_model_load_internal: using OpenCL for GPU acceleration
llama_model_load_internal: mem required  = 2223.88 MB (+ 1608.00 MB per state)
llama_model_load_internal: offloading 40 repeating layers to GPU
llama_model_load_internal: offloading non-repeating layers to GPU
llama_model_load_internal: offloading v cache to GPU
llama_model_load_internal: offloading k cache to GPU
llama_model_load_internal: offloaded 43/43 layers to GPU
llama_model_load_internal: total VRAM used: 8416 MB
llama_new_context_with_model: kv self size  =  400.00 MB

system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 | 
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 2048, n_keep = 0

...

llama_print_timings:        load time =  6946.39 ms
llama_print_timings:      sample time =  2172.76 ms /  2048 runs   (    1.06 ms per token,   942.58 tokens per second)
llama_print_timings: prompt eval time = 51096.76 ms /  1801 tokens (   28.37 ms per token,    35.25 tokens per second)
llama_print_timings:        eval time = 308604.23 ms /  2040 runs   (  151.28 ms per token,     6.61 tokens per second)
llama_print_timings:       total time = 362807.86 ms
```

We get 6.61 t/s.

`rocm-smi` looks something like:
```
GPU  Temp (DieEdge)  AvgPwr  SCLK     MCLK    Fan     Perf    PwrCap  VRAM%  GPU%
0    59.0c           112.0W  1801Mhz  800Mhz  44.71%  manual  250.0W   49%   40%
```
## llama.cpp HIP fork
Now let's see if [HIPified CUDA](https://github.com/ggerganov/llama.cpp/pull/1087) makes a difference using [this fork](https://github.com/SlyEcho/llama.cpp/tree/hipblas)

Here are the results from a 13b-q4_0 on 2023-06-29, [commit 04419f1](https://github.com/SlyEcho/llama.cpp/commit/04419f18947e7b0dc43c07869eac3965f22b34cf) 
```
git clone https://github.com/SlyEcho/llama.cpp llama.cpp-hip
cd llama.cpp-hip
git fetch
make -j8 LLAMA_HIPBLAS=1

CUDA_VISIBLE_DEVICES=0 ./main -m ../models/nous-hermes-13b.ggmlv3.q4_0.bin -ngl 99 -n 2048 --ignore-eos

main: build = 821 (04419f1)
main: seed  = 1688034262
ggml_init_cublas: found 1 CUDA devices:
  Device 0: AMD Radeon VII
llama.cpp: loading model from ../models/nous-hermes-13b.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32001
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 5120
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 40
llama_model_load_internal: n_layer    = 40
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 13824
llama_model_load_internal: model size = 13B
llama_model_load_internal: ggml ctx size =    0.09 MB
llama_model_load_internal: using CUDA for GPU acceleration
llama_model_load_internal: mem required  = 2135.99 MB (+ 1608.00 MB per state)
llama_model_load_internal: allocating batch_size x 1 MB = 512 MB VRAM for the scratch buffer
llama_model_load_internal: offloading 40 repeating layers to GPU
llama_model_load_internal: offloading non-repeating layers to GPU
llama_model_load_internal: offloading v cache to GPU
llama_model_load_internal: offloading k cache to GPU
llama_model_load_internal: offloaded 43/43 layers to GPU
llama_model_load_internal: total VRAM used: 9016 MB
llama_new_context_with_model: kv self size  =  400.00 MB

system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0
 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 |
 sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 2048, n_keep = 0

...

llama_print_timings:        load time =  4049.27 ms
llama_print_timings:      sample time =  1307.03 ms /  2048 runs   (    0.64 ms per token,  1566.91 tokens per second)
llama_print_timings: prompt eval time = 17486.67 ms /  1801 tokens (    9.71 ms per token,   102.99 tokens per second)
llama_print_timings:        eval time = 157571.58 ms /  2040 runs   (   77.24 ms per token,    12.95 tokens per second)
llama_print_timings:       total time = 176912.26 m
```
* See also: [https://github.com/ggerganov/llama.cpp/pull/1087](https://github.com/ggerganov/llama.cpp/pull/1087)

We get 12.95 t/s, almost 2X faster than the OpenCL version. If you are using llama.cpp on AMD GPUs, I think it's safe to say you should definitely use this HIP fork.

Note, the 4C Zen2 Ryzen 2400G CPU gets about 2.2 t/s, so performance is about 6X.

## exllama
[ROCm support was merged](https://github.com/turboderp/exllama/pull/7) 2023-06-07.

We run a 13B 4-bit GPTQ ([Manticore-13B-GPTQ](https://huggingface.co/TheBloke/Manticore-13B-GPTQ)) on 2023-06-29 w/ [commit 93d50d1](https://github.com/turboderp/exllama/commit/93d50d1cebf7105cba56f89fa057397e95d60572)

```
# make sure we have git-lfs working
yay -S git-lfs
git lfs install
# in models folder
git clone https://huggingface.co/TheBloke/Manticore-13B-GPTQ

git clone https://github.com/turboderp/exllama
cd exllama

# install ROCm PyTorch https://pytorch.org/get-started/locally/
pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.4.2
pip install -r requirements.txt

python test_benchmark_inference.py -d ~/llm/models/Manticore-13B-GPTQ/ -p
Successfully preprocessed all matching files.
 -- Tokenizer: /home/lhl/llm/models/Manticore-13B-GPTQ/tokenizer.model
 -- Model config: /home/lhl/llm/models/Manticore-13B-GPTQ/config.json
 -- Model: /home/lhl/llm/models/Manticore-13B-GPTQ/Manticore-13B-GPTQ-4bit-128g.no-act-order.safetensors
 -- Sequence length: 2048
 -- Tuning:
 -- --matmul_recons_thd: 8
 -- --fused_mlp_thd: 2
 -- --sdp_thd: 8
 -- --rmsnorm_no_half2
 -- --rope_no_half2
 -- --matmul_no_half2
 -- --silu_no_half2
 -- Options: ['perf']
 ** Time, Load model: 6.86 seconds
 ** Time, Load tokenizer: 0.01 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,873.52 MB - [cuda:1] 0.00 MB
 -- Warmup pass 1...
 ** Time, Warmup: 0.36 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 4.43 seconds
 -- Inference, first pass.
 ** Time, Inference: 4.52 seconds
 ** Speed: 425.13 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 9.92 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 19.21 tokens/second
 ** VRAM, Inference: [cuda:0] 2,253.20 MB - [cuda:1] 0.00 MB
 ** VRAM, Total: [cuda:0] 9,126.72 MB - [cuda:1] 0.00 MB
```

These results are actually a regression from [commit dd63e07](https://github.com/turboderp/exllama/commit/dd63e0734b7df5fcbd86d30ad82a582da25a3a73) (which was about 15 t/s). At 9.92 t/s, the llama.cpp HIP{ fork is now 30% faster.


## RTX 4090 Comparison
As a point of comparison, running `llama.cpp` with `make LLAMA_CUBLAS=1` runs at about 72 t/s:
```
./main -m /data/ai/models/llm/manticore/Manticore-13B-Chat-Pyg.ggmlv3.q4_0.bin -ngl 99 -n 2048 --ignore-eos

...

llama_print_timings:        load time =  3569.39 ms
llama_print_timings:      sample time =   930.53 ms /  2048 runs   (    0.45 ms per token,  2200.89 tokens per second)
llama_print_timings: prompt eval time =  2608.07 ms /  1801 tokens (    1.45 ms per token,   690.55 tokens per second)
llama_print_timings:        eval time = 28273.11 ms /  2040 runs   (   13.86 ms per token,    72.15 tokens per second)
llama_print_timings:       total time = 32225.03 ms
```

[exllama](https://github.com/turboderp/exllama), performs about on-par to llama.cpp and we get 74.79 t/s:
```
python test_benchmark_inference.py -p -d /models/llm/manticore/manticore-13b-chat-pyg-GPTQ
 -- Tokenizer: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/tokenizer.model
 -- Model config: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/config.json
 -- Model: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/Manticore-13B-Chat-Pyg-GPTQ-4bit-128g.no-act-order.safetensors
 -- Sequence length: 2048
 -- Tuning:
 -- --matmul_recons_thd: 8
 -- --fused_mlp_thd: 2
 -- --sdp_thd: 8
 -- Options: ['perf']
 ** Time, Load model: 3.98 seconds
 ** Time, Load tokenizer: 0.01 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,873.52 MB
 -- Warmup pass 1...
 ** Time, Warmup: 1.55 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 0.07 seconds
 -- Inference, first pass.
 ** Time, Inference: 0.25 seconds
 ** Speed: 7600.98 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 74.79 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 99.17 tokens/second
 ** VRAM, Inference: [cuda:0] 1,772.79 MB
 ** VRAM, Total: [cuda:0] 8,646.31 MB
```

## Recommendation
Radeon VII 16GB cards are going for about $250-$300 on eBay (equivalent to an Instinct MI50 which range a lot in price; MI60 or MI100 are similar also similar generation cards but with 32GB of RAM). 

For the performance, you're much better off paying about [$200](https://www.ebay.com/itm/195565620918) ([alt](https://www.ebay.com/itm/195745134833)) for an Nvidia Tesla P40 24GB (1080Ti class but with more RAM) or about $700 for an RTX 3090 24GB. The P40 can [reportedly run 13b models at about 15 tokens/s](https://www.reddit.com/r/LocalLLaMA/comments/13n8bqh/my_results_using_a_tesla_p40/), over 2X faster than a Radeon VII and with lots more software support. Also, 24GB cards support 30b models, which 16GB cards can't do.

## Bonus: GTX 1080 Ti Comparison
I dug out my old GTX 1080 Ti and installed it to get a ballpark vs P40 numbers. 

We are running `llama.cpp` with the same checkout (2023-06-29 commit 96a712c). The GPU refactor no longer has a CUDA kernel for the 1080 Ti, so I've used `LLAMA_CLBLAST=1` instead, but it still runs faster than the older (un-optimized) CUDA version (previous tests output at 5.8 t/s).

```
./main -m /data/ai/models/llm/manticore/Manticore-13B-Chat-Pyg.ggmlv3.q4_0.bin -ngl 99 -n 2048 --ignore-eos                                                (llama) 
main: build = 762 (96a712c)
main: seed  = 1688074299
ggml_opencl: selecting platform: 'NVIDIA CUDA'
ggml_opencl: selecting device: 'NVIDIA GeForce GTX 1080 Ti'
ggml_opencl: device FP16 support: false
llama.cpp: loading model from /data/ai/models/llm/manticore/Manticore-13B-Chat-Pyg.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 5120
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 40
llama_model_load_internal: n_layer    = 40
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 13824
llama_model_load_internal: model size = 13B
llama_model_load_internal: ggml ctx size =    0.09 MB
llama_model_load_internal: using OpenCL for GPU acceleration
llama_model_load_internal: mem required  = 2223.88 MB (+ 1608.00 MB per state)
llama_model_load_internal: offloading 40 repeating layers to GPU
llama_model_load_internal: offloading non-repeating layers to GPU
llama_model_load_internal: offloading v cache to GPU
llama_model_load_internal: offloading k cache to GPU
llama_model_load_internal: offloaded 43/43 layers to GPU
llama_model_load_internal: total VRAM used: 8416 MB
llama_new_context_with_model: kv self size  =  400.00 MB

system_info: n_threads = 16 / 32 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 | 
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 2048, n_keep = 0

...

llama_print_timings:        load time =  1459.18 ms
llama_print_timings:      sample time =   884.79 ms /  2048 runs   (    0.43 ms per token,  2314.67 tokens per second)
llama_print_timings: prompt eval time = 31935.74 ms /  1801 tokens (   17.73 ms per token,    56.39 tokens per second)
llama_print_timings:        eval time = 220695.57 ms /  2040 runs   (  108.18 ms per token,     9.24 tokens per second)
llama_print_timings:       total time = 253862.42 ms
```

Using CLBlast, we get 9.24 t/s, which is a little slower than the Radeon VII.

`exllama` is no longer very happy with Pascal cards, although reports are that gptq-for-llama/autogptq can output at 20 t/s: [https://github.com/turboderp/exllama/issues/75](https://github.com/turboderp/exllama/issues/75)

## ROCm Resources
ROCm support is outside the scope of this guide (maybe someone can make a new page if they have experience and can refactor).

* ROCm [officially supports RHEL, SLES, and Ubuntu](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
* For Arch Linux, there are packages in [extra] (eg [rocm-core](https://archlinux.org/packages/extra/x86_64/rocm-core/)) but packages may be behind - eg, currently on version 5.4.3 and 5.5.1 did not make it out of staging before 5.6.0 was released. These appear to be built from source. See also:
  * [rocm-arch/rocm-arch PKGBUILDS](https://github.com/rocm-arch/rocm-arch) - these appear different from the [extra] PKGBUILDS...
  * [CosmicFusion/rocm-polaris-arch](https://github.com/CosmicFusion/rocm-polaris-arch) - this looks useful if you're tyring to enable non-officially supported hardware

# Windows

## llama.cpp

For an easy time, go to [llama.cpp's release page](https://github.com/ggerganov/llama.cpp/releases) and download a `bin-win-clblast` version.

In the Windows terminal, run it with `-ngl 99` to load all the layers into memory.

```
.\main.exe -m model.bin -ngl 99
```

On a Radeon 7900XT, you should get about double the performance of CPU-only execution.

### Compile for ROCm
This was last update 2023-09-03 so things might change, but here's how I was able to get things working in Windows.

#### Requirements
* You'll need [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/) installed. Install it with the basic C++ environment.
* Follow AMD's directions and [install the ROCm software for Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/index.html).
* You'll need `git` if you want to pull the latest from the repo (you can either get the [official Windows installer](https://git-scm.com/download/win) or use a package manager like [Chocolatey](https://chocolatey.org/) to `choco install git`) - note, as an alternative, you could just download the Source code.zip from the [https://github.com/ggerganov/llama.cpp/releases/](https://github.com/ggerganov/llama.cpp/releases/)

#### Instructions

First, launch "x64 Native Tools Command Prompt" from the Windows Menu (you can hit the Windows key and just start typing x64 and it should pop up).


```
# You should probably change to a folder you want first for grabbing the source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Make a build folder
mkdir build
cd build

# Make sure the HIP stuff gets picked up
cmake.exe .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLAMA_HIPBLAS=on  -DCMAKE_C_COMPILER="clang.exe" -DCMAKE_CXX_COMPILER="clang++.exe" -DAMDGPU_TARGETS="gfx1100" -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"

# This should build binaries in a bin/ folder
cmake.exe --build .
```

That's it, now you have compiled executables in `build/bin`.


Start a new terminal to run llama.CPP
```
# You can do this in the GUI search for "environment variable" as well
setx /M PATH "C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"

# Or for session
set PATH="C:\Program Files\AMD\ROCm\5.5\bin;%PATH%"
```

If you set just the global you may need to start a new shell before running this in the `llama.cpp` checkout. You can double check it'S working by outputing the path `echo %PATH%` or just running `hipInfo` or another exe in the ROCm bin folder.

NOTE: If your PATH is wonky for some reason you may get missing .dll errors. You can either fix that, or if all else fails, copy the missing files from `"C:\Program Files\AMD\ROCm\5.5\bin` into the `build/bin` folder since life is too short.

#### Results
Here's my `llama-bench` results running a llama2-7b q4_0 and q4_K_M:
```
C:\Users\lhl\Desktop\llama.cpp\build\bin>llama-bench.exe -m ..\..\meta-llama-2-7b-q4_0.gguf -p 3968 -n 128 -ngl 99
ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, compute capability 11.0
| model                      	|   	size | 	params | backend	| ngl | test   	|          	t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA v2 7B mostly Q4_0    	|   3.56 GiB | 	6.74 B | ROCm   	|  99 | pp 3968	|	882.92 ± 1.10 |
| LLaMA v2 7B mostly Q4_0    	|   3.56 GiB | 	6.74 B | ROCm   	|  99 | tg 128 	| 	94.55 ± 0.07 |

build: 69fdbb9 (1148)


C:\Users\lhl\Desktop\llama.cpp\build\bin>llama-bench.exe -m ..\..\meta-llama-2-7b-q4_K_M.gguf -p 3968 -n 128 -ngl 99
ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, compute capability 11.0
| model                      	|   	size | 	params | backend	| ngl | test   	|          	t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA v2 7B mostly Q4_K - Medium |   3.80 GiB | 	6.74 B | ROCm   	|  99 | pp 3968	|	858.74 ± 1.32 |
| LLaMA v2 7B mostly Q4_K - Medium |   3.80 GiB | 	6.74 B | ROCm   	|  99 | tg 128 	| 	78.78 ± 0.04 |

build: 69fdbb9 (1148)
```

### Unsupported Architectures
On Windows, it may not be possible to apply an `HSA_OVERRIDE_GFX_VERSION` override. In that case, these instructions for compiling custom kernels may help: [https://www.reddit.com/r/LocalLLaMA/comments/16d1hi0/guide_build_llamacpp_on_windows_with_amd_gpus_and/](https://www.reddit.com/r/LocalLLaMA/comments/16d1hi0/guide_build_llamacpp_on_windows_with_amd_gpus_and/)