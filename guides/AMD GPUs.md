As of June 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is only supported by Linux.

# Linux
Testing was done with a Radeon VII (16GB HBM2 rAM, gfx906) on Arch Linux

## Arch Linux Setup
Install rocm:
```
yay -S radeontop rocm-hip-sdk rocm-ml-sdk rocm-opencl-sdk
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

We're running a random recent llama-13b fine tune: [https://huggingface.co/TheBloke/manticore-13b-chat-pyg-GGML](https://huggingface.co/TheBloke/manticore-13b-chat-pyg-GGML)

NOTE: updated 2023-06-13, we need at least `-ngl 41` (output layer in VRAM) but `-ngl 99` will ensure everything goes in memory. Re-running with latest code.

This works and we get about 5.9 tokens/s at full context:
```
# OC
sudo sh -c 'echo manual > /sys/class/drm/card0/device/power_dpm_force_performance_level'
sudo sh -c 'echo 8 > /sys/class/drm/card0/device/pp_dpm_sclk'
sudo amdmemorytweak --gpu 0 --ref 7500 --rtp 6 --rrds 3 --faw 12 --ras 19 --rc 30 --rcdrd 11 --rp 11

./main -m ../models/Manticore-13B-Chat-Pyg.ggmlv3.q5_1.bin -ngl 99 -n 2048 --ignore-eos

llama_print_timings:        load time =  5051.83 ms
llama_print_timings:      sample time =  2263.84 ms /  2048 runs   (    1.11 ms per token)
llama_print_timings: prompt eval time = 47732.06 ms /  1801 tokens (   26.50 ms per token)
llama_print_timings:        eval time = 346007.66 ms /  2040 runs   (  169.61 ms per token)
llama_print_timings:       total time = 399521.84 ms

```

`radeontop` looks something like:
```
GPU  Temp (DieEdge)  AvgPwr  SCLK     MCLK     Fan    Perf    PwrCap  VRAM%  GPU%
0    58.0c           153.0W  1801Mhz  1000Mhz  51.37%  manual  250.0W   65%   58%
```
## llama.cpp HIP fork
Now let's see if [HIPified CUDA](https://github.com/ggerganov/llama.cpp/pull/1087) makes a difference using [this fork](https://github.com/SlyEcho/llama.cpp/tree/hipblas)
```
git clone https://github.com/SlyEcho/llama.cpp llama.cpp-hip
cd llama.cpp-hip
git fetch
make -j8 LLAMA_HIPBLAS=1

llama_print_timings:        load time = 15094.90 ms
llama_print_timings:      sample time =   120.42 ms /   200 runs   (    0.60 ms per token)
llama_print_timings: prompt eval time =  8700.99 ms /     2 tokens ( 4350.49 ms per token)
llama_print_timings:        eval time = 25970.27 ms /   199 runs   (  130.50 ms per token)
llama_print_timings:       total time = 41207.19 ms

```
* See also: [https://github.com/ggerganov/llama.cpp/pull/1087](https://github.com/ggerganov/llama.cpp/pull/1087)

2023-06-13 Update: This is about 23% faster than the OpenCL version.

At the end of it though we're at 7.7 tokens/s, and about a 3.5X over running on CPU (2.2 token/s on an AMD 5 Ryzen 2400G).

## exllama
[ROCm support was merged](https://github.com/turboderp/exllama/pull/7) 2023-06-07.

Wow, that's not bad. It runs full context at 15 tokens/s, 2X faster than the llama.cpp HIP code:
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

python test_benchmark_inference.py -d ~/llm/models/Manticore-13B-GPTQ/ -p                       (llm) 
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
 ** Time, Load model: 7.24 seconds
 ** Time, Load tokenizer: 0.01 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,873.52 MB - [cuda:1] 0.00 MB
 -- Warmup pass 1...
 ** Time, Warmup: 4.77 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 4.54 seconds
 -- Inference, first pass.
 ** Time, Inference: 4.53 seconds
 ** Speed: 423.57 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 15.06 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 19.18 tokens/second
 ** VRAM, Inference: [cuda:0] 2,253.08 MB - [cuda:1] 0.00 MB
 ** VRAM, Total: [cuda:0] 9,126.60 MB - [cuda:1] 0.00 MB
```

## RTX 4090 Comparison
As a point of comparison, running the same model on my stock RTX 4090 on `llama.cpp` with `make LLAMA_CUBLAS=1` runs at about 21 tokens/s:
```
llama_print_timings:        load time =  2761.96 ms
llama_print_timings:      sample time =   864.29 ms /  2048 runs   (    0.42 ms per token)
llama_print_timings: prompt eval time =  6186.81 ms /  1801 tokens (    3.44 ms per token)
llama_print_timings:        eval time = 97018.52 ms /  2040 runs   (   47.56 ms per token)
llama_print_timings:       total time = 106992.91 ms
```
And while not exactly 1:1 (since we're comparing vs q5_1), running the 4bit GPTQ on [exllama](https://github.com/turboderp/exllama), we get up to 78.75 tokens/s, about 3.7X faster than llama.cpp on the same GPU, and about 12X faster performance than the Radeon VII:
```
python test_benchmark_inference.py -p -d /models/llm/manticore/manticore-13b-chat-pyg-GPTQ

** Time, Load model: 1.41 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,683.17 MB
 -- Warmup pass 1...
 ** Time, Warmup: 1.50 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 0.36 seconds
 -- Warmup pass 3...
 ** Time, Warmup: 0.35 seconds
 -- Inference, first pass.
 ** Time, Inference: 0.36 seconds
 ** Speed: 5391.67 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 78.75 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 78.77 tokens/second
 ** VRAM, Inference: [cuda:0] 2,254.17 MB
 ** VRAM, Total: [cuda:0] 8,937.34 MB
```
On 2023-06-12 the current HEAD is running faster (6% at full context, 25% at smaller context), so use theses numbers just as a ballpark...
```
 ** Time, Load model: 1.34 seconds
 ** Time, Load tokenizer: 0.00 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,873.52 MB
 -- Warmup pass 1...
 ** Time, Warmup: 0.65 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 0.26 seconds
 -- Inference, first pass.
 ** Time, Inference: 0.26 seconds
 ** Speed: 7425.85 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 83.39 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 98.84 tokens/second
 ** VRAM, Inference: [cuda:0] 1,772.67 MB
 ** VRAM, Total: [cuda:0] 8,646.19 MB
```
## Recommendation
Radeon VII 16GB cards are going for about $250-$300 on eBay (equivalent to an Instinct MI50 which range a lot in price; MI60 or MI100 are similar also similar generation cards but with 32GB of RAM). 

For the performance, you're much better off paying about [$200](https://www.ebay.com/itm/195565620918) ([alt](https://www.ebay.com/itm/195745134833)) for an Nvidia Tesla P40 24GB (1080Ti class but with more RAM) or about $700 for an RTX 3090 24GB. The P40 can [reportedly run 13b models at about 15 tokens/s](https://www.reddit.com/r/LocalLLaMA/comments/13n8bqh/my_results_using_a_tesla_p40/), over 2X faster than a Radeon VII and with lots more software support. Also, 24GB cards support 30b models, which 16GB cards can't do.

## Bonus: GTX 1080 Ti Comparison
I dug out my old GTX 1080 Ti and installed it to get a ballpark vs P40 numbers. 

As of [2023-06-07](https://github.com/ggerganov/llama.cpp/pull/1703), llama.cpp's CUDA support was refactored to for multi-GPU support (you can use `CUDA_VISIBLE_DEVICES` to force a device)

My 1080 Ti has 11GB of VRAM, which means it can't fit a q5_1 w/ full context, but here was a smaller `llama.cpp` run that gives a ballpark. It gets ~5.8 tokens/s, a similar speed to the Radeon VII:
```
export CUDA_VISIBLE_DEVICES=1
./main -m /data/ai/models/llm/manticore/Manticore-13B-Chat-Pyg.ggmlv3.q5_1.bin -ngl 40

llama_print_timings:        load time =  3365.11 ms
llama_print_timings:      sample time =   193.96 ms /   475 runs   (    0.41 ms per token)
llama_print_timings: prompt eval time =   392.34 ms /     2 tokens (  196.17 ms per token)
llama_print_timings:        eval time = 81143.00 ms /   474 runs   (  171.19 ms per token)
llama_print_timings:       total time = 84773.38 ms
```
`exllama` now runs on Pascal GPUs. It inferences at full context at about 5.9 tokens/second (the app reports 8.7 GiB of memory used, but `nvidia-smi` puts it at 9.4 GiB). Warmup for some reason takes a **very** long time:
```
 -- Tokenizer: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/tokenizer.model
 -- Model config: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/config.json
 -- Model: /data/ai/models/llm/manticore/manticore-13b-chat-pyg-GPTQ/Manticore-13B-Chat-Pyg-GPTQ-4bit-128g.no-act-order.safetensors
 -- Sequence length: 2048
 -- Tuning:
 -- --matmul_recons_thd: 8
 -- --fused_mlp_thd: 2
 -- --sdp_thd: 8
 -- Options: ['perf']
 ** Time, Load model: 3.60 seconds
 ** Time, Load tokenizer: 0.00 seconds
 -- Groupsize (inferred): 128
 -- Act-order (inferred): no
 ** VRAM, Model: [cuda:0] 6,873.52 MB
 -- Warmup pass 1...
 ** Time, Warmup: 245.24 seconds
 -- Warmup pass 2...
 ** Time, Warmup: 244.93 seconds
 -- Inference, first pass.
 ** Time, Inference: 244.96 seconds
 ** Speed: 7.84 tokens/second
 -- Generating 128 tokens, 1920 token prompt...
 ** Speed: 5.87 tokens/second
 -- Generating 128 tokens, 4 token prompt...
 ** Speed: 6.11 tokens/second
 ** VRAM, Inference: [cuda:0] 1,772.67 MB
 ** VRAM, Total: [cuda:0] 8,646.19 MB
```
I was personally unable to get GPTQ-for-LLama running on it.

# Windows

## llama.cpp

Go to [llama.cpp's release page](https://github.com/ggerganov/llama.cpp/releases) and download a `bin-win-clblast` version.

In the Windows terminal, run it with `-ngl 99` to load all the layers into memory.

```
.\main.exe -m model.bin -ngl 99
```

Performance on an RTX 7900XT was not impressive (about 9 t/s on a 13b-q5_1 model) but better than nothing (about 2X clblast w/o loading the layers into GPU memory).