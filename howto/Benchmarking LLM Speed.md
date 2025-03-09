NOTE: This document tries to avoid using the term "performance" since in ML research the term performance typically refers to measuring model quality/capabilities.

This is a cheat sheet for running a simple benchmark on consumer hardware for LLM inference using the most popular end-user inferencing engine, `llama.cpp` and its included `llama-bench`. Feel free to skip to the HOWTO section if you want.
- If you're looking for more information and industrial benchmarking, the best place to start is probably Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering/), particularly the [Compute/Accelerator](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator) section. Last year I did a long writeup on [tuning vLLM for MI300X](https://shisa.ai/posts/tuning-vllm-mi300x/) that links to many resources as a good starting point as well. I recommend using `mamf-finder.py` for hardware testing, and vLLM/SGLang's `benchmark_serving` implementations to generate throughput, TTFT, TPOT measurements at different concurrencies.

# Background and Definitions

## Large Language Model (LLM) Basics
- Parameters: LLMs are basically a big pile of numbers (matrices) They have different sizes, which is **parameter** count - When you see 7B, 8B, 14B, this is an approximate count of how many parameters (in billions) they are. Dense models activate all parameters on every token generated. Mixture of Expert (MoE) models typically only activate a percentage of parameters
- Quantization - Models weights (the parameters) used to be stored as FP32 (4 bytes), then FP16/BF16 (2 bytes). Commercially, FP8 and INT8 are quite common, and FP4 and INT4 are emerging. At home, "Q4", which is roughly ~4-bit is used most often used, but there are even smaller quants that are usable these days (down to ~1.58b). Note, performance loss is not linear - a good Q4 quant can be close to or for PTQ quants, can even surpass unquanted performance with the proper calibration set. Most home users will probably be running Q4 quants that generally only lose a few percentage points in quality while taking almost 4 times less memory and running 4 times faster than the FP16/BF16 full precision versions
- Weights, Activations, Computational Precision - when we talk about precision, there are actually are differences between the weights, activations and the actual computational precision. Just mentioning this to head off some confusion
- Almost all popular desktop tools like LM Studio, Ollama, Jan, AnythingLLM run `llama.cpp` as their inference backend. `llama.cpp` has [many backends](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#supported-backends) - Metal for Apple Silicon, CUDA, HIP (ROCm), Vulkan, and SYCL among them (for Intel GPUs, Intel maintains a fork with an [IPEX-LLM backend](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md) that performs much better than the upstream SYCL version).
	- [AMD GPUs](https://llm-tracker.info/howto/AMD-GPUs) - the most comprehensive guide on running AI/ML software on AMD GPUs
	- [Intel GPUs](https://llm-tracker.info/howto/Intel-GPUs) - some notes and testing w/ Intel Arc GPUs

# HOWTO
## Testing Notes
- It's important to record the exact version/build numbers of the `llama.cpp` software as they can have big changes on speed. Generally, you should just run the [latest release](https://github.com/ggml-org/llama.cpp/releases), as new models, features, and bugfixes are constantly being rolled out and old versions go stale very quickly.
- You should pick standard models for testing. I recommend using [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF#provided-files) as a baseline as it runs on almost everything (<4GB) and is fairly quick and to run and it is the model used by several long-running benchmarks discussions:
	- [Performance of llama.cpp on Apple Silicon M-series #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
	- [Performance of llama.cpp with Vulkan #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
	- Some of my benchmark posts with the same model:
		- [llama.cpp Compute and Memory Bandwidth Efficiency w/ Different Devices/Backends](https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/)
		- [Testing llama.cpp with Intel's Xe2 iGPU (Core Ultra 7 258V w/ Arc Graphics 140V) ](https://www.reddit.com/r/LocalLLaMA/comments/1gheslj/testing_llamacpp_with_intels_xe2_igpu_core_ultra/)
- You might want to download a big model as well to test. Something like a [Llama 3 70B Instruct Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF#download-a-file-not-the-whole-branch-from-below) (42.5GiB) might be good.
- You can also try speculative decoding. If so you can refer to my docs/code:
	- [Revisting llama.cpp speculative decoding w/ Qwen2.5-Coder 32B (AMD vs Nvidia results)](https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/revisting_llamacpp_speculative_decoding_w/)
	- https://github.com/AUGMXNT/speed-benchmarking/tree/main/llama.cpp-code

## Simple Benchmark
* Download the model, eg direct link: https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
* Download and unzip the latest release: https://github.com/ggml-org/llama.cpp/releases
```
# Default
build/bin/llama-bench -m $MODEL_PATH

# w/ Flash Attention enabled
build/bin/llama-bench -m $MODEL_PATH -fa 1
```
- This runs with prompt processing (compute limited) pp512 and token generation (memory limited) tg128 5 times and outputs a nice table for you - with this you can repeatedly test and compare these numbers in a much more reliable fashion
	- pp512 measures how fast you will process context/existing conversation history - eg, if you have 4000 tokens of context and your pp is 100 tok/s, you will have to wait 40s before you start generating any tokens.
	- tg128 is a measure of how fast your device will generate new tokens for a single user (batch size = 1, bs=1)
- Generally GGUF models take up the model size in memory for weights with additional memory required for the kvcache depending on your max context size. You can therefore do a rough speed calculation of tok/s by dividing MBW/GGUF size
- There are a lot of additional options - the big ones are if you have limited device memory and need to load specific layers with `-ngl` - if you get errors from running out of memory, lower the layer count until it all fits. The rest will be offloaded to system memory
- If you have multiple devices, you may need to set `GGML_VK_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` to select which ones you want to use/test. Typically inference will run as an average of your different device speeds
- It's best to benchmark headless devices of course, but if you have to use a GPU for display, try to have it running as little as possible. It'd be best to SSH in remotely to do your testing.
- `nvtop` (which works w/ Nvidia and AMD GPUs) is a useful tool for realtime debugging. `nvidia-smi` and `rocm-smi` can be used for logging runs (especially memory highwater marks, power consumption)
- Flash Attention lowers memory usage for context and it also slightly increases speed for CUDA, but can make other devices dramatically slower (due to limitations of llama.cpp's current FA implementation, not intrinsic to hardware or the Flash Attention algorithm)

## More comprehensive benchmarks
- use `nvidia-smi` and `rocm-smi` to track power usage
- You can [adjust power limits and see how inference performance is affected](https://www.reddit.com/r/LocalLLaMA/comments/1hg6qrd/relative_performance_in_llamacpp_when_adjusting/)
- `llama-bench` is good for a basic repeatable test for max throughput, but you probably want to use something like vLLM's benchmark for testing Time To First Token (TTFT) and Time Per Output Token (TPOT)
- As mentioned, llama.cpp supports speculative decode
- If you want to test multi-user or batched output (eg, you want to just process a lot of text), the llama.cpp kernels are usually not very good and you'll want to check out vLLM, SGLang, TGI, etc.
- If you have >130GB of memory, you can give some of these R1 quants a try: https://unsloth.ai/blog/deepseekr1-dynamic

# Examples

## llama.cpp on WSL2 on Windows
```
# Install WSL if necessary: https://learn.microsoft.com/en-us/windows/wsl/install
# this defaults to Ubuntu, is fine/recommended
wsl --install

# Assuming you have AMD Software: Adrenalin Edition 25.3.1 installed
# follow install instructions: # https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html
# There's no clean upgrade so be sure to `sudo amdgpu-uninstall` first if upgrading ROCm versions

# Note a known issue with 25.3.1: Intermittent build error may be observed when running ROCM/HIP workloads using CMake. Users experiencing this issue are recommended to replace the Native Linux library filename (for example `libhsa-runtime64.so.1.14.60304`) in `/opt/rocm/lib/cmake/hsa-runtime64/hsa-runtime64Targets-relwithdebinfo.cmake` with the WSL library filename `libhsa-runtime64.so.1.14.0` as a temporary workaround.
# You can fix it with this one-liner (backup the file first if you're concerned)
sed -i 's/libhsa-runtime64\.so\.1\.14\.60304/libhsa-runtime64.so.1.14.0/g' /opt/rocm/lib/cmake/hsa-runtime64/hsa-runtime64Targets-relwithdebinfo.cmake

# llama.cpp install
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Basically we can use the recommended compile command
# https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -- -j

# Get the standard test model
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

# run llama-bench
❯ time llama-bench -m llama-2-7b.Q4_0.gguf
```
- https://learn.microsoft.com/en-us/windows/wsl/install
- https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-25-3-1.html
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html
- https://rocm.docs.amd.com/projects/radeon/en/latest/docs/limitations.html#wsl-specific-issues
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip