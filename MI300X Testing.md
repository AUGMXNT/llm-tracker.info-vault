For more general info on running AI/ML on AMD GPUS: https://llm-tracker.info/howto/AMD-GPUs

See the repo here as well: https://github.com/AUGMXNT/MI300-testing

[Hot Aisle](https://hotaisle.xyz/) reached out on if I was still interested in benchmarking/testing out one of their new MI300X systems. Here's some other MI300X benchmarks and analysis: https://hotaisle.xyz/benchmarks-and-analysis/

# System Info
The system I am testing is an 8 x MI300X big boy. Here's the basic info:
```
OS: Ubuntu jammy 22.04 x86_64
Host: PowerEdge XE9680
Kernel: Linux 6.8.0-45-generic
CPU: Intel(R) Xeon(R) Platinum 8470 (208) @ 3.80 GHz
GPU 1: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 2: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 3: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 4: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 5: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 6: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 7: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
GPU 8: AMD gfx942:sramecc+:xnack- (304) @ 2.10 GHz (191.98 GiB) [Discrete]
Memory: 52.62 GiB / 1.97 TiB (3%)
Swap: 768.00 KiB / 8.00 GiB (0%)
Disk (/): 68.59 GiB / 437.94 GiB (16%) - ext4
Disk (/mnt/nvme0n1p1): 3.33 TiB / 13.86 TiB (24%) - ext4
Disk (/mnt/nvme1n1p1): 19.99 GiB / 13.86 TiB (0%) - ext4
Disk (/mnt/nvme2n1p1): 28.00 KiB / 13.86 TiB (0%) - ext4
Disk (/mnt/nvme5n1p1): 28.00 KiB / 13.86 TiB (0%) - ext4
Disk (/mnt/nvme6n1p1): 28.00 KiB / 13.86 TiB (0%) - ext4
Disk (/mnt/nvme7n1p1): 28.00 KiB / 13.86 TiB (0%) - ext4
Disk (/mnt/nvme8n1p1): 28.00 KiB / 13.86 TiB (0%) - ext4
Locale: en_US.UTF-8
```

Just for fun:
- Why you might not want to trust Geekbench scores: https://browser.geekbench.com/v6/cpu/8182056
- PassMark seems to be a bit more accurate https://www.passmark.com/baselines/V11/display.php?id=507520112395
	- CPU btw is 2 x [Xeon Platinum 8470](https://www.intel.com/content/www/us/en/products/sku/231728/intel-xeon-platinum-8470-processor-105m-cache-2-00-ghz/specifications.html), which is 2 x 52C 104T w/ 2.0GHz base and 3.8GHz boost clock, 105MB of cache per chip, running DDR5-4800
	- The MT score puts it in the ballpark of a 9554P (64C 128T) 
# Inference
Let's start with inference.
## llama.cpp
This was done mostly for fun, I didn't expect very high numbers and I wasn't proven wrong.

First, let's give 8 GPUs a try. Prompt pre-processing is slower than a single 7900 XT, and text generation barely beats a single 4090.

Note: a single MI300X has a theoretical 1307.4 FP16 TFLOPS and 5.3 TB/s of MBW.

```
$ ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 8 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 1: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 2: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 3: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 4: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 5: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 6: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 7: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |       1333.08 ± 4.99 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |        174.99 ± 2.20 |

build: d5cb8684 (3891)
```

OK, now lets give a single card a try. A hair faster. So, zero scaling from multiple cards:
```
HIP_VISIBLE_DEVICES=0 time ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |      1334.37 ± 12.73 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |        183.18 ± 0.77 |

build: d5cb8684 (3891)
8.51user 1.52system 0:08.57elapsed 117%CPU (0avgtext+0avgdata 5281008maxresident)k
0inputs+12256outputs (1major+496848minor)pagefaults 0swaps
```

And finally, for lolz let's enable llama.cpp's Flash Attention implementation. Like for RDNA3, this causes a slowdown, although a bit smaller, percentage-wise:
```
$ HIP_VISIBLE_DEVICES=0 time ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |         pp512 |       1272.03 ± 7.97 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |         tg128 |        157.84 ± 0.35 |

build: d5cb8684 (3891)
9.13user 1.43system 0:09.23elapsed 114%CPU (0avgtext+0avgdata 5283472maxresident)k
0inputs+12256outputs (1major+496366minor)pagefaults 0swaps
```

Note, there was an unmerged CDNA optimization that increased perfomance by almost 10X, but it wasn't merged due to lack of maintainer. I tried wedging the changes in, but it didn't work:
- https://github.com/ggerganov/llama.cpp/pull/8082
- https://github.com/ggerganov/llama.cpp/pull/8082/files
In the merge, this unmerged FA fix is mentioned which is another 2X performance boost apparently. These files have changed even more, so after the previous unsuccessful edits, I didn't bother to apply this, but maybe I'll give it a try at some point with an AI coding tool:
- https://github.com/ggerganov/llama.cpp/pull/7011
	- This is where the discussion on lack of an AMD maintainer for `llama.cpp` - while `llama.cpp` is unlikely to be used with CDNA atm, it's probably the most used inference engine on desktop CPUs and GPUs in the world...
## vLLM 
We are testing around 2024-10-07 and our source build is `v0.6.3.dev114+g4f95ffee`.

There are a few other vLLM benchmarks published, but they are all done before the latest massive [vLLM 0.6.0 performance enhancements](https://blog.vllm.ai/2024/09/05/perf-update.html) so this may be of interest.

### Install
We are also running with ROCm 6.2.2, and are using the current PyTorch nightly.
```
# Environment
mamba create -n vllm python=3.11
mamba activate vllm

# PyTorch Nightly + Triton
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
pip install triton

# AMD SMI - permission issues if you don't copy the folder
cp -r /opt/rocm/share/amd_smi ./
cd amd_smi
pip install .
cd ..

# vLLM time
git clone https://github.com/vllm-project/vllm
cd vllm

# Dependencies
pip install -U numba scipy huggingface-hub
pip install "numpy<2"
pip install -r requirements-rocm.txt

# Undocumented dependencies
pip install setuptools_scm

# Newer version of cmake needed
mamba install cmake

# Build for MI300
export PYTORCH_ROCM_ARCH="gfx942"
python setup.py develop

python -c "import vllm; print(vllm.__version__)"
```
- https://pytorch.org/get-started/locally/
- https://docs.vllm.ai/en/v0.5.5/getting_started/amd-installation.html
### Flash Attention
By default, vLLM defaults to the Triton Flash Attention implementation, however, there are some issues:
```
WARNING 10-07 15:49:20 registry.py:198] Model architecture MistralForCausalLM is partially supported by ROCm: Sliding window attention (SWA) is not yet supported in Triton flash attention. For ha
lf-precision SWA support, please use CK flash attention by setting `VLLM_USE_TRITON_FLASH_ATTN=0
```

The vLLM docs suggest you install the [ROCm/flash-attention](https://github.com/ROCm/flash-attention) fork ... but it doesn't work for me. The official upstream [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) claims ROCm support for MI300s! So let's try it.

```
export TORCH_CUDA_ARCH_LIST="gfx942"
export HIPCC_FLAGS="--offload-arch=gfx942"
export CXXFLAGS="--offload-arch=gfx942"
export HCC_AMDGPU_TARGET=gfx942
TORCH_CUDA_ARCH_LIST="8.9" python setup.py install
```

Let's exactly follow these docs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html#installing-flash-attention-2

Looks like our errors relate to the translation of `getCurrentHIPStream` from `getCurrentCUDAStream`?
```
FAILED: /home/hotaisle/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn_ck/mha_fwd.o 
/opt/rocm-6.2.2/bin/hipcc  -I/home/hotaisle/flash-attention/csrc/composable_kernel/include -I/home/hotaisle/flash-attention/csrc/composable_kernel/library/include -I/home/hotaisle/flash-attention/csrc/composable_kernel/example/ck_tile/01_fmha -I/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include -I/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/TH -I/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/THC -I/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/THH -I/opt/rocm-6.2.2/include -I/home/hotaisle/miniforge3/envs/llm/include/python3.11 -c -c /home/hotaisle/flash-attention/csrc/flash_attn_ck/mha_fwd.hip -o /home/hotaisle/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn_ck/mha_fwd.o -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 --offload-arch=gfx942 -O3 -std=c++17 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -DCK_ENABLE_BF16 -DCK_ENABLE_BF8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_FP8 -DCK_ENABLE_INT8 -DCK_USE_XDL -DUSE_PROF_API=1 -D__HIP_PLATFORM_HCC__=1 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 -fno-offload-uniform-block -mllvm -enable-post-misched=0 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -mllvm -amdgpu-coerce-illegal-types=1 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -fno-gpu-rdc
/home/hotaisle/flash-attention/csrc/flash_attn_ck/mha_fwd.hip:277:33: error: no member named 'getCurrentHIPStream' in namespace 'at::cuda'; did you mean 'getCurrentCUDAStream'?
  277 |         auto stream = at::cuda::getCurrentHIPStream().stream();
      |                       ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~
      |                                 getCurrentCUDAStream
/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/c10/hip/HIPStream.h:244:20: note: 'getCurrentCUDAStream' declared here
  244 | C10_API CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);
      |                    ^
1 error generated when compiling for gfx942.
```

This also happens when following the docs trying to install `xformers` https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html#xformers:
```
home/hotaisle/xformers/xformers/csrc/attention/hip_fmha/attention_backward_generic_ck_tiled.hip:116:34: error: no member named 'getCurrentHIPStream' in namespace 'at::cuda'; did you mean 'getCurrentCUDAStream'?
  116 |   hipStream_t stream = at::cuda::getCurrentHIPStream().stream();
      |                        ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~
      |                                  getCurrentCUDAStream
/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/include/c10/hip/HIPStream.h:244:20: note: 'getCurrentCUDAStream' declared here
  244 | C10_API CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);
      |                    ^
1 error generated when compiling for gfx942.
```

https://chatgpt.com/c/670805a7-bb3c-8012-b84a-ef9926ef9546
```
# We need this otherwise compiles will use distutil and be single threaded
mamba install ninja
```

#### Docker
```
sudo apt install docker-buildx

```

### hipblastlt
2024-10-10: Bug filed w/ PyTorch https://github.com/pytorch/pytorch/issues/137695
(maybe file a bug w/ vLLM too?)

Works with 1
```
$ python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 --model meta-llama/Llama-2-7b-chat-hf

INFO 10-10 05:34:53 gpu_executor.py:122] # GPU blocks: 20186, # CPU blocks: 512
INFO 10-10 05:34:53 gpu_executor.py:126] Maximum concurrency for 4096 tokens per request: 78.85x
INFO 10-10 05:34:54 model_runner.py:1385] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-10 05:34:54 model_runner.py:1389] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-10 05:35:01 model_runner.py:1513] Graph capturing finished in 7 secs.
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:42<00:00, 23.32it/s, est. speed input: 11938.87 toks/s, output: 2984.72 toks/s]
Throughput: 23.15 requests/s, 14813.60 tokens/s

INFO 10-10 05:40:26 model_runner.py:1062] Loading model weights took 12.5523 GB
INFO 10-10 05:40:43 gpu_executor.py:122] # GPU blocks: 20186, # CPU blocks: 512
INFO 10-10 05:40:43 gpu_executor.py:126] Maximum concurrency for 4096 tokens per request: 78.85x
INFO 10-10 05:40:43 model_runner.py:1385] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-10 05:40:43 model_runner.py:1389] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-10 05:40:50 model_runner.py:1513] Graph capturing finished in 7 secs.
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:42<00:00, 23.59it/s, est. speed input: 12080.41 toks/s, output: 3020.10 toks/s]
Throughput: 23.42 requests/s, 14987.04 tokens/s
```

w/o
```
INFO 10-10 05:42:39 model_runner.py:1062] Loading model weights took 12.5523 GB
INFO 10-10 05:42:43 gpu_executor.py:122] # GPU blocks: 20406, # CPU blocks: 512
INFO 10-10 05:42:43 gpu_executor.py:126] Maximum concurrency for 4096 tokens per request: 79.71x
INFO 10-10 05:42:44 model_runner.py:1385] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-10 05:42:44 model_runner.py:1389] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-10 05:42:51 model_runner.py:1513] Graph capturing finished in 7 secs.
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:39<00:00, 25.20it/s, est. speed input: 12903.22 toks/s, output: 3225.80 toks/s]
Throughput: 24.99 requests/s, 15995.82 tokens/s
```

Hmm, no hipblaslt is faster. confirmed w/ tp 2, 4 - tp 8 causes errors (2 threads can't load)?
need to file bug?

rocblaslt error: Could not load /home/hotaisle/miniforge3/envs/vllm/lib/python3.11/site-packages/torch/lib/hipblaslt/library/TensileLibrary_lazy_gfx942.dat

### Executors 
https://www.nonbios.ai/post/deploying-large-405b-models-in-full-precision-on-runpod
- didn't work.

--distributed-executor-backend ray
```
(RayWorkerWrapper pid=769965) INFO 10-07 15:58:23 selector.py:121] Using ROCmFlashAttention backend.
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] Error executing method init_device. This might cause deadlock in distributed execution.
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] Traceback (most recent call last):
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]   File "/home/hotaisle/vllm/vllm/worker/worker_base.py", line 456, in execute_method
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]     return executor(*args, **kwargs)
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]   File "/home/hotaisle/vllm/vllm/worker/worker.py", line 166, in init_device
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]     torch.cuda.set_device(self.device)
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]   File "/home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/cuda/__init__.py", line 478, in set_device
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464]     torch._C._cuda_setDevice(device)
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] RuntimeError: CUDA error: invalid device ordinal
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] For debugging consider passing AMD_SERIALIZE_KERNEL=3
(RayWorkerWrapper pid=769965) ERROR 10-07 15:58:23 worker_base.py:464] Device-side assertion tracking was not enabled by user.

```
### Performance
https://blog.vllm.ai/2024/09/05/perf-update.html


#### Basic Benchmark (1GPU)
1 x MI300
- 1000 x 512;128
	- 23.28 it/s
	- input: 11918.99 tok/s
	- output: 2979.50 tok/s
	- Throughput: 23.10 requests/s, 14787.00 tokens/s
```
$ python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 --model meta-llama/Llama-2-7b-chat-hf
WARNING 10-07 09:54:08 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', dataset=None, input_len=512, output_len=128, model='meta-llama/Llama-2-7b-chat-hf', tokenizer='meta-llama/Llama-2-7b-chat-hf', quantization=None, tensor_parallel_size=1, n=1, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', num_scheduler_steps=1, use_v2_block_manager=False, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto', disable_async_output_proc=False, async_engine=False, disable_frontend_multiprocessing=False)
INFO 10-07 09:54:22 config.py:904] Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.
INFO 10-07 09:54:22 llm_engine.py:237] Initializing an LLM engine (v0.6.3.dev114+g4f95ffee) with config: model='meta-llama/Llama-2-7b-chat-hf', speculative_config=None, tokenizer='meta-llama/Llama-2-7b-chat-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-2-7b-chat-hf, use_v2_block_manager=False, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
INFO 10-07 09:54:22 selector.py:121] Using ROCmFlashAttention backend.
INFO 10-07 09:54:22 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
INFO 10-07 09:54:22 selector.py:121] Using ROCmFlashAttention backend.
INFO 10-07 09:54:23 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:00<00:00, 53.30it/s]

INFO 10-07 09:54:29 model_runner.py:1060] Loading model weights took 12.5523 GB
INFO 10-07 09:55:00 gpu_executor.py:122] # GPU blocks: 20186, # CPU blocks: 512
INFO 10-07 09:55:01 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-07 09:55:01 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-07 09:55:09 model_runner.py:1511] Graph capturing finished in 8 secs.
Processed prompts: 100%|█████████████████| 1000/1000 [00:42<00:00, 23.28it/s, est. speed input: 11917.99 toks/s, output: 2979.50 toks/s]
Throughput: 23.10 requests/s, 14787.00 tokens/s
[rank0]:[W1007 09:55:53.400347903 ProcessGroupNCCL.cpp:1253] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

#### Basic Benchmark (TP 8)
8 x MI300
- 1000 x 512;128
	- 55.93 it/s
	- input: 28636.95 tok/s
	- output: 7159.24  tok/s
	- Throughput: 54.94 requests/s, 35163.95 tokens/s


|                | Default  | float16  | bfloat16 | mp       |
| -------------- | -------- | -------- | -------- | -------- |
| it/s           | 55.93    | 55.06    | 52.28    | 42.59    |
| input (tok/s)  | 28636.95 | 28191.54 | 26766.27 | 21805.15 |
| output (tok/s) | 7159.24  | 7047.88  | 6691.57  | 5451.29  |
| tp (req/s)     | 54.94    | 54.07    | 51.41    | 41.96    |
| tp (tok/s)     | 35163.95 | 34606.80 | 32902.67 | 26856.43 |
- default = float16
- ray hangs, but mp is slower so use just use the default (none)
- hipblaslt doesn't work even when symlinked to the proper one so... not tested
- https://docs.vllm.ai/en/latest/serving/distributed_serving.html is beta, not supported by benchmark



```
$ TORCH_BLAS_PREFER_HIPBLASLT=0 python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 --model meta-llama/Llama-2-7b-chat-hf -tp 8
WARNING 10-07 10:19:56 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', dataset=None, input_len=512, output_len=128, model='meta-llama/Llama-2-7b-chat-hf', tokenizer='meta-llama/Llama-2-7b-chat-hf', quantization=None, tensor_parallel_size=8, n=1, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', num_scheduler_steps=1, use_v2_block_manager=False, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto', disable_async_output_proc=False, async_engine=False, disable_frontend_multiprocessing=False)
INFO 10-07 10:20:09 config.py:875] Defaulting to use mp for distributed inference
INFO 10-07 10:20:09 config.py:904] Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.
INFO 10-07 10:20:09 llm_engine.py:237] Initializing an LLM engine (v0.6.3.dev114+g4f95ffee) with config: model='meta-llama/Llama-2-7b-chat-hf', speculative_config=None, tokenizer='meta-llama/Llama-2-7b-chat-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=8, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-2-7b-chat-hf, use_v2_block_manager=False, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
WARNING 10-07 10:20:10 multiproc_gpu_executor.py:53] Reducing Torch parallelism from 104 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
INFO 10-07 10:20:10 custom_cache_manager.py:17] Setting Triton cache manager to: vllm.triton_utils.custom_cache_manager:CustomCacheManager
INFO 10-07 10:20:10 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:14 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:14 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:15 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:15 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:15 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:15 pynccl.py:63] vLLM is using nccl==2.20.5
INFO 10-07 10:20:16 shm_broadcast.py:241] vLLM message queue communication handle: Handle(connect_ip='127.0.0.1', local_reader_ranks=[1, 2, 3, 4, 5, 6, 7], buffer=<vllm.distributed.device_communicators.shm_broadcast.ShmRingBuffer object at 0x756903e0cd90>, local_subscribe_port=40625, remote_subscribe_port=None)
INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:16 model_runner.py:1049] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:16 selector.py:121] Using ROCmFlashAttention backend.
INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:16 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:00<00:00, 263.17it/s]

INFO 10-07 10:20:17 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:18 model_runner.py:1060] Loading model weights took 1.5874 GB
INFO 10-07 10:20:37 distributed_gpu_executor.py:57] # GPU blocks: 169460, # CPU blocks: 4096
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:46 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362294) INFO 10-07 10:20:46 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:46 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362293) INFO 10-07 10:20:46 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-07 10:20:47 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-07 10:20:47 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:47 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362292) INFO 10-07 10:20:47 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:50 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362296) INFO 10-07 10:20:50 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:51 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362297) INFO 10-07 10:20:51 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:51 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362295) INFO 10-07 10:20:51 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:51 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=362298) INFO 10-07 10:20:51 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=362293) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 20 secs.
(VllmWorkerProcess pid=362294) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 20 secs.
(VllmWorkerProcess pid=362297) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 15 secs.
(VllmWorkerProcess pid=362296) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 16 secs.
(VllmWorkerProcess pid=362295) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 15 secs.
INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 19 secs.
(VllmWorkerProcess pid=362292) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 19 secs.
(VllmWorkerProcess pid=362298) INFO 10-07 10:21:06 model_runner.py:1511] Graph capturing finished in 15 secs.
Processed prompts: 100%|█████████████████| 1000/1000 [00:17<00:00, 55.93it/s, est. speed input: 28636.95 toks/s, output: 7159.24 toks/s]
INFO 10-07 10:21:25 multiproc_worker_utils.py:134] Terminating local vLLM worker processes
(VllmWorkerProcess pid=362292) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362296) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362298) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362293) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362295) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362294) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=362297) INFO 10-07 10:21:25 multiproc_worker_utils.py:242] Worker exiting
Throughput: 54.94 requests/s, 35163.95 tokens/s
[rank0]:[W1007 10:21:26.555187420 ProcessGroupNCCL.cpp:1253] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
/home/hotaisle/miniforge3/envs/llm/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

NOTE: PyTorch currently includes an incomplete hipblaslt that's a PITA, and the easiest way to bypass this is with `TORCH_BLAS_PREFER_HIPBLASLT=0` to avoid this blowing things up:
```
rocblaslt error: Could not load /home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/lib/hipblaslt/library/TensileLibrary_lazy_gfx942.dat
```
- https://www.reddit.com/r/ROCm/comments/1dkkxgo/tryiing_to_get_torchtune_working_with_rocm_for/
- https://github.com/pytorch/torchtune/discussions/1108

Alternatively, since the lib exists in the `/opt/rocm` folder, lets try:
```
cd /home/hotaisle/miniforge3/envs/llm/lib/python3.11/site-packages/torch/lib
mv hipblaslt hipblaslt.pytorch
ln -s /opt/rocm-6.2.2/lib/hipblaslt
mv rocblas rocblas.pytorch
ln -s /opt/rocm-6.2.2/lib/rocblas
```
Hmm, still complains.  Whevs.

### FP8 kvcache
This will 
```

Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:19<00:00, 52.39it/s, est. speed input: 26824.25 toks/s, output: 6706.06 toks/s]
Throughput: 51.52 requests/s, 32973.71 tokens/s

# FP16
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:19<00:00, 51.94it/s, est. speed input: 26593.27 toks/s, output: 6648.31 toks/s]
Throughput: 51.07 requests/s, 32685.36 tokens/s
```

### Big Model Testing
#### Llama3 405B
With 1.5TB of VRAM, a full, unquantized (FP16) Llama 3 405B fits on single MI300X node:

- 1000 x 512;128
	- 4.33 it/s
	- input: 2216.24 tok/s
	- output: 554.06  tok/s
	- Throughput: 4.32 requests/s, 2765.68 tokens/s

GPUs go brr....
```
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
^[3m              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                   ^[0m
==========================================================================================================================
0       2     0x74a1,   55354  83.0°C      740.0W    NPS1, SPX, 0        1430Mhz  900Mhz  0%   auto  750.0W  97%    100%
1       3     0x74a1,   41632  72.0°C      740.0W    NPS1, SPX, 0        1427Mhz  900Mhz  0%   auto  750.0W  95%    100%
2       4     0x74a1,   47045  74.0°C      736.0W    NPS1, SPX, 0        1460Mhz  900Mhz  0%   auto  750.0W  95%    100%
3       5     0x74a1,   60169  81.0°C      740.0W    NPS1, SPX, 0        1409Mhz  900Mhz  0%   auto  750.0W  95%    100%
4       6     0x74a1,   56024  79.0°C      744.0W    NPS1, SPX, 0        1352Mhz  900Mhz  0%   auto  750.0W  95%    100%
5       7     0x74a1,   705    65.0°C      736.0W    NPS1, SPX, 0        1420Mhz  900Mhz  0%   auto  750.0W  95%    100%
6       8     0x74a1,   59108  82.0°C      743.0W    NPS1, SPX, 0        1368Mhz  900Mhz  0%   auto  750.0W  95%    100%
7       9     0x74a1,   10985  70.0°C      741.0W    NPS1, SPX, 0        1370Mhz  900Mhz  0%   auto  750.0W  95%    100%
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
```


```
(llm) hotaisle@ENC1-CLS01-SVR09:~/vllm$ TORCH_BLAS_PREFER_HIPBLASLT=0 python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 -tp 8 meta-llama/Llama-3.1-405B-Instruct
WARNING 10-07 10:27:53 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
usage: benchmark_throughput.py [-h] [--backend {vllm,hf,mii}] [--dataset DATASET] [--input-len INPUT_LEN] [--output-len OUTPUT_LEN]
                               [--model MODEL] [--tokenizer TOKENIZER]
                               [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}]
                               [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--n N] [--num-prompts NUM_PROMPTS] [--seed SEED]
                               [--hf-max-batch-size HF_MAX_BATCH_SIZE] [--trust-remote-code] [--max-model-len MAX_MODEL_LEN]
                               [--dtype {auto,half,float16,bfloat16,float,float32}] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                               [--enforce-eager] [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                               [--quantization-param-path QUANTIZATION_PARAM_PATH] [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu}]
                               [--num-scheduler-steps NUM_SCHEDULER_STEPS] [--use-v2-block-manager] [--enable-prefix-caching]
                               [--enable-chunked-prefill] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                               [--download-dir DOWNLOAD_DIR] [--output-json OUTPUT_JSON] [--distributed-executor-backend {ray,mp}]
                               [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,bitsandbytes}]
                               [--disable-async-output-proc] [--async-engine] [--disable-frontend-multiprocessing]
benchmark_throughput.py: error: unrecognized arguments: meta-llama/Llama-3.1-405B-Instruct
(llm) 2 hotaisle@ENC1-CLS01-SVR09:~/vllm$ TORCH_BLAS_PREFER_HIPBLASLT=0 python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 -tp 8 -m meta-llama/Llama-3.1-405B-Instruct
WARNING 10-07 10:28:06 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
usage: benchmark_throughput.py [-h] [--backend {vllm,hf,mii}] [--dataset DATASET] [--input-len INPUT_LEN] [--output-len OUTPUT_LEN]
                               [--model MODEL] [--tokenizer TOKENIZER]
                               [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}]
                               [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--n N] [--num-prompts NUM_PROMPTS] [--seed SEED]
                               [--hf-max-batch-size HF_MAX_BATCH_SIZE] [--trust-remote-code] [--max-model-len MAX_MODEL_LEN]
                               [--dtype {auto,half,float16,bfloat16,float,float32}] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                               [--enforce-eager] [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                               [--quantization-param-path QUANTIZATION_PARAM_PATH] [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu}]
                               [--num-scheduler-steps NUM_SCHEDULER_STEPS] [--use-v2-block-manager] [--enable-prefix-caching]
                               [--enable-chunked-prefill] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                               [--download-dir DOWNLOAD_DIR] [--output-json OUTPUT_JSON] [--distributed-executor-backend {ray,mp}]
                               [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,bitsandbytes}]
                               [--disable-async-output-proc] [--async-engine] [--disable-frontend-multiprocessing]
benchmark_throughput.py: error: unrecognized arguments: -m meta-llama/Llama-3.1-405B-Instruct
(llm) 2 hotaisle@ENC1-CLS01-SVR09:~/vllm$ TORCH_BLAS_PREFER_HIPBLASLT=0 python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 -tp 8 --model meta-llama/Llama-3.1-405B-Instruct
WARNING 10-07 10:28:26 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', dataset=None, input_len=512, output_len=128, model='meta-llama/Llama-3.1-405B-Instruct', tokenizer='meta-llama/Llama-3.1-405B-Instruct', quantization=None, tensor_parallel_size=8, n=1, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', num_scheduler_steps=1, use_v2_block_manager=False, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto', disable_async_output_proc=False, async_engine=False, disable_frontend_multiprocessing=False)
INFO 10-07 10:28:40 config.py:875] Defaulting to use mp for distributed inference
INFO 10-07 10:28:40 config.py:904] Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.
WARNING 10-07 10:28:40 arg_utils.py:964] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
INFO 10-07 10:28:40 llm_engine.py:237] Initializing an LLM engine (v0.6.3.dev114+g4f95ffee) with config: model='meta-llama/Llama-3.1-405B-Instruct', speculative_config=None, tokenizer='meta-llama/Llama-3.1-405B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=8, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-3.1-405B-Instruct, use_v2_block_manager=False, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
WARNING 10-07 10:28:40 multiproc_gpu_executor.py:53] Reducing Torch parallelism from 104 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
INFO 10-07 10:28:40 custom_cache_manager.py:17] Setting Triton cache manager to: vllm.triton_utils.custom_cache_manager:CustomCacheManager
INFO 10-07 10:28:41 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:45 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:45 multiproc_worker_utils.py:216] Worker ready; awaiting tasks
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:45 utils.py:1005] Found nccl from library librccl.so.1
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:45 pynccl.py:63] vLLM is using nccl==2.20.5
INFO 10-07 10:28:47 shm_broadcast.py:241] vLLM message queue communication handle: Handle(connect_ip='127.0.0.1', local_reader_ranks=[1, 2, 3, 4, 5, 6, 7], buffer=<vllm.distributed.device_communicators.shm_broadcast.ShmRingBuffer object at 0x70cc67250c10>, local_subscribe_port=49957, remote_subscribe_port=None)
INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:47 model_runner.py:1049] Starting to load model meta-llama/Llama-3.1-405B-Instruct...
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
INFO 10-07 10:28:47 selector.py:121] Using ROCmFlashAttention backend.
(VllmWorkerProcess pid=373710) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373708) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373711) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373712) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373714) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373709) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
(VllmWorkerProcess pid=373713) INFO 10-07 10:28:47 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/191 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  79% Completed | 150/191 [00:00<00:00, 1498.97it/s]
Loading safetensors checkpoint shards: 100% Completed | 191/191 [00:00<00:00, 1756.62it/s]

INFO 10-07 10:34:12 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373713) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373714) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373710) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373709) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373708) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373711) INFO 10-07 10:34:14 model_runner.py:1060] Loading model weights took 94.5351 GB
(VllmWorkerProcess pid=373712) INFO 10-07 10:34:15 model_runner.py:1060] Loading model weights took 94.5351 GB
INFO 10-07 10:35:22 distributed_gpu_executor.py:57] # GPU blocks: 51554, # CPU blocks: 4161
(VllmWorkerProcess pid=373712) INFO 10-07 10:35:24 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373712) INFO 10-07 10:35:24 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373713) INFO 10-07 10:35:24 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373713) INFO 10-07 10:35:24 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373711) INFO 10-07 10:35:24 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373711) INFO 10-07 10:35:24 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373714) INFO 10-07 10:35:24 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373714) INFO 10-07 10:35:24 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373708) INFO 10-07 10:35:26 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373708) INFO 10-07 10:35:26 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-07 10:35:26 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-07 10:35:26 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373709) INFO 10-07 10:35:26 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373709) INFO 10-07 10:35:26 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=373710) INFO 10-07 10:35:26 model_runner.py:1383] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=373710) INFO 10-07 10:35:26 model_runner.py:1387] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 24 secs.
(VllmWorkerProcess pid=373709) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 24 secs.
(VllmWorkerProcess pid=373710) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 24 secs.
(VllmWorkerProcess pid=373711) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 26 secs.
(VllmWorkerProcess pid=373713) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 26 secs.
(VllmWorkerProcess pid=373712) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 26 secs.
(VllmWorkerProcess pid=373714) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 26 secs.
(VllmWorkerProcess pid=373708) INFO 10-07 10:35:50 model_runner.py:1511] Graph capturing finished in 24 secs.
Processed prompts: 100%|███████████████████████████████████| 1000/1000 [03:51<00:00,  4.33it/s, est. speed input: 2216.24 toks/s, output: 554.06 toks/s]
INFO 10-07 10:39:41 multiproc_worker_utils.py:134] Terminating local vLLM worker processes
(VllmWorkerProcess pid=373708) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373711) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373709) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373714) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373710) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373713) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
(VllmWorkerProcess pid=373712) INFO 10-07 10:39:41 multiproc_worker_utils.py:242] Worker exiting
Throughput: 4.32 requests/s, 2765.68 tokens/s
[rank0]:[W1007 10:39:42.994846788 ProcessGroupNCCL.cpp:1253] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
/home/hotaisle/miniforge3/envs/llm/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

```



```
./install.sh -d -a "gfx942:xnack+;gfx942:xnack-"
```


#### Mistral Large
Sliding Window

|                | Default |
| -------------- | ------- |
| it/s           | 10.24   |
| input (tok/s)  | 5253.82 |
| output (tok/s) | 1310.89 |
| tp (req/s)     | 10.20   |
| tp (tok/s)     | 6525.23 |
- missing Tens

Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:37<00:00, 10.24it/s, est. speed input: 5253.82 toks/s, output: 1310.89 toks/s]

```
WARNING 10-07 15:49:20 registry.py:198] Model architecture MistralForCausalLM is partially supported by ROCm: Sliding window attention (SWA) is not yet supported in Triton flash attention. For ha
lf-precision SWA support, please use CK flash attention by setting `VLLM_USE_TRITON_FLASH_ATTN=0
```



Flashinfer
https://github.com/flashinfer-ai/flashinfer/pull/491

### SGLang
```
mamba create -n sglang python=3.11
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
pip install triton

# Nope
pip install "sglang[all]"

# compile

# Keep reinstalling updated torch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2 -U

# install vllm from source
```
- https://github.com/sgl-project/sglang/pull/1420
- https://github.com/sgl-project/sglang/pull/1453

```
# Server
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --attention-backend triton --sampling-backend pytorch --tp-size 8
--enable-torch-compile=False

# Client
curl http://localhost:30000/generate   -H "Content-Type: application/json"   -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 200,
      "temperature": 0
    }
  }'
```

Note: `--dp` is not supported. `--tp` gets weird memory errors.

```
(sglang) 130 hotaisle@ENC1-CLS01-SVR09:~/vllm$ python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 32 --input-len 256 --output-len 32
WARNING 10-10 22:30:30 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
[22:30:32 TP0] Init nccl begin.
[22:30:32 TP0] Load weight begin. avail mem=191.33 GB
[22:30:44 TP0] FlashInfer is not available on Non-NV platforms. Fallback to other kernel libraries.
[22:30:44 TP0] FlashInfer is not available on Non-NV platforms. Fallback to other kernel libraries.
[22:30:44 TP0] Skipping import of cpp extensions
[22:30:44 TP0] lm_eval is not installed, GPTQ may not be usable
INFO 10-10 22:30:45 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  2.03it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:02<00:02,  1.47s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:05<00:01,  1.90s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:07<00:00,  2.06s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:07<00:00,  1.84s/it]

[22:30:52 TP0] Load weight end. type=LlamaForCausalLM, dtype=torch.bfloat16, avail mem=176.33 GB
[22:30:53 TP0] Memory pool end. avail mem=22.83 GB
[22:31:06 TP0] Capture cuda graph begin. This can take up to several minutes.
max_total_num_tokens=1256440
Warmup ...
Prefill. latency: 0.37333 s, throughput:  21943.23 token/s
Decode.  latency: 0.01433 s, throughput:   2232.98 token/s
Decode.  latency: 0.00986 s, throughput:   3244.72 token/s
Decode.  latency: 0.00986 s, throughput:   3245.97 token/s
Decode.  latency: 0.00975 s, throughput:   3283.05 token/s
Decode.  latency: 0.00973 s, throughput:   3287.72 token/s
Decode.  median latency: 0.00976 s, median throughput:   3280.16 token/s
Total. latency:  0.446 s, throughput:  18928.16 token/s
Benchmark ...
Prefill. latency: 0.25074 s, throughput:  32671.13 token/s
Decode.  latency: 0.01015 s, throughput:   3151.69 token/s
Decode.  latency: 0.00984 s, throughput:   3252.58 token/s
Decode.  latency: 0.00978 s, throughput:   3272.09 token/s
Decode.  latency: 0.00979 s, throughput:   3268.50 token/s
Decode.  latency: 0.00975 s, throughput:   3283.21 token/s
Decode.  median latency: 0.00973 s, median throughput:   3289.81 token/s
Total. latency:  0.553 s, throughput:  16665.96 token/s
/home/hotaisle/miniforge3/envs/sglang/lib/python3.11/multiprocessing/resource_tracker.py:123: UserWarning: resource_tracker: process died unexpectedly, relaunching.  Some resources might leak.
  warnings.warn('resource_tracker: process died unexpectedly, '
Traceback (most recent call last):
  File "/home/hotaisle/miniforge3/envs/sglang/lib/python3.11/multiprocessing/resource_tracker.py", line 239, in main
    cache[rtype].remove(name)
KeyError: '/mp-dv7az532'
[rank0]:[W1010 22:31:13.301784377 ProcessGroupNCCL.cpp:1304] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

```

# Training

## torchtune
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html

https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx

Axolotl

PEFT
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html
https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning
https://rocm.blogs.amd.com/artificial-intelligence/starcoder-fine-tune/README.html

https://rocm.blogs.amd.com/artificial-intelligence/megatron-deepspeed-pretrain/README.html

https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/

## axolotl
Docker doesn't work:
```
$ sudo docker run --gpus '"all"' --rm -it winglian/axolotl:main-latest
Unable to find image 'winglian/axolotl:main-latest' locally
main-latest: Pulling from winglian/axolotl
aece8493d397: Pull complete 
dd4939a04761: Pull complete 
b0d7cc89b769: Pull complete 
1532d9024b9c: Pull complete 
04fc8a31fa53: Pull complete 
a14a8a8a6ebc: Pull complete 
7d61afc7a3ac: Pull complete 
8bd2762ffdd9: Pull complete 
2a5ee6fadd42: Pull complete 
22ba0fb08ae2: Pull complete 
4d37a6bba88f: Pull complete 
4bc954eb910a: Pull complete 
bd3d55680e04: Pull complete 
f797fda66265: Pull complete 
068d7f887619: Pull complete 
49a71fa9aaec: Pull complete 
a35b1ad7a4db: Pull complete 
4f4fb700ef54: Pull complete 
0b06795f16c0: Pull complete 
134f72e94be3: Pull complete 
d78aa53a1a5a: Pull complete 
345eab5774ef: Pull complete 
b8accc3f9ccc: Pull complete 
bed157eeb6d4: Pull complete 
Digest: sha256:60a219a5de7893d7f868e33bd59fade0ee1eb0c1d4cc4a78e88db1b810768339
Status: Downloaded newer image for winglian/axolotl:main-latest
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
# TODO

## Benchmark Script
- Load Models
- Set # of Runs
- Export versions, info via collect script
- Run variants
	- label: Name
	- env: HIPBLAST=0
	- opsiont:
- Actual Variants
	- HIPBLAST
	- FA2
	- Q FP8
	- kvcache FP8
	- OMP
	- i/o
		- 128/128
		- 512/512
		- 1024/128
		- 1024/1024
		- 8192/1024
		- 8192/8192
	- pp vs tp

Reproducible Script, Multirun Average
Version
Output logs
grep and pull

Run through variations automatically
quant kvcache


inference
- existing VLLM numbers, match settings to get baseline?


runpod
https://blog.runpod.io/amd-mi300x-vs-nvidia-h100-sxm-performance-comparison-on-mixtral-8x7b-inference/



GEMM tuning

BentoML

Big Models
WizardLM 8x22b
nemotron 340b
DeepSeek 2.5


405B
- quants
* batchsize
* hipblaslt
* FP8
* Kvcache
* Quants

Docker:
[ ] torchtune standard
llama2 qlora - 1 gpu
llama2 qlora - 8 gpu
llama3 8b - 1 gpu
llama3 8b - 8 gpu
llama3 70b

torchtune
wandb
shisa replication
llama 8b qlora

[ ] axolotl
shisa-v2 ablation test

shaberi test
testing

voicechat


We should be able to validate and compare vs:
## 2024-06-05 BentoML # vLLM, LMDeploy, MLC-LLM, TensorRT-LLM, and TGI Llama 3 8B, 70Bq4
https://bentoml.com/blog/benchmarking-llm-inference-backends

## 2024-06-12 Tensorwave vLLM benchmarks Mixtral 8x7B
https://tensorwave.com/blog/amds-mi300x-outperforms-nvidias-h100-for-llm-inference
https://www.linkedin.com/pulse/amds-mi300x-outperforms-nvidias-h100-llm-inference-tensorwave-ymuhc

TP1, TP2
128:128
BS 1, 2, 4, 8 - 1024
Mixtral 8x7B

## 2024-06-28 Nscale vLLM benchmarks Mixtral 8x7B

nscale
https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x
https://www.reddit.com/r/AMD_Stock/comments/1dgirzl/benchmarking_brilliance_single_amd_mi300x_vllm/
Mixtral 8x7B

## 2024-08-24 AMD vLLM MLPerf
https://community.amd.com/t5/instinct-accelerators/engineering-insights-unveiling-mlperf-results-on-amd-instinct/ba-p/705623
In the offline scenario, we used a max_num_seqs parameter of 2048 to maximize throughput, while 768 was set for the server scenario to meet latency targets—both significantly higher than the default 256 value used in vLLM.
The vLLM’s support for paged attention enables efficient KV cache management, avoiding memory fragmentation issues because of large memory AMD Instinct MI300X accelerators.
AMD Instinct MI300X accelerator hardware supports the FP8 numerical format, and we extended this capability across the entire inference software stack. Using Quark, we quantized LLaMA2-70B model weights to FP8, retaining 99.9% accuracy as required by MLPerf. We also added FP8 support to vLLM, upgraded the hipBLASLt library, and implemented FP8 KV cache, significantly boosting performance.
## 2024-10-09 dstack vLLM benchmarks Llama 3 405B
https://dstack.ai/blog/amd-mi300x-inference-benchmark/#tokensec-per-batch-size
https://github.com/dstackai/benchmarks/tree/main/amd/inference

TGI 2X vLLM (especially after 0.6? Doesn't seem right...)

My initial validation run seems vLLM and TGI are actually pretty close?
TPS in same ballpark for bs=64 and bs=128

Neat, glad to see the repo since I'm doing independent testing on the same system. So, I've been focused on vLLM exclusively for the inference (actually been trying to get replicable training numbers first).  Anyway, interestingly, I've gotten some slightly different results from my testing running `vllm 0.6.3.dev114+g4f95ffee` - a day or two old version from source:

```
# run server
TORCH_BLAS_PREFER_HIPBLASLT=0 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve meta-llama/Llama-3.1-405B-Instruct  --tensor-parallel-size=8 --disable-log-requests

# bs=64
python benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-405B-Instruct  --dataset-name sonnet  --num-prompt=64 --dataset-path="sonnet.txt"
WARNING 10-09 20:38:39 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sonnet', dataset_path='sonnet.txt', model='meta-llama/Llama-3.1-405B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=64, logprobs=None, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf

============ Serving Benchmark Result ============
Successful requests:                     64        
Benchmark duration (s):                  35.65     
Total input tokens:                      32541     
Total generated tokens:                  9600      
Request throughput (req/s):              1.80      
Output token throughput (tok/s):         269.32    
Total Token throughput (tok/s):          1182.23   
---------------Time to First Token----------------
Mean TTFT (ms):                          11498.39  
Median TTFT (ms):                        11266.60  
P99 TTFT (ms):                           22434.31  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          144.45    
Median TPOT (ms):                        146.29    
P99 TPOT (ms):                           196.72    
---------------Inter-token Latency----------------
Mean ITL (ms):                           144.44    
Median ITL (ms):                         90.40     
P99 ITL (ms):                            345.39    
==================================================

# bs=128
$ python benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-405B-Instruct  --dataset-name sonnet  --num-prompt=128 --dataset-path="sonnet.txt"
WARNING 10-09 20:51:59 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sonnet', dataset_path='sonnet.txt', model='meta-llama/Llama-3.1-405B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=128, logprobs=None, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf

============ Serving Benchmark Result ============
Successful requests:                     128       
Benchmark duration (s):                  62.97     
Total input tokens:                      65027     
Total generated tokens:                  19200     
Request throughput (req/s):              2.03      
Output token throughput (tok/s):         304.91    
Total Token throughput (tok/s):          1337.58   
---------------Time to First Token----------------
Mean TTFT (ms):                          23621.80  
Median TTFT (ms):                        22912.31  
P99 TTFT (ms):                           48069.33  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          219.19    
Median TPOT (ms):                        225.35    
P99 TPOT (ms):                           320.04    
---------------Inter-token Latency----------------
Mean ITL (ms):                           219.18    
Median ITL (ms):                         316.10    
P99 ITL (ms):                            348.60    
==================================================
```

At both batch sizes, throughput looks a lot closer to what you'd expect (about on part w/ TGI).

Happy to discuss on testing if you want to connect. I'm still trying to get hipblaslt working w/ the latest PyTorch nightlies.

Accelerate


https://github.com/vllm-project/vllm/discussions/9251#discussioncomment-10906873

```
(vllm) hotaisle@ENC1-CLS01-SVR09:~/vllm$ TORCH_BLAS_PREFER_HIPBLASLT=0 python benchmarks/benchmark_throughput.py --backend vllm --input-len 512 --output-len 128 --model meta-llama/Llama-2-7b-chat-hf -tp 4 --quantization fp8

-tp4 Q FP8
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:32<00:00, 30.53it/s, est. speed input: 15631.80 toks/s, output: 3907.95 toks/s]
Throughput: 30.22 requests/s, 19338.72 tokens/s

-tp4
Processed prompts: 100%|█████████████████████████████████████████| 1000/1000 [00:24<00:00, 41.12it/s, est. speed input: 21054.43 toks/s, output: 5263.61 toks/s]
Throughput: 40.58 requests/s, 25971.88 tokens/s
```


# Publishing
- Move testing and docs into repo
- Quarto - can all scripting happen in Jupyter for reproducibility?
- Script to create recently updated from logs
- CSS for FAIL vs WORKs