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

# Build for MI300
export PYTORCH_ROCM_ARCH="gfx942"
python setup.py develop
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


### Executors 
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
| rp (tok/s)     | 35163.95 | 34606.80 | 32902.67 | 26856.43 |
- default = float16
- ray hangs, but mp is slower so use just use the default (none)
- hipblaslt doesn't work even when symlinked to the proper one so... not tested



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

#### Llama3 405B
8 x MI300
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


|                | Default |
| -------------- | ------- |
| it/s           | 10.24   |
| input (tok/s)  | 5253.82 |
| output (tok/s) | 1310.89 |
| tp (req/s)     | 10.20   |
| rp (tok/s)     | 6525.23 |
- missing Tens

Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:37<00:00, 10.24it/s, est. speed input: 5253.82 toks/s, output: 1310.89 toks/s]

```
WARNING 10-07 15:49:20 registry.py:198] Model architecture MistralForCausalLM is partially supported by ROCm: Sliding window attention (SWA) is not yet supported in Triton flash attention. For ha
lf-precision SWA support, please use CK flash attention by setting `VLLM_USE_TRITON_FLASH_ATTN=0
```



Flashinfer
https://github.com/flashinfer-ai/flashinfer/pull/491


# Training



https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx

Axolotl

PEFT
https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html
https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning
https://rocm.blogs.amd.com/artificial-intelligence/starcoder-fine-tune/README.html