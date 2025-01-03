omakase.in2024-09: The compatibility list has been rolled to the main AMD GPU page: https://llm-tracker.info/howto/AMD-GPUs


In January 2024 I applied for the [Hackster.io AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) and a W7900 card was delivered on 2024-04-19.
- Application: https://www.hackster.io/contests/amd2023/hardware_applications/16885
- Project: https://www.hackster.io/lhl/ultra-low-latency-local-voice-assistant-avatar-4c48f2
- Repo:
# 2024-05 Library Status
All tests on an Ubuntu 22.04 LTS HWE box w/ ROCm native install:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html

For more general AMD info, see: https://llm-tracker.info/howto/AMD-GPUs
## PyTorch (works)
https://pytorch.org/get-started/locally/
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```
- ROCm 6.0 now available in Stable (6.1 Nightly)

## Triton (mostly works)
```
# The nightly works for me:
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Takes a while but you could build as well
git clone https://github.com/openai/triton.git
cd triton
pip install ninja cmake wheel; # build-time dependencies
pip install -e python

# ROCm fork (old, probably don't use)
git clone https://github.com/ROCm/triton
cd triton/python
pip install -e .
```
You can run the some of the basic examples in `tutorials` which seems to work...
```
pip install matplotlib pandas pytest tabulate
```

Running the the python/tutorial scripts, `05-layer-norm.py` and `08-grouped-gemm.py` failed.
```
# this can be fixed by changing atol=2e-2 so some rounding error somewhere
# Intel has the same issue: https://github.com/intel/intel-xpu-backend-for-triton/issues/339
HIP_VISIBLE_DEVICES=0 python 05-layer-norm.py 
Traceback (most recent call last):
  File "/home/lhl/triton/triton/python/tutorials/05-layer-norm.py", line 376, in <module>
    test_layer_norm(1151, 8192, torch.float16)
  File "/home/lhl/triton/triton/python/tutorials/05-layer-norm.py", line 318, in test_layer_norm
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

# this seems to fail on the third set of tensors, if you print them most of them match, not sure if there's a real math error or not...
HIP_VISIBLE_DEVICES=0 python 08-grouped-gemm.py 
Traceback (most recent call last):
  File "/home/lhl/triton/triton/python/tutorials/08-grouped-gemm.py", line 208, in <module>
    assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

See also:
* https://github.com/ROCm/aotriton/issues/16
* https://github.com/triton-lang/triton/issues/3704

## Flash Attention (sort of works)
Last update 2024-04-08 - FA 2.5.5 being worked on internally
- https://github.com/ROCm/flash-attention/issues/35#issuecomment-2042391285
- Currently  gfx1100 not supported

Thre is a branch of 2.0.4  that works for forward pass only:
```
git clone https://github.com/ROCm/flash-attention
cd flash-attention
git fetch
git branch -a
git checkout howiejay/navi_support
python setup.py install
python -c "import flash_attn; print(flash_attn.__version__)"
```
- howiejay is no longer working on this project: https://github.com/Dao-AILab/flash-attention/issues/707#issuecomment-2049042957
## xformers (not working)
Neither the upstream or AMD's ROCm fork compile:
See: https://github.com/facebookresearch/xformers/issues/1026

We need to compile from code
```
# Make sure we have the ROCm version of PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

# You can double check
python -c "import torch; print(torch.version.hip)"

# Install from source - on a Ryzen 5600G takes ~
pip install ninja
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

pip wheel -v --no-build-isolation git+https://github.com/ROCm/xformers.git@main#egg=xformers


# Double check
python -m xformers.info
```
## bitsandbytes (works)
As of late May, the multi-backend-refactor branch works:
```
# Docs: https://github.com/TimDettmers/bitsandbytes/blob/multi-backend-refactor/docs/source/rocm_installation.mdx

git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
git checkout multi-backend-refactor
pip install -r requirements-dev.txt
#Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
cmake -DCOMPUTE_BACKEND=hip -S . 
make
pip install .
```
* Installation notes/discussion: https://github.com/TimDettmers/bitsandbytes/discussions/1219

ROCM fork works (0.44.0.dev0)
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -S .
make
pip install .
```
* Only 1% difference w/ FA2 2.0.4-rocm-howiejay
* About 3min to load 70B model (132GiB), 40GiB memory,  3.3 tok/s bs=1 inference speed
## vllm (works w/ no FA)

I was able to build the latest main:HEAD as of 2024-05-01 (failed a couple weeks prior)
```
# To build vllm on ROCm 6.0 for Radeon RX7900 series (gfx1100), you should specify BUILD_FA as below:
docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
```
- This took about 45 minutes to build (Ryzen 5600G CPU)
- see additional color in the bug I filed: https://github.com/vllm-project/vllm/issues/4514


In order to run vllm, you will need to use the `VLLM_USE_TRITON_FLASH_ATTN=0` environment variable

You can run with something like:
```
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /models/hf/NousResearch_Meta-Llama-3-8B:/app/model \
   vllm-rocm \
   bash
```

To connect to the instance you can `docker ps` and
```
docker exec -it <container-id> bash
```

Benchmark
```
root@rocm:/app/vllm/benchmarks# VLLM_USE_TRITON_FLASH_ATTN=0 python benchmark_throughput.py --model /app/model --input-len 3968 --output-len 128
Namespace(backend='vllm', dataset=None, input_len=3968, output_len=128, model='/app/model', tokenizer='/app/model', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='cuda', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 05:25:25 llm_engine.py:99] Initializing an LLM engine (v0.4.1) with config: model='/app/model', speculative_config=None, tokenizer='/app/model', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 05:25:25 utils.py:620] Found nccl from library /opt/rocm-6.0.0/lib/librccl.so.1
/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
INFO 05-01 05:25:26 selector.py:59] flash_atten is not supported on NAVI GPUs.
INFO 05-01 05:25:26 selector.py:38] Using ROCmFlashAttention backend.
INFO 05-01 05:25:38 model_runner.py:172] Loading model weights took 14.9595 GB
INFO 05-01 05:25:41 gpu_executor.py:114] # GPU blocks: 12003, # CPU blocks: 2048
INFO 05-01 05:25:41 model_runner.py:872] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-01 05:25:41 model_runner.py:876] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-01 05:25:46 model_runner.py:953] Graph capturing finished in 5 secs.
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Processed prompts:  28%|██████████████████████████████████████████████▉                                                                                    Processed prompts:  33%|███████████████████████████████▌                                                                | 329/1000 [18:28<33:42,  3.01s/it]Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [49:23<00:00,  2.96s/it]
Throughput: 0.34 requests/s, 1380.87 tokens/s

# 512/512
root@rocm:/app/vllm/benchmarks# VLLM_USE_TRITON_FLASH_ATTN=0 python benchmark_throughput.py --model /app/model --input-len 512 --output-len 512
Namespace(backend='vllm', dataset=None, input_len=512, output_len=512, model='/app/model', tokenizer='/app/model', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='cuda', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 07:10:44 llm_engine.py:99] Initializing an LLM engine (v0.4.1) with config: model='/app/model', speculative_config=None, tokenizer='/app/model', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 07:10:44 utils.py:620] Found nccl from library /opt/rocm-6.0.0/lib/librccl.so.1
/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
INFO 05-01 07:10:46 selector.py:59] flash_atten is not supported on NAVI GPUs.
INFO 05-01 07:10:46 selector.py:38] Using ROCmFlashAttention backend.
INFO 05-01 07:10:57 model_runner.py:172] Loading model weights took 14.9595 GB
INFO 05-01 07:11:00 gpu_executor.py:114] # GPU blocks: 12003, # CPU blocks: 2048
INFO 05-01 07:11:01 model_runner.py:872] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-01 07:11:01 model_runner.py:876] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-01 07:11:06 model_runner.py:953] Graph capturing finished in 5 secs.
Processed prompts: 100%|██████████████████████████| 1000/1000 [12:26<00:00,  1.34it/s]
Throughput: 1.34 requests/s, 1370.69 tokens/s
```

As a reference, here's a 3090
```
# 3968/128
❯ CUDA_VISIBLE_DEVICES=0 python benchmark_throughput.py --model /models/hf/NousResearch_Meta-Llama-3-8B --input-len 3968 --output-len 128
Namespace(backend='vllm', dataset=None, input_len=3968, output_len=128, model='/models/hf/NousResearch_Meta-Llama-3-8B', tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='cuda', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 15:31:51 llm_engine.py:98] Initializing an LLM engine (v0.4.1) with config: model='/models/hf/NousResearch_Meta-Llama-3-8B', speculative_config=None, tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 15:31:51 utils.py:608] Found nccl from library /home/lhl/.config/vllm/nccl/cu12/libnccl.so.2.18.1
INFO 05-01 15:31:51 selector.py:28] Using FlashAttention backend.
INFO 05-01 15:32:02 model_runner.py:173] Loading model weights took 14.9595 GB
INFO 05-01 15:32:04 gpu_executor.py:119] # GPU blocks: 2354, # CPU blocks: 2048
INFO 05-01 15:32:05 model_runner.py:976] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-01 15:32:05 model_runner.py:980] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-01 15:32:08 model_runner.py:1057] Graph capturing finished in 3 secs.
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [26:25<00:00,  1.59s/it]
Throughput: 0.63 requests/s, 2580.85 tokens/s

# 512/512
❯ CUDA_VISIBLE_DEVICES=0 python benchmark_throughput.py --model /models/hf/NousResearch_Meta-Llama-3-8B -
-input-len 512 --output-len 512             
Namespace(backend='vllm', dataset=None, input_len=512, output_len=512, model='/models/hf/NousResearch_Met
a-Llama-3-8B', tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', quantization=None, tensor_parallel_si
ze=1, n=1, use_beam_search=False, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=Fal
se, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='cuda', enable_prefix_caching=False, enable_chunked_prefill=Fal
se, max_num_batched_tokens=None, download_dir=None)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned
 or trained.
INFO 05-01 16:09:19 llm_engine.py:98] Initializing an LLM engine (v0.4.1) with config: model='/models/hf/
NousResearch_Meta-Llama-3-8B', speculative_config=None, tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_
code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)                                                                             Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.                                        
INFO 05-01 16:09:19 utils.py:608] Found nccl from library /home/lhl/.config/vllm/nccl/cu12/libnccl.so.2.18.1         
INFO 05-01 16:09:20 selector.py:28] Using FlashAttention backend.                                        INFO 05-01 16:09:29 model_runner.py:173] Loading model weights took 14.9595 GB                           INFO 05-01 16:09:31 gpu_executor.py:119] # GPU blocks: 2354, # CPU blocks: 2048                          INFO 05-01 16:09:32 model_runner.py:976] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.                                                                           INFO 05-01 16:09:32 model_runner.py:980] CUDA graphs can take additional 1~3 GiB memory per GPU. If you a
re running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-01 16:09:35 model_runner.py:1057] Graph capturing finished in 3 secs.                            Processed prompts: 100%|█████████████████████████████████████████████| 1000/1000 [09:41<00:00,  1.72it/s]
Throughput: 1.72 requests/s, 1759.44 tokens/s 

# 512 - no FA2 (XFormers)
❯ CUDA_VISIBLE_DEVICES=0 python benchmark_throughput.py --model /models/hf/NousResearch_Meta-Llama-3-8B --input-len 512 --output-len 512
Namespace(backend='vllm', dataset=None, input_len=512, output_len=512, model='/models/hf/NousResearch_Meta-Llama-3-8B', tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=1000, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='cuda', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 16:20:12 llm_engine.py:98] Initializing an LLM engine (v0.4.1) with config: model='/models/hf/NousResearch_Meta-Llama-3-8B', speculative_config=None, tokenizer='/models/hf/NousResearch_Meta-Llama-3-8B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-01 16:20:12 utils.py:608] Found nccl from library /home/lhl/.config/vllm/nccl/cu12/libnccl.so.2.18.1
INFO 05-01 16:20:12 selector.py:77] Cannot use FlashAttention backend because the flash_attn package is not found. Please install it for better performance.
INFO 05-01 16:20:12 selector.py:33] Using XFormers backend.
INFO 05-01 16:20:19 model_runner.py:173] Loading model weights took 14.9595 GB
INFO 05-01 16:20:21 gpu_executor.py:119] # GPU blocks: 2354, # CPU blocks: 2048
INFO 05-01 16:20:22 model_runner.py:976] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-01 16:20:22 model_runner.py:980] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-01 16:20:25 model_runner.py:1057] Graph capturing finished in 4 secs.
Processed prompts: 100%|█████████████████████████████████████████████| 1000/1000 [09:48<00:00,  1.70it/s]
Throughput: 1.70 requests/s, 1739.09 tokens/s
```


## ExLlamaV2 (works)

Inference Speed:
```
$ GPU_MAX_HW_QUEUES=1 python test_inference.py -m /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/ -s
 -- Model: /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/
 -- Options: []
 -- Loading model...
 -- Loaded model in 2802.5172 seconds
 -- Loading tokenizer...
 -- Measuring token speed...
...

 ** Position  3968 + 128 tokens:    7.0301 t/s

...

 ** Position  8064 + 128 tokens:    4.5124 t/s
```
* 39GiB VRAM usages at 4096 tokens
* Insanely long (46min lol) load times on machines w/ 16GiB RAM - 30s w/ 64GiB of RAM
* `GPU_MAX_HW_QUEUES=1` not required w/ fast loading?

w/ FA2 2.0.4, no difference in perf
```
 ** Position  3968 + 128 tokens:    6.9836 t/s
```

Prompt Processing Speed:
```
$ GPU_MAX_HW_QUEUES=1 python test_inference.py -m /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/ -ps
Successfully preprocessed all matching files.
 -- Model: /data/models/exl2/LoneStriker_Meta-Llama-3-70B-Instruct-4.0bpw-h6-exl2/
 -- Options: []
 -- Loading model...
 -- Loaded model in 3402.6222 seconds
 -- Loading tokenizer...
 -- Warmup...
 -- Measuring prompt speed...
 ** Length   128 tokens:    154.0550 t/s
 ** Length   256 tokens:    269.5589 t/s
 ** Length   384 tokens:    358.5119 t/s
 ** Length   512 tokens:    359.8361 t/s
 ** Length   640 tokens:    365.1964 t/s
 ** Length   768 tokens:    429.5664 t/s
 ** Length   896 tokens:    426.6023 t/s
 ** Length  1024 tokens:    430.6259 t/s
 ** Length  2048 tokens:    416.8521 t/s
 ** Length  3072 tokens:    394.7572 t/s
 ** Length  4096 tokens:    363.3365 t/s
 ** Length  8192 tokens:    283.3092 t/s
```
## llama.cpp (works)
```
git clone https://github.com/ggerganov/llama.cpp/
cd llama.cpp
make LLAMA_HIPBLAS=1

$ ./llama-bench -m Meta-Llama-3-70B-Q4_K_M.gguf -p 3968
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon PRO W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 | pp 3968    |    255.59 ± 0.94 |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | ROCm       |  99 | tg 128     |     11.34 ± 0.01 |

build: b8109bc0 (2701)
```
- llama3 template/stop tokens still in progress: https://github.com/ggerganov/llama.cpp/issues/6747
- https://github.com/ggerganov/llama.cpp/pull/6745
- https://github.com/ggerganov/llama.cpp/pull/6751

Just for a baseline comparison, the W7900 is about 15% slower in prefill and 20% slower in generation than a 7900 XTX (lower TDP, slower clocks and memory?)
```
$ ./llama-bench -m /data/models/gguf/llama-2-7b.Q4_0.gguf -p 3968
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon PRO W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | pp 3968    |   2193.89 ± 3.09 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 | tg 128     |     93.94 ± 0.18 |

build: 784e11de (2725)
```
- For comparison numbers, see: https://llm-tracker.info/howto/AMD-GPUs#llamacpp

## MLC (works)
Install:
```
# Env
mamba create -n mlc python=3.11

# Pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

# Install
# https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm57 mlc-ai-nightly-rocm57
python -c "import mlc_llm; print(mlc_llm)"

# Required otherwise errors
mamba install conda-forge::lld

mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```
- https://github.com/mlc-ai/relax/issues/316
- https://github.com/mlc-ai/mlc-llm/issues/2144
- see also: https://github.com/mlc-ai/mlc-llm/issues/2160

Convert model
See: https://llm.mlc.ai/docs/compilation/convert_weights.html
```
mkdir dist

# Convert - takes about 10min for 70B
mlc_llm convert_weight /models/hf/NousResearch_Meta-Llama-3-70B/ --quantization q4f16_1 -o dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC

mlc_llm gen_config /models/hf/NousResearch_Meta-Llama-3-70B/ --quantization q4f16_1 --conv-template llama-3 -o dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/

mlc_llm chat dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/


mlc_llm bench dist/NousResearch_Meta-Llama-3-70B-q4f16_1-MLC/ --generate-length 4096

[2024-04-21 13:19:32] INFO model_metadata.py:96: Total memory usage: 40345.77 MB (Parameters: 37849.77 MB. KVCache: 0.00 MB. Temporary buffer: 2496.00 MB)
[2024-04-21 13:19:32] INFO model_metadata.py:105: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`

Statistics:
----------- prefill -----------
throughput: 36.910 tok/s
total tokens: 7 tok
total time: 0.190 s
------------ decode ------------
throughput: 12.041 tok/s
total tokens: 4096 tok
total time: 340.169 s

# --prompt "000...."

Statistics:
----------- prefill -----------
throughput: 95.501 tok/s
total tokens: 3376 tok
total time: 35.351 s
------------ decode ------------
throughput: 10.686 tok/s
total tokens: 128 tok
total time: 11.979 s
```
- 42.8GiB memory usage
- llama.cpp has about the same inference speed, 2.5X prompt processing
- exllama has 50% slower inference speed, but 4X prompt processing

## Whisper (works)
You can use the [OpenAI Whisper](https://github.com/openai/whisper) package or Whisper directly via [Huggingface Transformers directly](https://huggingface.co/docs/transformers/en/model_doc/whisper)

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) also works and performs a bit better.

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and anything that depends on it like [WhisperX](https://github.com/m-bain/whisperX) don't work as they depend on CTranslate2 that [has no AMD support](https://github.com/OpenNMT/CTranslate2/issues/1072).

## StyleTTS2 (works)
(slowly)
```
python -c "import nltk; nltk.download('punkt')"
RTF = 0.306594
```

## Unsloth (not working)
https://github.com/unslothai/unsloth
At minimum Unsloth requires
* Triton
* xformers

## TRL (sort of works)
Huggingface's TRL / SFTTrainer work on a single GPU, but accelerate and DeepSpeed are currently not happy campers.
https://github.com/huggingface/trl