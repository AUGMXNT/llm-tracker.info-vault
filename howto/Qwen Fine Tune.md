I've done some raw Qwen tuning before with the included scripts:
* https://github.com/AUGMXNT/shisa/tree/main/train/qwen
* https://github.com/QwenLM/Qwen/#finetuning

Qwen is pretty tricky to work with.  Some things to watch out for:
## Compatibility

### Transformers
Qwen wants you to use `transformers==4.32.0` primarily because `4.35.0` changes how gradient checkpointing works. Still we should use the latest to get the fixes for NEFTune etc.
```shell
pip install -U git+https://github.com/huggingface/transformers.git
```

To fix this we have to change `modeling_qwen.py` in the model folder:
```python
def _set_gradient_checkpointing(self, enable: bool = False, gradient_checkpointing_func: Callable = None):
        is_gradient_checkpointing_set = False

        if isinstance(self, QWenModel):
            self.gradient_checkpointing = enable
            self._gradient_checkpointing_func = gradient_checkpointing_func
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if isinstance(module, QWenModel):
                module.gradient_checkpointing = enable
                module._gradient_checkpointing_func = gradient_checkpointing_func
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute 'gradient_checkpointing' to modules of the model that uses checkpointing.")

```
* https://github.com/QwenLM/Qwen/issues/661#issuecomment-1835520079

### QLoRA
I ran into this problem trying to QLoRA. This also requires a change to `modeling_qwen.py` otherwise you will get an error in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) like:

```shell
RuntimeError: value cannot be converted to type at::Half without overflow
```
or with Jon Durbin's [QLora fork](https://github.com/jondurbin/qlora) like:
```shell
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [2, 1, 1, 363]] is at version 41; expected version 39 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```

Here's the fix:
```python
# Change `attention_mask.masked_fill_(~causal_mask, torch.finfo(query.dtype).min)`

attention_mask.masked_fill(~causal_mask, -65504.0)
```
* In Chinese: https://github.com/hiyouga/LLaMA-Factory/issues/1475
* Translated: https://github-com.translate.goog/hiyouga/LLaMA-Factory/issues/1475?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp

When I was using `use_flash_attn` I was also getting this error:
```python
assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
```

For 4/8-bit usage, I think you can't use Flash Attention...

### Tuning Tips
- [Analysis of LoRA parameters](https://medium.com/@drishtisharma96505/comparative-analysis-of-lora-parameters-on-llama-2-with-flash-attention-574b913295d4) - bigger `lora_alpha` better, rest doesn't really matter
## jondurbin/qlora

Set in model `config.json`:
```json
"use_flash_attn": false,
```
* https://github.com/hiyouga/LLaMA-Factory/issues/601

Our dataset, [ultra-orca-boros-en-ja-v1](https://huggingface.co/datasets/augmxnt/ultra-orca-boros-en-ja-v1) is a sharegpt-formatted parquet file (but with system prompts) and this fork is built to handle it.

Once we make the modeling fixes, the current code should work OOTB. You may have to lower max_tokens, max_length, and the gradient and batch sizes to get it to fit in 24GB of RAM (also add a memory limit), even if you are using DeepSpeed-ZeRO3 (using Axolotl's [zero3_bf16.json](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed/zero3_bf16.json) seems to work).

However, I noticed that when training, loss almost immediately goes to 0, so... so you might need to check on that...
## LLaMA-Factory

This time, we'll try a QLoRA w/ https://github.com/hiyouga/LLaMA-Factory that has just integrated https://github.com/unslothai/unsloth support for improved performance.

We will be doing a tune on the new https://huggingface.co/rinna/nekomata-14b continued pre-train (+66B JA/EN tokens).

```shell
# Base
git clone https://github.com/hiyouga/LLaMA-Factory.git
mamba create -n llama-factory python=3.11
mamba activate llama-factory
cd LLaMA-Factory
pip install -r requirements.txt
pip install bitsandbytes
pip install wandb

# Qwen
pip install einops transformers_stream_generator
pip install -U flash_attn
# this will take 10min+ to build...
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install csrc/layer_norm

# Unsloth
pip install xformers
pip install "unsloth[kaggle] @ git+https://github.com/unslothai/unsloth.git"
```

Basic config that worked:
```shell
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /models/llm/hf/rinna_nekomata-14b \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template llama2 \
    --flash_attn False \
    --shift_attn False \
    --use_unsloth True \
    --dataset_dir data \
    --dataset sharegpt-clean-ja \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200000 \
    --warmup_steps 0 \
    --neftune_noise_alpha 5 \
    --upcast_layernorm True \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target c_attn \
    --output_dir saves/Qwen-14B/lora/train_2023-12-23-19-04-13 \
    --bf16 True \
    --report_to wandb True
```
### More QLoRA
Settings + DeepSpeed 3 from [XVERSE-65B repo](https://github.com/xverse-ai/XVERSE-65B#%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83):

```shell
deepspeed --num_gpus 8 src/train_bash.py \
    --deepspeed deepspeed.json \
    --stage sft \
    --model_name_or_path /  \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir  output_model_path \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16
```

`deepspeed.json`
```json
{
    "train_micro_batch_size_per_gpu":"auto",
    "gradient_accumulation_steps":"auto",
    "gradient_clipping":"auto",
    "zero_allow_untested_optimizer":true,
    "fp16":{
        "enabled":false
    },
    "bfloat16":{
        "enabled":true
    },
    "zero_optimization":{
        "stage":3,
        "allgather_partitions":true,
        "reduce_scatter":true,
        "overlap_comm":false,
        "contiguous_gradients":true
    }
}
```

### ShareGPT Formt
It doesn't work with our dataset:
```shell
^^^^^^^^^^^^^
  File "/home/local/shisa/train/nekomata/LLaMA-Factory/src/llmtuner/data/loader.py", line 122, in convert_format
    raise ValueError("Only accepts conversation in u/a/u/a/u/a order.")
ValueError: Only accepts conversation in u/a/u/a/u/a order.
```

But you can use the (smaller, so better for testing anyway) [chatntq sharegpt dataset](https://huggingface.co/datasets/NTQAI/sharegpt-clean-ja) as a sharegpt formatted example.

### unsloth
**STATUS:** Uh, I couldn't get this working...

To take advantage of unsloth, first we need to llamafy Qwen models with https://github.com/hiyouga/LLaMA-Factory/blob/main/tests/llamafy_qwen.py:
```shell
# Convert to safetensors first:
wget https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/convert-to-safetensors.py
time CUDA_VISIBLE_DEVICES=0 python convert-to-safetensors.py /models/llm/hf/rinna_nekomata-14b --output /models/llm/hf/rinna_nekomata-14b --max-shard-size=10GB --bf16

# Now we llamafy
pip install fire
time python tests/llamafy_qwen.py --input_dir /models/llm/hf/rinna_nekomata-14b --output_dir /models/llm/hf/rinna_nekomata-14b-llamafied --shard_size 10GB

# Only has the bin files (probably should modify to safetensors) so copy rest
cd /models/llm/hf/rinna_nekomata-14b-llamafied
cp /models/llm/hf/rinna_nekomata-14b/*.cu ./
cp /models/llm/hf/rinna_nekomata-14b/*.py ./
cp /models/llm/hf/rinna_nekomata-14b/qwen* ./
cp /models/llm/hf/rinna_nekomata-14b/token* ./
```

In `data/dataset_info.json` add as the first item:
```json
"ultra-orca-boros-en-ja-v1": {
    "hf_hub_url": "augmxnt/ultra-orca-boros-en-ja-v1",
    "formatting": "sharegpt"
  },
```

OK, now we should be ready to tune. To get the web interface:
```shell
# only single device supported by llama-factory and unsloth
CUDA_VISIBLE_DEVICES=0 python src/train_web.py
```

It'll generate a script we'll mostly use:
```shell
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /models/llm/hf/rinna_nekomata-14b-llamafied \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template llama2 \
    --flash_attn True \
    --shift_attn False \
    --use_unsloth True \
    --dataset_dir data \
    --dataset ultra-orca-boros-en-ja-v1 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neftune_noise_alpha 5 \
    --upcast_layernorm True \
    --lora_rank 8 \
    --lora_dropout 0.0 \
    --lora_target c_attn \
    --output_dir saves/Qwen-14B/lora/train_2023-12-23-19-04-13 \
    --bf16 True \
    --report_to wandb True
```
* unsloth fast patching only works with `--lora_dropout 0`
* for llamafied qwen, you may need to edit your `~/.conda/envs/llama-factory/lib/python3.11/site-packages/unsloth/models/llama.py` and add `trust_remote_code=True,` to the `tokenizer` loading.
* unsloth gets an `assert` error looking at the llamafied modules :(

## Axolotl
To get Axolotl with Qwen working we need to be *very* careful and specific about our libraries:

### Manual Setup
Default environment setup:
```shell
# in case you need to start over... (which I did, many, many, times)
# mamba env remove --name axolotl

# Base
mamba create -n axolotl python=3.11
mamba activate axolotl
```

We will install the latest CUDA 11.8.0. See the available versions here: https://anaconda.org/nvidia/cuda-toolkit/labels
```shell
# CUDA
mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
mamba env config vars set CUDA_PATH="$CONDA_PREFIX"
mamba env config vars set CUDA_HOME="$CONDA_PREFIX"

# required for nvcc 11:
mamba install gxx=11.4.0
# zomg this was messing w/ my cheerios - somehow ccbin set and was screwing up compiles
mamba env config vars set NVCC_PREPEND_FLAGS=""
mamba activate axolotl
```

Let's install the important libs ourselves. We need to do this our we will end up in CUDA hell (eg some things need 11, some need 12)

PyTorch
```shell
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Make sure we're on the right CUDA version
python3 -c "import torch; print(torch.__version__)"

# If you need to blast it - this is 2GB+ so the biggest thing to get right
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Transformers (see above note on Qwen models needing modifications to `qwen_modeling.py` w/ >=4.35.0)
```shell
pip install transformers
```

Flash Attention (barfs if you install regularly, wants CUDA 11)
```shell
pip install packaging
mamba install ninja
pip install flash-attn --no-build-isolation --no-cache-dir
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

DeepSpeed (tries to install CUDA 12)
```shell
pip install deepspeed --no-deps --no-cache-dir
pip install hjson pydantic pynvml py-cpuinfo
# pay attention to the PyTorch CUDA verson - if it changed to 12.1 you f'd up
ds_report
```

Qwen Libraries
```shell
pip install einops transformers_stream_generator
# You don't strictly need this, but it's supposed to be faster
# and it's a good way to make sure your gcc setup is OK
# otherwise it will bite you later
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install csrc/layer_norm
```

Now we should be ready for `axolotl`. We are *not* going to install dependencies, which messes with our above libs and will handle it all manuallly.

```shell
git clone https://github.com/OpenAccess-AI-Collective/axolotl

# Axolotl
cd axolotl
pip install -e '.' --no-deps

# Install rest of reqs
pip install accelerate addict art auto-gptq bert-score bitsandbytes colorama datasets evaluate fire fschat gcsfs gradio hf_transfer numba optimum peft rouge-score s3fs scikit-learn scipy sentencepiece tensorboard wandb xformers

# You may need to reinstall gradio (pydantic version issue)
pip install gradio
```
### Potential Gotchas
You may still need to rebuild Flash Attention:
```python
ImportEror: ... flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi
```
* https://github.com/Dao-AILab/flash-attention/issues/620

You *shouldn't* get this DeepSpeed problem (check `ds_report`) if you installed everything manually:
```shell
deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.7 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without
```
* https://www.deepspeed.ai/tutorials/advanced-install/
* You might be able to `export DS_SKIP_CUDA_CHECK=1` but to force things, but who knows.
## TPU
TODO!

12/25-1/25 TPU Research

Manage TPUS
https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm

TPUs in Collab
https://colab.research.google.com/notebooks/tpu.ipynb#scrollTo=clSFHJkFNylD

Example Collab
https://colab.research.google.com/notebooks/tpu.ipynb?authuser=2#scrollTo=FpvUOuC3j27n

Predict Shakespeare w/ Keras
https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb

Accelerate
https://huggingface.co/docs/accelerate/concept_guides/training_tpu
https://github.com/huggingface/accelerate/issues/471
https://github.com/huggingface/accelerate
https://github.com/huggingface/accelerate/releases
https://github.com/christianversloot/machine-learning-articles/blob/main/quick-and-easy-gpu-tpu-acceleration-for-pytorch-with-huggingface-accelerate.md
https://github.com/huggingface/accelerate/issues/29

* Using PyTorch XLA
	* https://github.com/ssbuild/qwen_finetuning
	* https://github.com/hiyouga/LLaMA-Factory
	* https://github.com/young-geng/EasyLM/issues/61
	* https://nlpcloud.com/how-to-fine-tune-llama-openllama-xgen-with-jax-on-tpu-gpu.html?utm_source=reddit&utm_campaign=i859w625-3816-11ed-a261-0242ac140016
	* https://pypi.org/project/torch-xla/
	* https://pytorch.org/xla/master/

xtuner
https://github.com/InternLM/xtuner
https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing