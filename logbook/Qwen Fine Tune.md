I've done some raw Qwen tuning before with the included scripts:
* https://github.com/AUGMXNT/shisa/tree/main/train/qwen
* https://github.com/QwenLM/Qwen/#finetuning
## Compatibility

### Transformers
Qwen wants you to use `transformers==4.32.0` primarily because `4.35.0` changes how gradient checkpointing works. Still we should use the latest to get the fixes for NEFTune etc.
```
pip install -U git+https://github.com/huggingface/transformers.git
```

To fix this we have to change `modeling_qwen.py` in the model folder:
```
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

Errors:
```
/home/local/.cache/huggingface/modules/transformers_modules/rinna_nekomata-14b/modeling_qwen.py:527: UserWarning: Use of masked_fill_ on expanded tens
ors is deprecated. Please clone() the tensor before performing this operation. This also applies to advanced indexing e.g. tensor[mask] = scalar (Trig
gered internally at ../aten/src/ATen/native/cuda/Indexing.cu:1564.)                                                                                   
  attention_mask.masked_fill_(~causal_mask, torch.finfo(query.dtype).min)                                                                             
/home/local/.cache/huggingface/modules/transformers_modules/rinna_nekomata-14b/modeling_qwen.py:527: UserWarning: Use of masked_fill_ on expanded tens
ors is deprecated. Please clone() the tensor before performing this operation. This also applies to advanced indexing e.g. tensor[mask] = scalar (Trig
gered internally at ../aten/src/ATen/native/cuda/Indexing.cu:1564.)                                                                                   
  attention_mask.masked_fill_(~causal_mask, torch.finfo(query.dtype).min)

...


RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [2, 1, 1, 363]] is at version 41; expected version 39 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```


## jondurbin/qlora

Set in model `config.json`:
```
"use_flash_attn": false,
```
* https://github.com/hiyouga/LLaMA-Factory/issues/601



## LLaMA-Factory

This time, we'll try a QLoRA w/ https://github.com/hiyouga/LLaMA-Factory that has just integrated https://github.com/unslothai/unsloth support for improved performance.

We sill be doing a tune on the new https://huggingface.co/rinna/nekomata-14b continued pre-train (+66B JA/EN tokens).

```
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

To take advantage of unsloth, first we need to llamafy Qwen models with https://github.com/hiyouga/LLaMA-Factory/blob/main/tests/llamafy_qwen.py:
```
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
```
"ultra-orca-boros-en-ja-v1": {
    "hf_hub_url": "augmxnt/ultra-orca-boros-en-ja-v1",
    "formatting": "sharegpt"
  },
```

OK, now we should be ready to tune. To get the web interface:
```
# only single device supported by llama-factory and unsloth
CUDA_VISIBLE_DEVICES=0 python src/train_web.py
```

It'll generate a script we'll mostly use:
```
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

Let's try without:
```
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


## Axolotl

### Setup
```
# Base
https://github.com/OpenAccess-AI-Collective/axolotl
mamba create -n axolotl python=3.11
mamba activate axolotl

# CUDA
mamba install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
mamba install -c "nvidia/label/cuda-12.3.1" cuda-toolkit
mamba env config vars set CUDA_PATH="$CONDA_PREFIX"
mamba env config vars set CUDA_HOME="$CONDA_PREFIX"
mamba activate axolotl

# Axolotl
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'

# Qwen
pip install einops transformers_stream_generator

# this will take 10min+ to build...
# git clone https://github.com/Dao-AILab/flash-attention
# cd flash-attention
# pip install csrc/layer_norm

pip install flash_attn -U --force-reinstall
```

### Flash Attention
Grr:
```
ImportError: /home/local/.conda/envs/axolotl/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi
```
* https://github.com/Dao-AILab/flash-attention/issues/620

Let's make sure we have the right nvcc...
```
nvcc --version
which nvcc
```

We need to do this
```
pip install flash_attn -U --force-reinstall
```
* https://github.com/Dao-AILab/flash-attention/issues/620

### DeepSpeed
Well w/ FlashAttention happy, we need to update CUDA for DeepSpeed... 
```
# deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.7 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without

mamba install -c "nvidia/label/cuda-12.1.1" cuda-toolkit
pip install deepspeed -U --force-reinstall

# Give up
export DS_SKIP_CUDA_CHECK=1
```

## TPU
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