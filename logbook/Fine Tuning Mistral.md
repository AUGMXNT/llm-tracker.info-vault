We'll try to fine tune [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/
).

# autotune-advanced


# Airoboros
We use Jon Durbin's QLora fork for training Airoboros (has the Airoboros dataset/instruction formatting):
* [https://github.com/jondurbin/qlora](https://github.com/jondurbin/qlora)

We also start with the latest gist available for the training script and `deepspeed.json`:
* [https://gist.github.com/jondurbin/7aabf2d8e1ded7b29897a3dd95d81e01](https://gist.github.com/jondurbin/7aabf2d8e1ded7b29897a3dd95d81e01)

Starting from a clean env:
```
# The basics
mamba create -n airoboros
mamba install pip
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

# This is required but not in the requirements.txt
pip install packaging
# If you don't have this and need flash_attn you will be in trouble
pip install ninja
```

We will want to modify `requirements.txt`. I remove `flash-attn` as 1) it forces a compile w/ PyTorch Nightly, but 2) Mistral requires the latest version to work from my understanding.
```
# remove flash-attn then
pip install -r requirements.txt
```

Once we finish, we need our Mistral compatible libs:
```
pip install flash-attn
pip install git+https://github.com/huggingface/transformers
```

At this point we need to poke around with the `train.sh` script:
```
export BASE_DIR="."
export WANDB_PROJECT=mistral-instruct-7b-airoboros-2.2.1

torchrun --nnodes=1 --nproc_per_node=2 qlora.py \
  --model_name_or_path "/models/llm/hf/mistralai_Mistral-7B-Instruct-v0.1" \
  --output_dir $BASE_DIR/$WANDB_PROJECT \
  --num_train_epochs 5 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 25 \
  --save_total_limit 1 \
  --data_seed 11422 \
  --evaluation_strategy no \
  --eval_dataset_size 0.001 \
  --max_new_tokens 4096 \
  --dataloader_num_workers 3 \
  --logging_strategy steps \
  --remove_unused_columns False \
  --do_train \
  --double_quant \
  --quant_type nf4 \
  --bits 4 \
  --bf16 \
  --dataset $BASE_DIR/instructions.jsonl \
  --dataset_format airoboros \
  --model_max_len 4096 \
  --per_device_train_batch_size 1 \
  --learning_rate 0.000022 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.005 \
  --weight_decay 0.0 \
  --seed 11422 \
  --report_to wandb \
  --deepspeed deepspeed-7b.json \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --max_memory_MB 24000
```
* `nproc_per_node` = # of GPUs
* I have a local path for the model
* You'll want to do `wandb login` first if you want reporting
* Jon's script was set to `16bit`, `full_train`, `per_device_train_batch_size 12` which is fine if you have 8xA100s...
* I'm using `double_quant`, `nf4`, `4 bits`, no `full_train`, batch 1 and it still uses 22-23GB/card, but I'm not quite sure how the math works out, just that it takes about 1-2min/OOM failure, so that's a huge pain...

You can add, but maybe I'll do that next time (or just do it separately)
```
  --do_eval
  --do_mmlu_eval
```

Sharing the train/loss chart: [https://api.wandb.ai/links/augmxnt/eznbmx2x](https://api.wandb.ai/links/augmxnt/eznbmx2x)