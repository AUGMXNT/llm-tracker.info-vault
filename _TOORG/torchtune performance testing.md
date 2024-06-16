Trainers
https://github.com/OpenAccess-AI-Collective/axolotl

# RTX 4090
```bash
❯ CUDA_VISIBLE_DEVICES=0 tune run lora_finetune_single_device --config torchtune/recipes/configs/llama3/8B_lora_single_device.yaml

1|25880|Loss: 0.8484776020050049: 100%|█████████████████████████████████████████████| 25880/25880 [2:30:00<00:00,  2.88it/s]
INFO:torchtune.utils.logging:Model checkpoint of size 16.06 GB saved to /tmp/Meta-Llama-3-8B-Instruct/meta_model_0.pt
INFO:torchtune.utils.logging:Adapter checkpoint of size 0.01 GB saved to /tmp/Meta-Llama-3-8B-Instruct/adapter_0.pt
```

# RTX 3090
```bash
❯ CUDA_VISIBLE_DEVICES=1 tune run lora_finetune_single_device --config torchtune/recipes/configs/llama3/8B_lora_single_device.yaml

1|25880|Loss: 0.8496665358543396: 100%|███████████| 25880/25880 [4:59:53<00:00,  1.44it/s]
INFO:torchtune.utils.logging:Model checkpoint of size 16.06 GB saved to /tmp/Meta-Llama-3-8B-Instruct/meta_model_0.pt
INFO:torchtune.utils.logging:Adapter checkpoint of size 0.01 GB saved to /tmp/Meta-Llama-3-8B-Instruct/adapter_0.pt
```

# 7900 XTX
wtf hipblas!
https://github.com/ROCm/rocBLAS/issues/1339





Compile is 15% Faster


21.247GB
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 1
gradient_accumulation_steps: 8
4h

21.247GB
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 1
gradient_accumulation_steps: 16
4h

21.247GB
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 1
gradient_accumulation_steps: 64
4h

21.320GB
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 1
gradient_accumulation_steps: 64
compile: True
3h

26.946GB
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 4096
batch_size: 1
gradient_accumulation_steps: 64
3:40

44.677Gi
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 8192
batch_size: 1
gradient_accumulation_steps: 64
x

38.337
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 4096
batch_size: 2
gradient_accumulation_steps: 64
x

38.337
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 4096
batch_size: 2
gradient_accumulation_steps: 64
x

26.734
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 2
gradient_accumulation_steps: 64
3:20

? bs=4
torch.optim.AdamW (bitsandbytes doesn't work for ROCm)
max_seq_len: 2048
batch_size: 2
gradient_accumulation_steps: 64


compiled version

210