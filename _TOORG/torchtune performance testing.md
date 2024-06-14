RTX 4090
```bash
❯ CUDA_VISIBLE_DEVICES=0 tune run lora_finetune_single_device --config torchtune/recipes/configs/llama3/8B_lora_single_device.yaml

1|25880|Loss: 0.8484776020050049: 100%|█████████████████████████████████████████████| 25880/25880 [2:30:00<00:00,  2.88it/s]
INFO:torchtune.utils.logging:Model checkpoint of size 16.06 GB saved to /tmp/Meta-Llama-3-8B-Instruct/meta_model_0.pt
INFO:torchtune.utils.logging:Adapter checkpoint of size 0.01 GB saved to /tmp/Meta-Llama-3-8B-Instruct/adapter_0.pt
```

RTX 3090
```bash
❯ CUDA_VISIBLE_DEVICES=1 tune run lora_finetune_single_device --config torchtune/recipes/configs/llama3/8B_lora_single_device.yaml

1|25880|Loss: 0.8496665358543396: 100%|███████████| 25880/25880 [4:59:53<00:00,  1.44it/s]
INFO:torchtune.utils.logging:Model checkpoint of size 16.06 GB saved to /tmp/Meta-Llama-3-8B-Instruct/meta_model_0.pt
INFO:torchtune.utils.logging:Adapter checkpoint of size 0.01 GB saved to /tmp/Meta-Llama-3-8B-Instruct/adapter_0.pt
```