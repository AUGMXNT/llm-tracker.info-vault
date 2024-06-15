Post in r/localllama, finetuning

2024-06 Comparison


* Repo
* WandB
	* trainer-test

Set same LR, warmup, etc - should be same
weight_decay, dropout

Hardware
Driver version
nvcc --version
PyTorch

memory-high-water-4090
* highwater
* starting
* max memory usage

# RTX 4090
100%|█████████████████████████████████████████████| 25881|25880|Loss: 0.8484776020050049: 100%|█████████████████████████████████████████████| 25880/25880 [2:30:00<00:00,  2.88it/s]

# RTX 3090

# 7900 XTX

# W7900

19.540GiB
~280/303W

1|404|Loss: 0.8711342215538025: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 404/404 [4:43:52<00:00, 42.16s/it]



hiplast

gdb

# vs Unsloth

4090
330/450W
17.178Gi
❯ time CUDA_VISIBLE_DEVICES=0 python train.py


vs autotrain

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 51,760 | Num Epochs = 1
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 64
\        /    Total batch size = 128 | Total steps = 60
 "-____-"     Number of trainable parameters = 20,971,520
 ```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [14:58<00:00, 14.98s/it]

real    15m20.123s
user    12m34.279s
sys     3m17.287s

vs Axolotl


```
❯ time CUDA_VISIBLE_DEVICES=0 python train.py 
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
==((====))==  Unsloth: Fast Llama patching release 2024.6
   \\   /|    GPU: NVIDIA GeForce RTX 3090. Max memory: 23.677 GB. Platform = Linux.
O^O/ \_/ \    Pytorch: 2.3.0+cu121. CUDA = 8.6. CUDA Toolkit = 12.1.
\        /    Bfloat16 = TRUE. Xformers = 0.0.26.post1. FA = True.
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.41it/s]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
meta-llama/Meta-Llama-3-8B-Instruct does not have a padding token! Will use pad_token = <|reserved_special_token_250|>.
Unsloth 2024.6 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 51,760 | Num Epochs = 1
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 64
\        /    Total batch size = 128 | Total steps = 60
 "-____-"     Number of trainable parameters = 20,971,520
```