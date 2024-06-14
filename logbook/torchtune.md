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

# RTX 3090

# 7900 XTX

# W7900

19.540GiB
~280/303W

hiplast

gdb

# vs Unsloth

4090
330/450W
17.178Gi



vs autotrain

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 51,760 | Num Epochs = 1
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 64
\        /    Total batch size = 128 | Total steps = 60
 "-____-"     Number of trainable parameters = 20,971,520
 ```


vs Axolotl
