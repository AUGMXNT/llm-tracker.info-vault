Monitor your Nvidia GPUs with either:
```
watch nvidia-smi
```
* or [nvtop](https://github.com/Syllo/nvtop)

You can lower power limits if you're inferencing:
```
sudo nvidia-smi -i 0 -pl 360
```
* You can get your GPU IDs with `nvidia-smi -L`
* For inferencing, I find I can lower my 4090 from 450W to 360W and only lose about 1-2% performance but everyone should test for themselves what works best for their setup.