# CUDA

## Version Hell
The bane of your existence is probably going to be managing all the different CUDA versions that are required for various libraries. Recommendations:

* Use `conda` (well, `mamba` lest you want to grow old and die waiting for dependencies to calculate). If you don't know where to start, just [install Mambaforge directly](https://github.com/conda-forge/miniforge#mambaforge) and create a new environment for every single library.
* Install the [exact version of CUDA](https://anaconda.org/nvidia/cuda-toolkit/labels) that you need for each environment and point to it, eg:
  ```
  conda create -n autogptq
  conda activate autogptq
  mamba install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
  conda env config vars set CUDA_PATH="$CONDA_PREFIX"
  ```
* This should be good enough, but if all else fails, you can use a [custom Docker container](https://docs.docker.com/compose/gpu-support/) as well.


# Tips and Tricks
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
* For inferencing, I can lower my 4090 from 450W to 360W and only lose about 1-2% performance but everyone should test for themselves what works best for their setup.