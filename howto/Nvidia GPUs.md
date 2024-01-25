Nvidia GPUs are the most compatible hardware for AI/ML. All of Nvidia's GPUs (consumer and professional) support CUDA, and basically all popular ML libraries and frameworks support CUDA.

The biggest limitation of what LLM models you can run will be how much GPU VRAM you have. The [r/LocalLLaMA wiki](https://www.reddit.com/r/LocalLLaMA/wiki/models/) gives a good overview of how much VRAM you need for various quantized models.

Nvidia cards can run [CUDA with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) which means that generally, all software will work both in Linux and Windows. If you are serious about ML, there are still advantages to Linux like [better performance](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/), less VRAM usage (ability to run headless), and probably some [other edge cases](https://www.reddit.com/r/LocalLLaMA/comments/14fb9c0/last_nvidia_drivers_let_you_use_the_shared_memory/).

For inferencing you have a few options:
* [llama.cpp](https://github.com/ggerganov/llama.cpp) - As of July 2023, llama.cpp's CUDA performance is on-par with the [ExLlama](https://github.com/turboderp/exllama), generally be the fastest performance you can get with quantized models. GGMLv3 is a convenient single binary file and has a variety of well-defined quantization levels (k-quants) that have slightly better perplexity than the most widely supported alternative, GPTQ. It however, is slightly less memory efficient, eg, potentially running OOM on 33B models on 24GiB GPUs when exllama does not.
  * llama.cpp is best for low-VRAM GPUs since you can offload layers to run on the GPU (use `-ngl <x>` to set layers and `--low-vram` to move the cache to system memory as well. The more layers you can load into VRAM, the faster your model will run.
  * llama.cpp is a *huge* project with many active contributors, and now has some VC backing as well
* [ExLlama](https://github.com/turboderp/exllama) - if you have an RTX 3000/4000 GPU, this is probably going to be your best option. It is on par in performance with llama.cpp, and also is [the most memory efficient implementation](https://github.com/turboderp/exllama#results-so-far) available. If you are splitting a model between multiple GPUs, ExLLama seems to have the most efficient performance when splitting inferencing between cards.
  * ExLlama is a smaller project but contributions are being actively merged (I submitted a PR) and the maintainer is super responsive.
* [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) - this engine, while generally slower may be better for older GPU architectures. There is a CUDA and Triton mode, but the biggest selling point is that it can not only inference, but also quantize and fine-tune many model types.
  * Unfortunately, recently PRs have been slow to get upstreamed and [the maintainer has been busy/inaccessible](https://github.com/PanQiWei/AutoGPTQ/issues/187).
* [MLC LLM](https://mlc.ai/mlc-llm/) - this was a bit of a challenge to setup, but turns out to perform quite well (perhaps better than all other engines?)


##  CUDA Version Hell
The bane of your existence is probably going to be managing all the different CUDA versions that are required for various libraries. Recommendations:

* Use `conda` (well, `mamba` lest you want to grow old and die waiting for dependencies to calculate). If you don't know where to start, just [install Mambaforge directly](https://github.com/conda-forge/miniforge#mambaforge) and create a new environment for every single library.
* Install the [exact version of CUDA](https://anaconda.org/nvidia/cuda-toolkit/labels) that you need for each environment and point to it, eg:
  ```
  conda create -n autogptq
  conda activate autogptq
  mamba install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
  conda env config vars set CUDA_PATH="$CONDA_PREFIX"
  conda env config vars set CUDA_HOME="$CONDA_PREFIX"
  ```
  Additionally, if you need to use a specific g++ version (eg, CUDA 11.7 requires g++ < 12):
  ```
  # see valid versions: https://anaconda.org/conda-forge/gxx/files
  mamba install gxx=11.4.0
  ```
  And if you need to install PyTorch manually...
  ```
  mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
* This should be good enough, but if all else fails, you can use a [custom Docker container](https://docs.docker.com/compose/gpu-support/) as well.
* There's [envd](https://github.com/tensorchord/envd), a Docker addon that promises easier dev environments for AI/ML, although it also has a number of [open bugs](https://github.com/tensorchord/envd/issues?q=is%3Aissue+is%3Aopen+label%3A%22type%2Fbug+%F0%9F%90%9B%22)

# Inferencing Packages
| Package       | Commit  | Model         | Quant    | Memory Usage | 4090 @ 400PL | 3090 @ 360PL |
|---------------|---------|---------------|----------|--------------|--------------|--------------|
| MLC LLM CUDA  | 3c53eeb | llama2-7b-chat | q4f16_1  | 5932         | 115.87       | 83.63        |
| MLC LLM Perf  | c40be6a | llama2-7b-chat | q4f16_1  | 5244         | 165.57       | 131.73       |
| llama.cpp     | 8183159 | llama2-7b-chat | q4_0     | 5226         | 146.79       | 125.54       |
| llama.cpp     | 8183159 | llama2-7b     | q4_K_M   | 5480         | 138.83       | 114.66       |
| ExLlama       | 91b9b12 | llama2-7b-chat | q4_128gs | 5466         | 115.92       | 81.91        |
| ExLlama       | 91b9b12 | llama2-7b     | q4_32gs_act | 5672    | 107.21       | 73.54        |
* See [LLM Worksheet](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1788227831) for more details


## MLC LLM
[mlc-llm](https://github.com/mlc-ai/mlc-llm) is an interesting project that lets you compile models (from HF format) to be used on multiple platforms (Android, iOS, Mac/Win/Linux, and even WebGPU). On PC however, the [install instructions](https://mlc.ai/mlc-llm/#windows-linux-mac) will only give you a pre-compiled Vulkan version, which is [much slower than ExLLama or llama.cpp](https://github.com/mlc-ai/mlc-llm/issues/15#issuecomment-1657190790), however when a CUDA version is compiled, it looks like it's actually possibly the fastest inferencing engine currently available (2023-08-03).

Here's how to set up on Arch Linux
```
# Required
paru -S rustup
rustup default stable
paru -S llvm

# Environment
conda create -n mlc
conda activate mlc
mamba install pip

# Compile TVM
git clone https://github.com/mlc-ai/relax.git --recursive
cd relax
mkdir build
cp cmake/config.cmake build
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' build/config.cmake
sed -i 's/set(USE_GRAPH_EXECUTOR_CUDA_GRAPH OFF)/set(USE_GRAPH_EXECUTOR_CUDA_GRAPH ON)/g' build/config.cmake
sed -i 's/set(USE_CUDNN OFF)/set(USE_CUDNN ON)/g' build/config.cmake
sed -i 's/set(USE_CUBLAS OFF)/set(USE_CUBLAS ON)/g' build/config.cmake

make -j`nproc`
export TVM_HOME=`pwd`
cd ..

# Make model
# IMPORTANT: CUDA is targeted per GPU. Be sure to use CUDA_VISIBLE_DEVICES if you have multiple generations of CUDA...
# NOTE: the maximum context length is determined for a model at compile-time here. It defaults to 2048 so you will want to set it to a longer one if your model supports it (I don't believe MLC currently supports RoPE extension
git clone https://github.com/mlc-ai/mlc-llm.git --recursive
cd mlc-llm
python3 -m mlc_llm.build --target cuda --quantization q4f16_1 --model /models/llm/llama2/meta-llama_Llama-2-7b-chat-hf --max-seq-len 4096

# Compile mlc-llm
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j`nproc`
cd ..

# In `mlc-llm` folder you should now be able to run
build/mlc_chat_cli --local-id meta-llama_Llama-2-7b-chat-hf-q4f16_1 --device-name cuda --device_id 0 --evaluate --eval-prompt-len 3968 --eval-gen-len=128
```
* [https://mlc.ai/mlc-llm/docs/compilation/compile_models.html](https://mlc.ai/mlc-llm/docs/compilation/compile_models.html)
* [https://github.com/mlc-ai/mlc-llm/issues/229#issuecomment-1564139277](https://github.com/mlc-ai/mlc-llm/issues/229#issuecomment-1564139277)


Note, the main branch, as of 2023-08-03 runs at about the same speed as ExLlama and a behind llama.cpp, however there is a separate "benchmark" version that has performance optimizations that have not yet made it's way back to the main branch. This can be found at this repo: https://github.com/junrushao/llm-perf-bench

And here's how to get it working:
```
paru -S cutlass
# Otherwise GLIBCXX_3.4.32 not happy
mamba install cmake

git clone --recursive https://github.com/junrushao/mlc-llm/ --branch benchmark mlc-llm.junrushao-benchmark
cd mlc-llm.junrushao-benchmark

# Adapted from https://github.com/junrushao/llm-perf-bench/blob/main/install/tvm.sh
export MLC_HOME=`pwd`
export TVM_HOME=$MLC_HOME/3rdparty/tvm
export PYTHONPATH=$TVM_HOME/python
cd $TVM_HOME && mkdir build && cd build && cp ../cmake/config.cmake .
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >>config.cmake
echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >>config.cmake
echo "set(USE_GTEST OFF)" >>config.cmake
echo "set(USE_CUDA ON)" >>config.cmake
echo "set(USE_LLVM ON)" >>config.cmake
echo "set(USE_VULKAN OFF)" >>config.cmake
echo "set(USE_CUTLASS ON)" >>config.cmake
cmake .. && make -j$(nproc)

# Adapted from https://github.com/junrushao/llm-perf-bench/blob/main/install/mlc.sh
cd $MLC_HOME && mkdir build && cd build && touch config.cmake
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >>config.cmake
echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >>config.cmake
echo "set(USE_CUDA ON)" >>config.cmake
echo "set(USE_VULKAN OFF)" >>config.cmake
echo "set(USE_METAL OFF)" >>config.cmake
echo "set(USE_OPENCL OFF)" >>config.cmake
cmake .. && make -j$(nproc)

cd $MLC_HOME
CUDA_VISIBLE_DEVICES=0 python build.py --target cuda --quantization q4f16_1 --model /models/llm/llama2/meta-llama_Llama-2-7b-chat-hf --use-cache=0

### or in the Docker container
micromamba activate python311
cd $MLC_HOME
CUDA_VISIBLE_DEVICES=0 python build.py   --model /models/llm/llama2/meta-llama_Llama-2-7b-chat-hf   --target cuda   --quantization q4f16_1   --artifact-path "./dist"   --use-cache 0
mv dist/meta-llama_Llama-2-7b-chat-hf-q4f16_1 dist/4090-meta-llama_Llama-2-7b-chat-hf-q4f16_1

CUDA_VISIBLE_DEVICES=1 python build.py   --model /models/llm/llama2/meta-llama_Llama-2-7b-chat-hf   --target cuda   --quantization q4f16_1   --artifact-path "./dist"   --use-cache 0
mv dist/meta-llama_Llama-2-7b-chat-hf-q4f16_1 dist/3090-meta-llama_Llama-2-7b-chat-hf-q4f16_1

# copy dist folder to where you want
scp -r -P 45678 root@0.0.0.0:/mlc-llm/dist ./

```

On my 4090, the `q4f16_1` is 165.98 t/s vs 106.70 t/s for a `q4 32g act-order GPTQ` w/ ExLlama, and 138.83 t/s with a `q4_K_M` GGMLv3 with llama.cpp.


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

# Mobile Power Limits
Mostly locked by VBIOS, but you might be able to run `nvidia-powerd` and get +10W
- https://forums.developer.nvidia.com/t/power-limit-on-3000-mobile-series/193443/3
- https://github.com/NVIDIA/open-gpu-kernel-modules/issues/483
- https://www.reddit.com/r/razer/comments/oqnjbl/11_performance_with_135w_vbios_on_my_razer_blade/