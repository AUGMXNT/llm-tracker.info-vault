# Summary
OmniQuant (omnidirectionally calibrated quantization) is a quantization technique published (2023-08-25) by [Wenqi Shao](https://wqshao126.github.io/) and [Mengzhao Chen](https://chenmnz.github.io/) from the [General Vision Group, Shanghai AI Lab](https://github.com/OpenGVLab). Instead of hand-crafted quantization parameters, OmniQuant uses trained Learnable Weight Clipping (LWC) that modulates the extreme values of weights by optimizing the clipping threshold and Learnable Equivalent Transformation (LET) to handle activation outliers.

The paper (Table 1) shows better WikiText2 perplexity than GPTQ and AWQ at every weight quantization for Llama 1 & 2, 7B-70B. Most interesting is that it has good (close to W4A16 RTN) perplexity at W3A16/W3A16g128. Also, there is also weight activation quantization that scores well in real-world benchmarks (ARC, HellaSwag, Winogrande, etc), with W6A6 results close to FP16 (Table 2).

On performance, the paper reports that using MLC the W4A16g128 performs well (2X unquantized FP16), but that W3A16g128 is only slightly faster - spoiler alert, but at the bottom of the page you can see my test results, which are much improved from the paper results.

* Paper: [https://arxiv.org/abs/2308.13137](https://arxiv.org/abs/2308.13137) ([PDF](https://arxiv.org/pdf/2308.13137.pdf))
* Code: [https://github.com/OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)
* [My memory usage testing of other quants on llama2-7b](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1788227831)

# Running a model
Per tradition, the installation instructions in the repo are incomplete/do not work.

Basic install works:
```
conda create -n omniquant python=3.10 -y
conda activate omniquant
git clone https://github.com/OpenGVLab/OmniQuant.git
cd OmniQuant
pip install --upgrade pip 
pip install -e .
```

If you would like to run one of the pre-compiled modules [using the instructions](https://github.com/OpenGVLab/OmniQuant#runing-quantized-models-with-mlc-llm), you will get an error (I've added comments):

```
# You probably have this already
conda install git git-lfs
git lfs install

# Where MLC expects to find models
mkdir dist && cd dist

# test Llama-2-7b-chat with w3a16g128 quantization
git clone https://huggingface.co/ChenMnZ/Llama-2-7b-chat-omniquant-w3a16g128asym

# not included, but duh
cd ..

# wah wah, the included executable won't run
./mlc_chat_cli --local-id Llama-2-7b-chat-omniquant-w3a16g128asym --device-name cuda
./mlc_chat_cli: error while loading shared libraries: libmlc_llm.so: cannot open shared object file: No such file or directory
```

So, I've [familiarized myself with MLC LLM before](https://llm-tracker.info/books/howto-guides/page/nvidia-gpus#bkmrk-mlc-llm) and the next obvious step is to install our own `mlc_chat_cli`:

```
conda install -c mlc-ai -c conda-forge mlc-chat-cli-nightly

# boo-urns - also note, the command line parameters have changed w/ recent versions
❯ mlc_chat_cli --model Llama-2-7b-chat-omniquant-w3a16g128asym --device cuda
Use MLC config: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/params/mlc-chat-config.json"
Use model weights: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/params/ndarray-cache.json"
Use model library: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/Llama-2-7b-chat-omniquant-w3a16g128asym-cuda.so"
You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out the latest stats (token/sec)
  /reset              restart a fresh chat
  /reload [model]  reload model `model` from disk, or reload the current model if `model` is not specified

Loading model...
mlc_chat_cli: symbol lookup error: /home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/Llama-2-7b-chat-omniquant-w3a16g128asym-cuda.so: undefined symbol: __cudaRegisterFatBinary
```
* [https://mlc.ai/mlc-llm/docs/get_started/try_out.html](https://mlc.ai/mlc-llm/docs/get_started/try_out.html)

Unfortunately, the nightly build **does not work with CUDA**. (if you search for `undefined symbol: __cudaRegisterFatBinary` you will find that it refers to binaries not being linked properly to the CUDA runtime libs (eg `-lcudart`). To solve this, you need to build your own `mlc_chat_cli`. I [use my previous docs](https://llm-tracker.info/books/howto-guides/page/nvidia-gpus#bkmrk-mlc-llm):
```
# Note, you need TVM, but you no longer need to build them separately for CUDA support
# I'm using CUDA 12.1, but pick your appropriate package
pip install --pre --force-reinstall mlc-ai-nightly-cu121 mlc-chat-nightly-cu121 -f https://mlc.ai/wheels

# yes, even with mlc-chat-nightlyt-cu121 CUDA error remains

# Compile
git clone https://github.com/mlc-ai/mlc-llm.git --recursive
cd mlc-llm
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j`nproc`

# You now have a usable mlc_chat_cli in mlc-llm/build
 ../mlc-llm/build/mlc_chat_cli --model Llama-2-7b-chat-omniquant-w3a16g128asym
Use MLC config: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/params/mlc-chat-config.json"
Use model weights: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/params/ndarray-cache.json"
Use model library: "/home/local/llm/omniquant/OmniQuant/dist/Llama-2-7b-chat-omniquant-w3a16g128asym/Llama-2-7b-chat-omniquant-w3a16g128asym-cuda.so"
You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out the latest stats (token/sec)
  /reset              restart a fresh chat
  /reload [model]  reload model `model` from disk, or reload the current model if `model` is not specified

Loading model...
Loading finished
Running system prompts...
System prompts finished
[INST]: /stats
prefill: 42862.4 tok/s, decode: -nan tok/s
[INST]: 
```
* [https://mlc.ai/mlc-llm/docs/install/tvm.html#install-tvm-unity](https://mlc.ai/mlc-llm/docs/install/tvm.html#install-tvm-unity)
* [https://mlc.ai/package/](https://mlc.ai/package/)

# Quantize A Model
OK, well that was a PITA, but lets turn our eye onto quantizing a model. Here, [the docs](https://github.com/OpenGVLab/OmniQuant#usage) are sadly incomplete, although one of the primary authors, [ChenMnZ](https://github.com/ChenMnZ) was [very quick to respond](https://github.com/OpenGVLab/OmniQuant/issues/4). Thanks!

I'll go through this step by step. First the shifts and scales available for download are only for their specific model-zoo, so you'll definitely need to generate them for your specific raw unquantized model:

```
python generate_act_scale_shift.py --model /models/llm/hf/meta-llama_Llama-2-7b-hf
```

From here, they suggest that you run 2) a weight-only quantization:
```
# W3A16g128
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /models/llm/hf/meta-llama_Llama-2-7b-hf  \
--epochs 20 --output_dir ./log/llama2-7b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc
```
However, if you do this, you will end up with a set of logs (about 200MB for llama2-7b) that you will need to further process. If you accidentally did this (which takes almost 2 hours on a 4090), then you can "set `--epochs` to `0` and add `--resume` ([see full reply](https://github.com/OpenGVLab/OmniQuant/issues/4#issuecomment-1703768260)) to the "fake quantize" step, but this "fake quantize" is really what you want to do. I recommend skipping to this directly:

```
# W3A16g128
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /models/llm/hf/meta-llama_Llama-2-7b-hf  \
--epochs 20 --output_dir ./log/llama2-7b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc \
--save_dir /models/llm/hf/llama2-7b-omniquant-w3a16g128
```

You now have a "fake quantize" checkpoint that is ready to be processed into a "real" quant. This pre-quant checkpoint will be the same size as an FP16 model (~13GB for llama2-7b).

After that, you will want to run `build.py` from your `mlc-llm` repo checkout, but here's the crucial step that's **undocumented in the README.md**. You need to add their quantization scheme to [mlc_llm/quantization/__init__.py](https://github.com/mlc-ai/mlc-llm/blob/main/mlc_llm/quantization/__init__.py):

```
    "w3a16g128asym": QuantizationScheme(
        name="w3a16g128asym",
        linear_weight=GroupQuantizationSpec(
            dtype="float16",
            mode="int3",
            sym=False,
            storage_nbit=16,
            group_size=128,
            transpose=False,
        ),
        embedding_table=None,
        final_fc_weight=None,
    ),
```
You can probably figure out the general pattern for any other quants you want to do.

Once you have this, then you can compile the MLC-LLM model, which should be quick:
```
python build.py --target cuda --quantization w3a16g128asym --model /models/llm/hf/llama2-7b-omniquant-w3a16g128 --use-cache=0

# Now you will have a new dist/ folder with the compiled model
```
* Note: `build.py` still looks at your active GPU to decide which architecture to use, so I set `CUDA_VISIBLE_DEVICES=1` to allow the model to get arch=sm_86 and allow my 3090 and 4090 to run the compiled model. There's no inferencing speed difference atm between architectures on the 4090.

OK, so this was everything I needed to get things working.

## Quantize Times
Note, OmniQuant runs 20 epochs per layer and takes some time to run. Here is the quantization time provided from Table A1 of the paper on how long LLaMA (1) models took the researchers to quant using a single NVIDIA A100-80G:

| LLaMA          | 7B   | 13B  | 30B  | 65B  |
|----------------|------|------|------|------|
| weight-only    | 1.1h | 2.2h | 4.5h | 8.9h |
| weight-activation | 1.6h | 3.3h | 7.3h | 14.4h |

On my RTX 4090, a llama2-7b took ~1.7-1.8h to for a weight-only quantize.


# Inferencing Performance
mlc_chat_cli afaik doesn't have an easy method for measuring perplexity, but it *does* have an ``--evaluate` flag now that lets me set my standard comparison parameters.

Here I was pleasantly surprised. The llama2-7b inferencing on the 4090 at 176 t/s actually manages to [beat MLC LLM's q4f16_1 results](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1788227831) (165 t/s, the previous speed champ).

The 3090 results are slower, but still respectable.

```
# 4090
❯ CUDA_VISIBLE_DEVICES=0 build/mlc_chat_cli --model llama2-7b-omniquant-w3a16g128-w3a16g128asym --evaluate --eval-prompt-len 128 --eval-gen-len 1920
...
[20:40:33] /home/local/llm/omniquant/mlc-llm/cpp/llm_chat.cc:706: [i: 2048] decoding-time=6.4203ms tok/s: 176.101.

# 3090
❯ CUDA_VISIBLE_DEVICES=0 build/mlc_chat_cli --model llama2-7b-omniquant-w3a16g128-w3a16g128asym --evaluate --eval-prompt-len 128 --eval-gen-len 1920
...
[20:45:35] /home/local/llm/omniquant/mlc-llm/cpp/llm_chat.cc:706: [i: 2048] decoding-time=12.3842ms tok/s: 88.736.
```

Just to make sure we were doing an apples-to-apples comparison though, I've also run the same test against a new `q4f16_1` build with the same (2023-09-02 TVM and MLC):

```
❯ CUDA_VISIBLE_DEVICES=1 python build.py --target cuda --quantization q4f16_1 --model /models/llm/hf/meta-llama_Llama-2-7b-hf --use-cache=0

❯ CUDA_VISIBLE_DEVICES=0 build/mlc_chat_cli --model meta-llama_Llama-2-7b-hf-q4f16_1 --evaluate --eval-prompt-len 128 --eval-gen-len 1920
...
[21:32:42] /home/local/llm/omniquant/mlc-llm/cpp/llm_chat.cc:706: [i: 2048] decoding-time=6.50777ms tok/s: 172.947.

❯ CUDA_VISIBLE_DEVICES=1 build/mlc_chat_cli --model meta-llama_Llama-2-7b-hf-q4f16_1 --evaluate --eval-prompt-len 128 --eval-gen-len 1920
...
[21:35:55] /home/local/llm/omniquant/mlc-llm/cpp/llm_chat.cc:706: [i: 2048] decoding-time=8.13898ms tok/s: 141.371.
```

It looks like TVM/MLC have gotten even faster in the past month or so, although on the 4090, WA3A16g128 still *barely* edges out q4f16_1. 

Check out my [performance page](https://llm-tracker.info/books/howto-guides/page/performance) for links to more perf benchmarks, although as this test shows, keep in mind that software perf is constantly improving.

Also note, in my testing, the W3A16g128 doesn't seem to save much more memory - nvidia-smi reported a top usage of 5.1GiB VRAM used vs 5.2-5.5GiB for MLC-LLM q4f16_1, GGML q4_K_M, and GPTQ q4_128gs. With MLC LLM, the W3A16g128 saves about 250MiB of memory vs the q4f16_1 or about 4-5%, so it's not nothing, especially as model sizes grow. And according to Table 3 of the paper, a LLaMA-65B W4A16 uses 41.0G and the W3A16 uses only 35.1G.

While OmniQuant does seem like an incremental improvement overall, it's probably not worth running out and requantizing all your models, especially considering how time consuming it can be (and how much less usable MLC LLM is atm for most batch=1 workflows). That being said, it looks like it *does* work, and is a viable option to use if you are trying to squeeze into the least memory available. Also, I've only tested out the weight quantization model, and have no idea how the weight-activation models perform, but I'll leave that for others to test.