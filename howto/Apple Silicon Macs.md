Macs are popular with (non-ML) developers, and the combination of (potentially) large amounts of unified GPU memory and decent memory bandwidth are appealing. [llama.cpp](https://github.com/ggerganov/llama.cpp) started as a project to run inference of LLaMA models on Apple Silicon (CPUs).

For non-technical users, there are several "1-click" methods that leverage llama.cpp:

- Nomic's [GPT4All](https://gpt4all.io/) - a Mac/Windows/Linux installer, model downloader, has a GUI, CLI, and API bindings
- [Ollama](https://ollama.ai/) - a brand new project with a slightly nicer chat window

**NOTE:** One important note is that while it's possible to use Macs for inference, if you're tempted to buy one primarily to use for LLMs (eg, a Mac Studio with 192GiB of RAM will cost about the same as a 48GB Nvidia A6000 Ada so seems like a good deal), be aware that Macs have some severe issues/limitations atm:

* When context becomes full, [llama.cpp currently suffers huge slowdowns](https://github.com/ggerganov/llama.cpp/issues/1730#issuecomment-1585580602) that manifest as multi-second pauses (computation falls back to CPU). If your goal is simply to run inference (chat with) the largest public models, you will get much better performance with say, 2 x 24GB RTX 3090s (~$1500 used) or a single RTX A6000 48GB ($4000).
* If you are planning on using Apple Silicon for ML/training, I'd also be wary. There are [multi-year long open bugs](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+apple+metal+sort%3Acreated-asc+) in PyTorch, and most major LLM libs like [bitsandbytes have no Apple Silicon support](https://github.com/TimDettmers/bitsandbytes/issues/485)

## llama.cpp

llama.cpp is a breeze to get running without any additional dependencies:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# where 8 is your threads for faster compiles
make clean && make LLAMA_METAL=1 -j8
```

Grab any Llama compatible GGML you want to try ([you can start here](https://huggingface.co/TheBloke)). I recommend q4_K_M as the sweet spot for quantize if you don't know which one to get.

You can run a simple benchmark to check for output and performance (most LLaMA 1 models should be `-c 2048`):
```
./main -m  ~/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin -ngl 1 -c 4096 -n 200 --ignore-eos
```

You can then run the built in web server and be off chatting at [http://localhost:8080/](http://localhost:8080/):
```
./server -c 4096 -ngl 1 -m ~/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin
```

If you are benchmarking vs other inference engines, I recommend using these standard settings:
```
./main -m <model> -ngl 1 -n 2048 --ignore-eos
```
* Metal uses `-ngl 1` (or any really) since it's unified memory, but for CUDA systems you'd want something like `-ngl 99` to get all layers in memory
* Default prompt context is 512 - this is probably fine to leave as is? Most testing I've seen online doesn't change this
* `-n` should be the max context you want to test to and `--ignore-eos` is required so it doesn't end prematurely (as context gets longer, speed tends to slow down

## MLC LLM
[MLC LLM](https://mlc.ai/mlc-llm/) is an implementation that runs not just on Windows, Linux, and Mac, but also iOS, Android, and even in web browsers w/ WebGPU support. Assuming you have `conda` setup already, the [instructions for installing](https://mlc.ai/mlc-llm/docs/get_started/try_out.html) are up to date and work without hitches.

Currently, the performance is about 50% slower than llama.cpp on my M2 MBA.