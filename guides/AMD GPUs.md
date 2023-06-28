As of June 2023, AMD's [ROCm](https://github.com/RadeonOpenCompute/ROCm) GPU compute software stack is only supported by Linux.

 

## Windows

### llama.cpp

Go to [llama.cpp's release page](https://github.com/ggerganov/llama.cpp/releases) and download a `bin-win-clblast` version.

In the Windows terminal, run it with `-ngl 99` to load all the layers into memory.

You may be able to run a llama.cpp release

If you are looking to get hardware acceleration on Window