[llama.cpp](https://github.com/ggerganov/llama.cpp) is the most popular backend for inferencing Llama models for single users.

- Started out for CPU, but now supports GPUs, including best-in-class CUDA performance, and recently, ROCm support. It also has fallback CLBlast support, but performance on that is not great.
- Has their own GGUF file format (was GGMLv1-3 before) that is a single file metadata container (think MOV or MKV for models). GGUF uses their own custom quant and binary layout. Note, these format changes have always breaking/non-compatible, but maybe GGUF will be flexible enough to not be.
- See [ggml](https://github.com/ggerganov/ggml) for other model types (some of which are getting folded into llama.cpp?)
- Very active community with dozens of contributors: [https://github.com/ggerganov/ggml/graphs/contributors](https://github.com/ggerganov/ggml/graphs/contributors)
- Many end-user client projects like LMStudio, Ollama, GPT4all, KoboldCPP, etc either use llama.cpp or a fork as their backend.

More:

- [A guide to running lora/qlora models on Llama.cpp](https://ragntune.com/blog/A-guide-to-running-Llama-2-qlora-loras-on-Llama.cpp)