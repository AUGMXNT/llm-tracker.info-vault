* [2023-08-14 Aman Sanger (cursor.so) comparing high batch throughput](https://twitter.com/amanrsanger/status/1690828453233463297)
- [2023-08-11 Optimizing latency](https://hamel.dev/notes/llm/inference/03_inference.html)
  - mlc, ctranslate2, vllm, tgi
  - A6000
  - batch 1 but focused on serving
- [2023-08-09 [Survey] Supported Hardwares and Speed](https://github.com/mlc-ai/mlc-llm/issues/15)
  - MLC LLM speeds for all their hardware (SOTA batch 1 perf)
  - [https://github.com/mlc-ai/llm-perf-bench](https://github.com/mlc-ai/llm-perf-bench)
    - MLC LLM vs ExLlama, llama.cpp
  - [2023-08-09 Making AMD GPUs competitive for LLM inference](https://blog.mlc.ai/2023/08/09/Making-AMD-GPUs-competitive-for-LLM-inference)
- [2023-07-31 7 Frameworks for Serving LLMs  
](https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407)
  - vLLM, TGI, CTranslate2, DS, OpenLLM, Ray Serve, MLC LLM
- [2023-07-06 LLaMa 65B GPU benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/14s7j9j/comment/jqy8shq/) - great benchmark and writeups 
  - 3090 v 4090 v A6000 v A6000 ADA
  - ExLlama, ExLlama_HF, llama.cpp

My testing:

- [2023-08-16 CPU shootoff](https://github.com/lhl/linuxlaptops/wiki/Minisforum-UM790-Pro#llamacpp)
  - 7940HS v 5950X v 1260P v M2
- [2023-08-03 Inference Engine Shootout](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1788227831)
  - MLC v llama.cpp v ExLlama
- [2023-07-28 3090 and 4090 Power Limit performance](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=535675890)
  - You can shave 50-100W off the PL and retain 97% of performance

 

Theory:

- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)