This document tries to avoid using the term "performance" since in ML research the term performance typically refers to measuring model quality/capabilities.

This guide is a cheat sheet for the other type of performance:  testing the speed, throughput, and efficiency between different consumer hardware platforms for LLM inference and other end-user tasks.
- If you're looking for industrial benchmarking, the best place to start is probably Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering/), particularly the [Compute/Accelerator](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator) section. Last year I did a long writeup on [tuning vLLM for MI300X](https://shisa.ai/posts/tuning-vllm-mi300x/) that links to many resources as a good jump

# Background and Definitions

What we need to know about LLMs:
- LLMs are basically a big pile of numbers (matrices) They have different sizes, which is **parameter** count - When you see 7B, 8B, 14B, this is an approximate count of how many parameters (in billions).
- Quantization - Models weights (the parameters) used to be stored as FP32 (4 bytes), then FP16/BF16. Commercially, FP8 and INT8 are quite common, and FP4 and INT4 are emerging. At home, "Q4", which is roughly ~4-bit is used most often used, but there are even smaller quants that are usable these days (down to ~1.58b). Note, performance loss is not linear - a good Q4 quant can be close to or even
- Weights, Activations, Computational Precision - 
- 
- for quants, normally we are talking about the size and memory of weights, but new accelerators support 

What we want to measure/compare
- Memory usage

Testing notes:
- The pace of development is very quickly and exact version numbers can have dramatic changes on speed. It's important for comparisons to specify the exact build number and driver/env for reproducibility
- You can't stick to an older build since there will be unfixed bugs, unsupported models etc, and all end-user front-ends (LM Studio, Ollama, Jan, etc are constantly releasing updates to trail llama.cpp releases)
