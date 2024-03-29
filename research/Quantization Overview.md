## Tests
* [How does quantisation affect model output?](https://rentry.org/quants) - 15 basic tests on different quant levels
* [A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, model size, and loading time.](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)
* [A direct comparison between llama.cpp, AutoGPTQ, ExLlama, and transformers perplexities](https://oobabooga.github.io/blog/posts/perplexities/)

To test:
* Memory high water mark
* Prompt/Inference speed
* Perplexity
* Standardized benchmark numbers (OpenAPI layer w/ lm-eval?)
  * Sweep of fastest tests (Wiki, C4, Winogrande, PIQA, ARCe) - lambada and coqa? - run standard timings first
  * EvalPlus (code!)
* KL Divergence: https://www.reddit.com/r/LocalLLaMA/comments/1816h1x/how_much_does_quantization_actually_impact_models/


## Formats

### QuIP#

- [https://cornell-relaxml.github.io/quip-sharp/](https://cornell-relaxml.github.io/quip-sharp/)
  - https://github.com/turboderp/exllamav2/issues/176

### EXL2 (ExLlamaV2)

- [https://github.com/turboderp/exllamav2](https://github.com/turboderp/exllamav2)
- Based off of GPTQ but iteratively selects from quants vs calibration and averages bit depth to target an arbitrary bit-weight

### OmniQuant

Here are my docs for how run it: [OmniQuant](https://llm-tracker.info/books/llms/page/omniquant)

- [https://github.com/OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)
- [https://arxiv.org/abs/2308.13137](https://arxiv.org/abs/2308.13137)
- Better than GPTQ (OPTQ), AWQ, SmoothQuant
- MLC compatible

### QuIP

- [https://github.com/jerry-chee/QuIP](https://github.com/jerry-chee/QuIP)
- [https://github.com/AlpinDale/QuIP-for-Llama](https://github.com/AlpinDale/QuIP-for-Llama)
- [https://arxiv.org/abs/2307.13304](https://arxiv.org/abs/2307.13304)
- Not just PPL but also benchmark accuracy tests
- 3-bit almost matches FP1


### SqueezeLM

- [https://github.com/SqueezeAILab/SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM)
- [https://arxiv.org/abs/2306.07629](https://arxiv.org/abs/2306.07629)

### AWQ

- [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
- [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- [https://github.com/casper-hansen/AutoAWQ/](https://github.com/casper-hansen/AutoAWQ/)

### GGML k-quants

- [https://github.com/ggerganov/llama.cpp/pull/1684](https://github.com/ggerganov/llama.cpp/pull/1684)

### SmoothQuant

- [https://github.com/mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
- [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)

### SpQR

- [https://github.com/Vahe1994/SpQR](https://github.com/Vahe1994/SpQR)
- [https://arxiv.org/abs/2306.03078](https://arxiv.org/abs/2306.03078)

### GPTQ/OPTQ

- [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- [https://openreview.net/forum?id=tcbBPnfwxS](https://openreview.net/forum?id=tcbBPnfwxS)