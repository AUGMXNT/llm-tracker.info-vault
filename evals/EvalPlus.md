https://github.com/evalplus/evalplus
# Install
```
# evalplus[vllm] installs vllm
# evalplus[perf,vllm] - didn't use it...
pip install --upgrade "evalplus @ git+https://github.com/evalplus/evalplus"
```
# Model Info (Qwen 2.5 Coder 32B)
We're testing things 2024-12-01 using Qwen2.5-Coder 32B Instruct, basically the best open weights coding model available at that date (QwQ maybe competes):
- https://qwenlm.github.io/blog/qwen2.5-coder-family/
- https://github.com/QwenLM/Qwen2.5-Coder?tab=readme-ov-file

I'm running this on an AMD Radeon W7900 (gfx1100, basically a 48GB 7900 XTX at 240W TDP) with Qwen2.5-Coder 1.5B Instruct Q8 as Speculative Decoding:
- https://github.com/mostlygeek/llama-swap/blob/main/examples/speculative-decoding/README.md
- https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/
- https://www.reddit.com/r/LocalLLaMA/comments/1h2lrh2/dual_rx_7900_xtx/

Notes:
- I did some basic testing and found 1.5B to be a bit faster than 0.5B
- Due to various tokenizer differences between different sized models, I used Bartowski's GGUFs https://huggingface.co/bartowski
- I use https://github.com/hjc4869/llama.cpp which I found to be up to 20-40% faster than upstream for long context. See also this discussion and marvel at how AMD can't be bothered to even send a 7900 XTX to the llama.cpp ROCm maintainer, much less spend like 0.2 FTE time to help do CI and keep optimizations unbroken on the most widely used desktop inferencing engine in the world: https://github.com/ggerganov/llama.cpp/issues/10439

Running the OpenAI compatible server:
```
~/ai/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q5_K_L.gguf -md /models/gguf/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf --draft-max 24 --draft-min 1 --draft-p-min 0.6 -ngl 99 -ngld 99
```
- 32B Q8 + 1.5B Q8
	- 36GB (@ 4K)
	- 45GB w/ 28000 (max @ FP16 context)
- 32B Q5_K_L + 1.5B Q8
	- 26GB @ 4K
	- 37GB @ 32768 (`-c 32768`)
	- 44GB @ 61000 (max @ FP16 context)
- Native context for Qwen2.5 Coder is 32K although you can apply RoPE extension. You can use `-ctk` and `-ctv` to q4 or q8 if you need to save some meory:
	- https://www.reddit.com/r/LocalLLaMA/comments/1dalkm8/memory_tests_using_llamacpp_kv_cache_quantization/
- I did some tuning on efficiency with draft parameters, but didn't do a very systematic hyper-parameter sweep...
# Results
```
time evalplus.evaluate --model "Qwen2.5 Coder 32B Instruct Q8" \
                        --dataset 'humaneval'             \
                        --base-url http://localhost:8080/v1    \
                        --backend openai --greedy
```

```
time evalplus.evaluate --model "Qwen2.5 Coder 32B Instruct Q8" \
                        --dataset 'mbpp'             \
                        --base-url http://localhost:8080/v1    \
                        --backend openai --greedy
```