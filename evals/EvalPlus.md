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
Running the tests:
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

|                       | Q5_K_L | Q8     | EPL  | Delta |
| --------------------- | ------ | ------ | ---- | ----- |
| Humaneval Pass@1      | 90.9   | 90.9   | 92.1 | -1.3% |
| Humaneval+ Pass@1     | 86.0   | 86.0   | 87.2 | -1.4% |
| Humaneval Time (mins) | 16.430 | 16.360 |      |       |
| MBPP Pass@1           | 90.7   | 90.7   | 90.5 | +0.2% |
| MBPP+ Pass@1          | 77.2   | 77.0   | 77.0 | +0.3% |
| MBPP Time (mins)      | 24.230 | 23.530 |      |       |
- The EPL column is the [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) results. Q5_K_L and Q8 has a relatively minor loss to the full FP16 model, and there isn't much difference between the Q8 (33GB) vs the Q5_K_L (23GB)
- As you can see by the Delta, the performance loss is pretty negligble/within margin of error on both the Q5 and Q8 quants. This corroborate recent testing of HumanEval results at various quants where Qwen2.5-Coder-32B is strong enough to see almost no loss in benchmark performance down to 2.5bpw: https://www.reddit.com/r/LocalLLaMA/comments/1gsyp7q/comment/lxjrw01/
- Interestingly, on the W7900 w/ the 1.5B draft model, there's almost no performance difference using the Q8. Real world usage may differ I suspect. Sadly, there isn't a great way for me to log the server performance data, but eyeballing it, speed was about 400 t/s for prefill, and about 50 t/s for text generation

# Other Coding Models
Qwen2.5-Coder is an incredibly strong coding model family and all sizes except for 3B are Apache 2.0: https://qwenlm.github.io/blog/qwen2.5-coder-family/

Note, the [new Qwen QwQ model](https://qwenlm.github.io/blog/qwq-32b-preview/) (also 32B) may be stronger for hard coding challenges as it scores a 50.0 on LiveCodeBench 2024.08-2024.11 while [Qwen2.5-Coder-32B-Instruct scores a 28.9 on the same test set](https://livecodebench.github.io/leaderboard.html).

