https://github.com/evalplus/evalplus

# Install
```
# evalplus[vllm] installs vllm
# evalplus[perf,vllm] - didn't use it...
pip install --upgrade "evalplus @ git+https://github.com/evalplus/evalplus"
```

# Model
We're testing things 2024-12-01 using Qwen2.5-Coder 32B Instruct, basically the best open weights coding model available at that date (QwQ maybe competes):
- https://qwenlm.github.io/blog/qwen2.5-coder-family/
- https://github.com/QwenLM/Qwen2.5-Coder?tab=readme-ov-file

I'm running this on an AMD Radeon W7900 (gfx1100, basically a 48GB 7900 XTX at 240W TDP) with Qwen2.5-Coder 1.5B Instruct Q8 as Speculative Decoding 

I'm using 37GB


```
~/ai/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf -md /models/gguf/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf --draft-max 24 --draft-min 1 --draft-p-min 0.6 -ngl 99 -ngld 99 -n 32768
```

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