FYI, I ran the math through O1 (no code execution), Sonnet 3.5 (JS code execution) and Gemini 2.0 Pro (Python code execution) w/ the config JSON and Python to try to get a good sense of the architecture and some more exact stats. Hopefully, this is broadly right (but corrections welcomed):

- 28.81B activations per fwd pass / 452.82B total parameters
- Hybrid architecture: 3 dense layers + 58 8x256+1 MoE
- Uses YaRN RoPE extension to achieve 160K token context
- FP16 weights: 905.65GB , FP8 weights: 452.82GB
- FP16 kvcache: 286.55GB , FP8 kvcache: 143.28GB

At FP8 everything, might *just* fit into 1xH100 node, otherwise you'd need two, or an H200 or MI300X node...

# vs Llama 3
Here is a comparison to Llama 3:

| Parameter | DeepSeek-V3 | Llama3-70B | Llama3-405B |
|----------|---------------|---------------|---------------|
| Hidden Size | 7168 | 8192 | 16384 |
| Num Layers | 61 | 80 | 126 |
| Attn Heads | 128 | 64 | 128 |
| KV Heads | 128 | 8 | 8 |
| GQA Ratio | 1:1 | 8:1 | 16:1 |
| Head Dim | 56 | 128 | 128 |
| Interm Size | 18432 | 28672 | 53248 |
| Context Len | 163840 | 8192 | 131072 |
| Vocab Size | 129280 | 128256 | 128256 |

**FFN Expansion Ratios:**
- DeepSeek-V3 Dense Layers: 2.57x
- DeepSeek-V3 MoE Experts: 0.29x (but with 257 experts)
- Llama3-70B: 3.50x
- Llama3-405B: 3.25x

**Effective FFN Dimensions per Token:**
- DeepSeek-V3 Dense Layers: 18432
- DeepSeek-V3 MoE Layers: 16384 (2048 × 8 experts)
- Llama3-70B: 28672
- Llama3-405B: 53248

# vs Snowflake Arctic

The dense+moe hybrid maybe best compared to Snowflake Arctic (128 experts). Snowflake runs w/ parallel routing (more like Switch Transformer?) and DeepSeek-V3 is sequential (GLaM?) but they arrive at similar intermediate sizes (in most other ways, DeepSeek-V3 is bigger and badder, but to be expected):

| Parameter | DeepSeek-V3 | Arctic |
|------------|-------------|---------|
| Hidden Size | 7168 | 7168 |
| Num Layers | 61 | 35 |
| Attention Heads | 128 | 56 |
| KV Heads | 128 | 8 |
| GQA Ratio | 1:1 | 7:1 |
| Head Dimension | 56 | 128 |
| Context Length | 163840 | 4096 |
| Vocab Size | 129280 | 32000 |

**MoE Architecture:**

| Parameter | DeepSeek-V3 | Arctic |
|------------|-------------|---------|
| Architecture | 3 dense + 58 MoE layers | Dense-MoE hybrid (parallel) |
| Num Experts | 257 | 128 |
| Experts/Token | 8 | 2 |
| Base Params | ~10B | 10B |
| Expert Size | ~1.7B | 3.66B |
| Total Params | ~452B | ~480B |
| Active Params | ~29B | ~17B |

**FFN Expansion Ratios (DeepSeek-V3):**
- Dense Layers: 2.57x
- MoE Layers (per expert): 0.29x
- MoE effective expansion: 2.29x

**Effective FFN Dimensions per Token (DeepSeek-V3):**
- Dense Layers: 18432
- MoE Layers: 16384 (2048 × 8 experts)

**FFN Expansion Ratios (Arctic):**
- Dense (Residual) Path: 1.00x
- MoE Path (per expert): 0.68x
- Combined effective expansion: 2.36x

**Effective FFN Dimensions per Token (Arctic):**
- Dense Path: 7168
- MoE Path: 9728 (4864 × 2 experts)
- Total: 16896


# Reference
- https://claude.ai/chat/b725c82f-f3e8-402c-9023-956514e16242
- https://chatgpt.com/share/676c59ce-a060-8012-8cfb-36b416e9b61d
- https://claude.ai/chat/a9cd537a-f3fe-4d92-9f05-b8b7bea62d1a
- https://aistudio.google.com/app/prompts?state=%7B%22ids%22%3A%5B%2210PiaxDDrB03As-n8-5rQEWE2daleGmgp%22%5D%2C%22action%22%3A%22open%22%2C%22userId%22%3A%22107429578742599355954%22%2C%22resourceKeys%22%3A%7B%7D%7D&usp=drive_link

Other resources:
https://simonwillison.net/2024/Dec/25/deepseek-v3/

LiveBench Results:
https://www.reddit.com/r/LocalLLaMA/comments/1hm4959/benchmark_results_deepseek_v3_on_livebench/