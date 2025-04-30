https://andymasley.substack.com/p/individual-ai-use-is-not-bad-for
https://news.ycombinator.com/item?id=42745847

Nice to see independent numerical analysis. BTW, it may be worth noting that the ChatGPT power numbers you use come from a April 2023 paper (Li et al, arXiv:2304.03271) that estimates water/power usage based off of GPT-3 (175B dense model) numbers published from *2021* numbers. There was a newer paper that directly measured power usage on a Llama 65B (on V100/A100 hardware) that showed a 14X better efficiency (Samsi et al, arXiv:2310.03003).

Since I've been running my own tests, I decided what this looked like using similar direct testing methodology in 2025. On the latest vLLM w/ Llama 3.3 70B FP8, my results were 120X more efficient than the commonly cited Li et al numbers.

For those interested in a full analysis/table with all the citations (including my full testing results) see this o1 chat:

[https://chatgpt.com/share/678b55bb-336c-8012-97cc-b94f70919daa](https://chatgpt.com/share/678b55bb-336c-8012-97cc-b94f70919daa)

https://news.ycombinator.com/item?id=42746644
I published this as a comment as well, but it's probably worth nothing that the ChatGPT water/power numbers cited (the one that is most widely cited in these discussions) comes from an April 2023 paper (Li et al, arXiv:2304.03271) that estimates water/power usage based off of GPT-3 (175B dense model) numbers published from OpenAI's original *GPT-3 2021 paper*. From Section 3.3.2 Inference:

> As a representative usage scenario for an LLM, we consider a conversation task, which typically includes a CPU-intensive prompt phase that processes the user’s input (a.k.a., prompt) and a memory-intensive token phase that produces outputs [37]. More specifically, we consider a medium-sized request, each with approximately ≤800 words of input and 150 – 300 words of output [37]. The official estimate shows that GPT-3 consumes an order of 0.4 kWh electricity to generate 100 pages of content (e.g., roughly 0.004 kWh per page) [18]. Thus, we consider 0.004 kWh as the per-request server energy consumption for our conversation task. The PUE, WUE, and EWIF are the same as those used for estimating the training water consumption.

There is a slightly newer paper (Oct 2023) that directly measured power usage on a Llama 65B (on V100/A100 hardware) that showed a 14X better efficiency. [2] Ethan Mollick linked to it recently and got me curious since I've recently been running my own inference (performance) testing and it'd be easy enough to just calculate power usage. My results [3] on the latest stable vLLM from last week on a standard H100 node w/ Llama 3.3 70B FP8 was almost a 10X better token/joule than the 2023 V100/A100 testing, which seems about right to me. This is without fancy look-ahead, speculative decode, prefix caching taken into account, just raw token generation. This is 120X more efficient than the commonly cited "ChatGPT" numbers and 250X more efficient than the Llama-3-70B numbers cited in the latest version (v4, 2025-01-15) of that same paper.

For those interested in a full analysis/table with all the citations (including my full testing results) see this o1 chat that calculated the relative efficiency differences and made a nice results table for me: https://chatgpt.com/share/678b55bb-336c-8012-97cc-b94f70919daa

(It's worth point out that that used 45s of TTC, which is a point that is not lost on me!)

[1] https://arxiv.org/abs/2304.03271

[2] https://arxiv.org/abs/2310.03003

[3] https://gist.github.com/lhl/bf81a9c7dfc4244c974335e1605dcf22



https://fediverse.randomfoo.net/notice/AqCTDwXFsX7aMom4y8

So, here's a in interesting independent analysis by Andy Masley: [Using ChatGPT is not bad for the environment](https://andymasley.substack.com/p/individual-ai-use-is-not-bad-for)

The general point is that in terms of relative lifestyle choices, ChatGPT (LLM) usage is largely irrelevant to one's environmental footprint. One interesting thing that is probably worth nothing that the ChatGPT water/power numbers cited (the one that is most widely cited in these discussions) comes from an April 2023 paper ([Li et al, arXiv:2304.03271](https://arxiv.org/abs/2304.03271)) that estimates water/power usage based off of GPT-3 (175B dense model) numbers published from OpenAI's original *GPT-3 2021 paper*. From Section 3.3.2 Inference:

> As a representative usage scenario for an LLM, we consider a conversation task, which typically includes a CPU-intensive prompt phase that processes the user’s input (a.k.a., prompt) and a memory-intensive token phase that produces outputs [37]. More specifically, we consider a medium-sized request, each with approximately ≤800 words of input and 150 – 300 words of output [37]. The official estimate shows that GPT-3 consumes an order of 0.4 kWh electricity to generate 100 pages of content (e.g., roughly 0.004 kWh per page) [18]. Thus, we consider 0.004 kWh as the per-request server energy consumption for our conversation task. The PUE, WUE, and EWIF are the same as those used for estimating the training water consumption.

There is a slightly newer paper (Oct 2023, [Samsi et al, arXiv:2310.03003](https://arxiv.org/abs/2310.03003)) that directly measured power usage on a Llama 65B (on V100/A100 hardware) that showed a 14X better efficiency than the Li et al estimates. [Ethan Mollick linked to that recently](https://bsky.app/profile/emollick.bsky.social/post/3lflpodl3bc2x) and got me curious since I've recently been running my own inference (performance) testing and it'd be easy enough to just calculate power usage. [My results](https://gist.github.com/lhl/bf81a9c7dfc4244c974335e1605dcf22) on the latest stable vLLM from last week on a standard H100 node w/ Llama 3.3 70B FP8 was almost a 10X better token/joule than the 2023 V100/A100 testing, which seems about right to me (in terms of software/hardware efficiency gains). This is without fancy look-ahead, speculative decode, prefix caching taken into account, just raw token generation.

This is also 120X more efficient than the commonly cited "ChatGPT" numbers and 250X more efficient than the Llama-3-70B numbers cited in the latest version (v4, 2025-01-15) of the Li et al paper.

For those interested in a full analysis/table with all the citations (including my full testing results) see this o1 chat that calculated the relative efficiency differences and made a nice results table for me: https://chatgpt.com/share/678b55bb-336c-8012-97cc-b94f70919daa

(It's worth point out that that used 45s of test time compute, which is a point that is not lost on me!)


https://bsky.app/profile/lhl.bsky.social/post/3lfytnob3x22l


# Feb 2025
New analysis, directionally closer to my test results:
- https://engineeringprompts.substack.com/p/ai-energy-use
- https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use

Of course we can use DeepSeek-V3/R1 as a new benchmark for GPT-4 class.
- https://dstack.ai/blog/h200-mi300x-deepskeek-benchmark/
- https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html
- https://github.com/deepseek-ai/open-infra-index?tab=readme-ov-file#day-6---one-more-thing-deepseek-v3r1-inference-system-overview
 73.7k/14.8k input/output tokens per second per H800 node
 - 10KWh

The DeepSeek # s match mine pretty closely
https://claude.ai/chat/88c16990-50a2-412c-94e1-ea71935e8b19
https://aistudio.google.com/prompts/1kEQLqdGF2SF9B4xAvkKq_fruHWxFhbnd
https://chatgpt.com/c/67e9a77d-54e4-8012-89c4-0c9ba9e76c18

You could increase dense models w/ SD like EAGLE3

https://blog.kyleggiero.me/Image-generators-energy-usage/


# 2025-04-30 Integrated Analysis

Most recent article:
https://andymasley.substack.com/p/a-cheat-sheet-for-conversations-about

Previously:
https://andymasley.substack.com/p/individual-ai-use-is-not-bad-for


via Gemini 2.5 Pro:

**LLM Power Usage: Integrated Summary and Report**

This report synthesizes information from academic papers, empirical testing, and expert analyses to provide an overview of Large Language Model (LLM) power consumption during inference, highlighting trends in efficiency improvements and contextualizing energy use.

**Source Summaries and Citations:**

1. **Li, P., Yang, J., Islam, M. A., & Ren, S. (2023). Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models. arXiv:2304.03271 [cs.LG]. (Accepted by Communications of the ACM).**
    
    - **Summary:** This paper primarily focuses on the water footprint of AI but derives energy estimates as a basis. It cites an older "official estimate" for GPT-3 (a 175B dense model on pre-2023 hardware) suggesting ~0.004 kWh per "medium request" (~1 page/~300 words output). It also provides contrasting, higher estimates for newer models like Llama-3-70B (~0.010 kWh/request) and Falcon-180B (~0.016 kWh/request) on modern H100 hardware when considering enterprise Service Level Objectives (SLOs) and overheads, suggesting real-world deployment constraints can increase energy per request significantly compared to raw potential.
        
    - **Citation:** (Li et al., 2023)
        
2. **Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergeron, W., Kepner, J., Tiwari, D., & Gadepally, V. (2023). From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference. arXiv:2310.03003 [cs.CL].**
    
    - **Summary:** This study empirically benchmarks the energy cost of LLM inference, specifically using LLaMA models (up to 65B parameters) on NVIDIA V100 and A100 GPUs. It measures the energy per decoded token under various conditions (batch size, sharding, sequence length), finding it to be around 3-4 Joules per output token for LLaMA-65B with longer generation lengths. This provides a more direct measurement on slightly older hardware compared to Li et al.'s GPT-3 estimate.
        
    - **Citation:** (Samsi et al., 2023)
        
3. **Lin, L. H. (2025). Internal User Test: Llama3-70B Inference Efficiency on H100. Gist: bf81a9c7dfc4244c974335e1605dcf22.**
    
    - **Summary:** This personal testing measured the inference performance and power consumption of a Llama3-70B Instruct model on a modern 8xH100 node (AWS p5) using the vLLM framework with FP8 quantization. At a high concurrency (batch size 128), the measured energy cost was approximately 0.39 Joules per total token (input + output), demonstrating significant efficiency gains achieved through modern hardware (H100), optimized software (vLLM), and quantization techniques (FP8). The author notes further potential gains from MoE models, speculative decoding, and prefix caching.
        
    - **Citation:** (Lin, 2025)
        
4. **Masley, A. (2025, Jan 14 & Apr 28). Using ChatGPT is not bad for the environment & A cheat sheet for why using ChatGPT is not bad for the environment. Substack.**
    
    - **Summary:** These posts argue against individual user guilt regarding chatbot energy/water consumption. Using a baseline estimate of ~3 Wh per query (which translates to roughly 10-30 J/token depending on query length assumptions), Masley contextualizes this cost as negligible compared to everyday activities (driving, streaming, appliances) and total household energy use. Crucially, he estimates that consumer-facing chatbots like ChatGPT account for only ~3% of total AI energy consumption, with the vast majority used by recommender systems, enterprise analytics, search algorithms, computer vision, etc. He also contextualizes water use relative to energy generation and agriculture.
        
    - **Citation:** (Masley, 2025)
        
5. **Salathé, M. (2025, Mar 29). AI Energy Use in Everyday Terms. Engineering Prompts Substack.**
    
    - **Summary:** This analysis aims to provide easily understandable comparisons for chatbot energy use. It uses a lower estimate of ~0.2 Wh per interaction for a GPT-4-like model. Based on this, a year of heavy usage (100 interactions/day) equates to ~7.2 kWh, less energy than driving a gasoline car 10 km, taking five hot showers, or running two hot baths. It reinforces the perspective that typical chatbot usage has a low individual energy footprint.
        
    - **Citation:** (Salathé, 2025)
        
6. **You, J. (2025, Feb 07). How much energy does ChatGPT use?. Gradient Updates (Epoch AI).**
    
    - **Summary:** This post re-evaluates the commonly cited 3 Wh/query figure. Using updated assumptions for modern hardware (H100), model efficiency (GPT-4o, likely MoE), typical token counts (~500 output tokens), and GPU utilization (~10% compute, ~70% power), it estimates a typical GPT-4o query consumes closer to ~0.3 Wh. This is 10x lower than the older estimate. It highlights that very long input contexts significantly increase energy cost (up to ~40 Wh for 100k tokens) but notes this scaling can likely be improved. It reinforces the low marginal cost for typical queries compared to overall energy use but acknowledges the significant projected growth in total AI energy demand.
        
    - **Citation:** (You, 2025)
        

**Integrated Summary:**

The energy consumption of LLM inference has been a topic of significant discussion, but estimates and real-world performance have evolved rapidly. Early estimates, such as the one derived for GPT-3 by Li et al. (2023) based on 2021 data (~48 J/token), suggested substantial energy costs. However, these figures represented older, dense models on less efficient hardware.

Empirical studies like Samsi et al. (2023) on LLaMA-65B with A100/V100 GPUs already showed significant improvement, measuring around 3.5 J/token. More recent testing by Lin (2025) using Llama3-70B on H100 GPUs with modern optimizations like vLLM and FP8 quantization demonstrated a dramatic further reduction to approximately 0.39 J/token under high load. This ~120x improvement over the baseline GPT-3 estimate highlights the impact of advancements in hardware (H100 vs older generations), software (optimized inference engines like vLLM), and techniques like quantization.

Furthermore, the trend towards Mixture-of-Experts (MoE) models, as noted by Lin (2025), offers a path to maintain or improve performance with significantly fewer activated parameters (e.g., 17-22B MoE performing like a 70B dense model), directly translating to 3-4x lower computational cost (FLOPs) and thus lower energy per token for equivalent quality output.

Analyses by You (2025) and Salathé (2025) provide updated estimates for current models like GPT-4o, suggesting per-query costs are likely in the range of 0.2-0.3 Wh for typical interactions, significantly lower than older widely cited figures (like 3 Wh/query). This translates to roughly 2-3 J/token assuming ~300-500 tokens per interaction, aligning better with the empirical findings of Samsi et al. and showing the path towards the highly optimized results from Lin. However, You (2025) cautions that very long context windows drastically increase energy consumption per query due to the computational cost of processing the input attention, though algorithmic improvements are likely.

It is also crucial to contextualize these numbers. Masley (2025), Salathé (2025), and You (2025) emphasize that the energy cost of an individual chatbot query is extremely small compared to everyday activities and overall household energy consumption. Masley (2025) further argues that consumer chatbots represent only a small fraction (~3%) of the rapidly growing total energy demand attributed to AI, which is dominated by enterprise applications, recommendation engines, and other backend AI systems.

Interestingly, Li et al. (2023) also provide an estimate for Llama-3-70B on H100s under enterprise SLO conditions (~120 J/token or ~0.01 kWh/request), which is significantly higher than both their GPT-3 estimate and the empirical results from Samsi et al. and Lin. This highlights that deployment constraints, overheads, and potentially lower utilization or different optimization targets in specific enterprise scenarios can lead to much higher energy consumption per request compared to optimized benchmark conditions.

Water usage, primarily discussed by Li et al. (2023), is closely linked to energy consumption via data center cooling (direct use) and electricity generation (indirect use). Masley (2025) provides context, noting the vast amounts of water used in energy generation overall and in other sectors like agriculture.

**Comparison Table: LLM Inference Energy Estimates**

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Source & Year|Model & Setup|Est. Joules/Token|Relative Efficiency<br>(vs. Li GPT-3)|Est. kWh per 300-Token Output*|Notes|
|Li et al. (2023)|GPT-3 (175B dense), pre-2023 DC Hardware|~48 J|**1× (Baseline)**|~0.004 kWh|Derived from cited OpenAI figure (0.4 kWh/100 pages); Represents older tech.|
|Li et al. (2023)|Llama-3-70B, Enterprise DGX H100 w/ SLOs & overhead|~120 J|~0.4×|~0.010 kWh|Estimate including real-world constraints & overheads; Higher than baseline despite newer hardware.|
|Samsi et al. (2023)|LLaMA-65B, Multi-GPU V100/A100 Cluster|~3.5 J|~14×|~0.00029 kWh|Empirical measurement on previous-gen GPUs; Includes sharding overhead.|
|Salathé (2025) /<br>You (2025)|GPT-4o (likely MoE), H100 GPU|~2.2 - 2.4 J|~20-22×|~0.00018 - 0.00020 kWh|Implied from 0.2-0.3 Wh/query estimates assuming ~300-500 tokens/query; Reflects modern efficiency.|
|**Lin (2025)**|**Llama3-70B, 8×H100 (AWS p5), vLLM, FP8 Quant, Concurrency=128**|**~0.39 J**|**~123×**|**~0.000033 kWh**|**Empirical test on latest hardware/software; High throughput optimization. J/token includes input.**|
|Masley (2025)|Generic Chatbot (Implied)|~10-36 J (Est.)|~1.3-4.8×|~0.0008 - 0.003 kWh|Implied from 3 Wh/query baseline assuming 250-1000 tokens/query; Used for contextualization.|

*Note on kWh/300 Tokens: Calculated as (Joules/Token) * 300 / 3,600,000. This provides a rough comparison point based on Li et al.'s "medium request" size. For Lin (2025), J/token includes input tokens, making this slightly overestimate output-only energy. For Wh/query sources, token count per query is an assumption.*  
*Note on Lin (2025) J/token: The reported throughput is total tokens (input+output). The 0.39 J/token reflects this total throughput against total power. Energy per output token might be slightly higher depending on the input/output ratio of the benchmark dataset.

**Key Takeaways & Discussion:**

1. **Massive Efficiency Gains:** There has been an improvement of over two orders of magnitude (~120x) in energy efficiency (Joules per token) for LLM inference from the early GPT-3 era estimates to current optimized setups on H100 hardware (Lin, 2025).
    
2. **Drivers of Efficiency:** These gains stem from multiple factors:
    
    - **Hardware:** Newer GPUs (H100, Blackwell) are significantly more performant and energy-efficient per FLOP than older generations (V100/A100).
        
    - **Software:** Optimized inference frameworks (like vLLM, TensorRT-LLM) improve GPU utilization and reduce overhead.
        
    - **Quantization:** Techniques like FP8 reduce memory bandwidth requirements and allow faster computation with minimal quality loss.
        
    - **Model Architecture:** MoE models achieve high performance with fewer active parameters, directly reducing computational load.
        
3. **Low Marginal Cost for Typical Use:** For typical chatbot interactions (short-to-medium length), the energy cost per query on modern systems is very low (estimated ~0.2-0.3 Wh or ~2-3 J/token by You/Salathé, demonstrated even lower at <0.4 J/token under high load by Lin), making the individual environmental footprint negligible compared to many routine activities.
    
4. **Context Length Matters:** Processing very long input sequences remains computationally intensive and significantly increases the energy cost per query (You, 2025), although algorithmic improvements are mitigating this.
    
5. **Deployment Matters:** Real-world energy consumption can be higher than optimized benchmarks due to factors like SLOs, lower average utilization, and system overheads (Li et al., 2025 - Llama/H100 estimate).
    
6. **Chatbots vs. Total AI Energy:** While individual query costs are low, the aggregate energy demand for AI is projected to grow significantly. However, consumer-facing chatbots currently constitute a small fraction (~3% estimated by Masley, 2025) of this total AI energy consumption. The bulk is driven by enterprise AI, recommenders, search, etc.
    

**Conclusion:**

The energy efficiency of LLM inference has improved dramatically due to advances in hardware, software optimization techniques, and model architectures like MoE. For typical usage patterns, the energy consumption per query on modern systems is minimal. While the rapidly growing overall demand for AI computation presents a significant energy challenge that requires ongoing attention to sustainable practices and efficient deployment, the narrative surrounding the high energy cost of individual chatbot interactions often relies on outdated figures or lacks crucial context regarding the scale of other energy uses and the small proportion of total AI energy consumed by these applications. Future efficiency gains are expected, but the overall energy footprint of AI will depend heavily on the scale of deployment across all sectors.


https://claude.ai/chat/5596059b-1961-48e7-aeec-72de4680c66a

# Integrated Summary and Power Usage Report for LLM Inference

## Report Sources and Key Findings

This report integrates three primary sources on LLM power consumption:

1. Li et al. (April 2023): "Making AI Less 'Thirsty'" - Highlighted the significant water and power footprint of AI models, estimating GPT-3 training directly evaporated 700,000 liters of freshwater and calculating power usage for inference.
2. Samsi et al. (October 2023): "From Words to Watts" - Conducted empirical measurements of LLaMA-65B inference on V100/A100 GPUs, finding approximately 3-4 joules per generated token.
3. Lin (2025): User testing with Llama-3.3-70B on modern H100 hardware with FP8 quantization and high concurrency (batch size 128), measuring dramatically improved efficiency at approximately 0.39 joules per token.

## Comparative Analysis of Power Efficiency

The following table presents a comparative analysis of power efficiency across the three sources, with the Li et al. (2023) GPT-3 baseline set to 1× efficiency:

|Source & Year|Model & Hardware|Joules/Token|Relative Efficiency vs. Li (GPT-3)|Est. kWh/300 Tokens|
|---|---|---|---|---|
|Li et al. 2023 (GPT-3 baseline)|GPT-3, older DC|~48 J|1× (baseline)|~0.004 kWh|
|Li et al. 2023 (H100 scenario)|Llama3-70B on enterprise DGX H100 w/ overhead|~120 J|0.4×|~0.010 kWh|
|Samsi et al. 2023|LLaMA-65B on V100/A100 HPC|~3.5 J|~14×|~0.00003 kWh|
|Lin, 2025|Llama3-70B on 8×H100, FP8, concurrency=128|~0.39 J|~120×|~0.000033 kWh|

## Key Factors Affecting Power Efficiency

The dramatic efficiency variations (spanning more than two orders of magnitude) can be attributed to several key factors:

1. **Hardware Generation**: Modern GPUs (A100/H100) and optimized inference engines significantly reduce energy consumption compared to older hardware.
2. **System Utilization and Concurrency**: Higher batch sizes and concurrency allow better amortization of overhead, keeping GPUs running closer to peak FLOP efficiency.
3. **Quantization & Optimization**: Moving from FP16 to FP8, using speculative decoding, or advanced partitioning strategies (like vLLM's token scheduling) yields more tokens per second for the same power draw.
4. **Overhead Inclusion**: Studies vary in whether they include significant CPU, networking, or idle overhead in their calculations.

## Model Architecture Impact

Recent developments in model architecture have further improved efficiency. The most current models (as of 2025) include 17-22B MOE (Mixture of Experts) models that perform better than earlier 70B dense models. These MOE architectures achieve the same quality output with 3-4× fewer activations, contributing significantly to power efficiency gains.

## Conclusion

The data demonstrates remarkable progress in LLM inference efficiency from 2023 to 2025. While early estimates suggested high power requirements for LLM inference, modern hardware, quantization techniques, and architectural improvements have dramatically reduced power consumption - achieving up to 120× better efficiency compared to 2023 baseline measurements for GPT-3 inference.

These improvements suggest that concerns about LLM inference power consumption should be reassessed in light of rapid technological advances. The most current systems (represented by Lin's 2025 testing) demonstrate that high-performance inference can be achieved with substantially lower energy requirements than previously estimated.