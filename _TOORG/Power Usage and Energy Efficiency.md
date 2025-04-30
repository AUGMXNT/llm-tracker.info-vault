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