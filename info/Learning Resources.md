# Getting Started
 If you're starting from nothing. Just go to Wikipedia and start reading:
* [https://en.wikipedia.org/wiki/Large_language_model](https://en.wikipedia.org/wiki/Large_language_model)
* [https://en.wikipedia.org/wiki/Foundation_models](https://en.wikipedia.org/wiki/Foundation_models)
* [https://en.wikipedia.org/wiki/Artificial_neural_network](https://en.wikipedia.org/wiki/Artificial_neural_network)
* [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)
* etc, just keep reading

You can use an LLM to help you summarize and better understand/query, although I would not use anything less than ChatGPT 4 or ChatGPT/with Web Browsing (or Bing Chat, which is also designed to do retrieval-augmented replies) to minimize hallucinations. This has the added benefit of getting better first hand experiences of what LLMs can do well (or poorly).

There are plenty of resource lists for research:
* [https://github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) - a good list of fundamental papers to read
* [https://github.com/Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide) - another good resource that is a little better organized IMO and supports a survey paper https://arxiv.org/abs/2304.13712 of LLMs and their applications
* [Understanding Large Language Models -- A Transformative Reading List, Sebastian Raschka](https://sebastianraschka.com/blog/2023/llm-reading-list.html) - rather than just a list of papers, it also has a short description of why the linked papers are important.
* [Anti-hype LLM reading list
](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e) - a nice condensed reading list

## Announcements
For a layperson wanting to learn more, I actually think that reading the various announcements on models (and using an LLM to interrogate parts you don't understand) are probably a decent way to get started. You might not understand everything, but they start give you the "flavor-text" so to speak of AI attributes, keywords, etc:

* OpenAI GPT-4
  * [https://openai.com/gpt-4](https://openai.com/gpt-4)
  * [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)
  * [https://cdn.openai.com/papers/gpt-4-system-card.pdf](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
* Meta LLaMA
  * [https://ai.facebook.com/blog/large-language-model-llama-meta-ai/](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
  * [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
  * [https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
* Cerebras-GPT
  * [https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
* MosaicML MPT-30B
  * [https://www.mosaicml.com/blog/mpt-30b](https://www.mosaicml.com/blog/mpt-30b)
* TII Falcon
  * [https://huggingface.co/blog/falcon](https://huggingface.co/blog/falcon)
* Salesforce XGen
  * [https://blog.salesforceairesearch.com/xgen/](https://blog.salesforceairesearch.com/xgen/)
* OpenOrca
  * [https://erichartford.com/openorca](https://erichartford.com/openorca)

Reading the announcements and model cards for models (and looking up what you don't understand) is a great way to get up to speed fast.

## Basics

### Overview
* https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Overview-of-Large-Language-Models-LLMs---VmlldzozODA3MzQz
* https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Transformer-Networks--VmlldzoyOTE2MjY1

### Foundational Models
* https://research.ibm.com/blog/what-are-foundation-models
* https://blogs.nvidia.com/blog/2023/03/13/what-are-foundation-models/

### Context Window
* https://www.linkedin.com/pulse/whats-context-window-anyway-caitie-doogan-phd/
* https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c
* https://blog.langchain.dev/auto-evaluation-of-anthropic-100k-context-window/

# Learn From Scratch
For programmers who don't know ML, it may be easier to learn by doing:
* [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
* Discussion: https://news.ycombinator.com/item?id=34726115
* [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
* https://www.youtube.com/watch?v=kCc8FmEb1nY
* https://github.com/karpathy/minGPT
* [From Transformer to LLM: Architecture, Training and Usage](https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/transformers)

# Structured Courses
* [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
* [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)

# Deep Dive Explanations
* https://jalammar.github.io/illustrated-transformer/
* [Attn: Illustrated Attention, Raimi Karim](https://archive.is/NSH37)
* Transformers Explained Visually, Ketan Doshi
* [Transformers Explained Visually (Part 1): Overview of Functionality](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
* [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)
* [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
* [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
* [LLM Parameter Counting](https://kipp.ly/blog/transformer-param-count/)
* A Conceptual Guide to Transformers
  * Part I: https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers
  * Part II: https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers-b70
  * Part III: https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers-024
  * Part IV: https://benlevinstein.substack.com/p/how-to-think-about-large-language
  * Part V: https://benlevinstein.substack.com/p/whats-going-on-under-the-hood-of

# Fine Tuning Guides
* https://erichartford.com/uncensored-models
* https://huggingface.co/blog/stackllama
* https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning
* https://lightning.ai/pages/community/tutorial/accelerating-llama-with-fabric-a-comprehensive-guide-to-training-and-fine-tuning-llama/
* https://github.com/hiyouga/LLaMA-Efficient-Tuning
* https://github.com/Lightning-AI/lit-llama/blob/main/howto/finetune_lora.md
* https://github.com/Lightning-AI/lit-llama/blob/main/howto/finetune_adapter.md
* https://github.com/zphang/minimal-llama
* https://github.com/OpenGVLab/LLaMA-Adapter
* https://github.com/zetavg/LLaMA-LoRA-Tuner
* https://github.com/artidoro/qlora


# Prompting
https://www.promptingguide.ai/
https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api


## Resource Lists
* https://www.reddit.com/r/LocalLLaMA/wiki/models/
* https://gist.github.com/rain-1/eebd5e5eb2784feecf450324e3341c8d
* https://github.com/KennethanCeyer/awesome-llm
* https://github.com/kasperjunge/LLM-Guide
* https://github.com/imaurer/awesome-decentralized-llm
* https://github.com/snehilsanyal/self_learning_llms


# Latest Research

## arXiv

[https://arxiv.org/list/cs.AI/recent](https://arxiv.org/list/cs.AI/recent)

[https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)

## AK's Daily Papers

- [https://huggingface.co/papers](https://huggingface.co/papers)
- [https://twitter.com/\_akhaliq](https://twitter.com/_akhaliq)

## Papers With Code

[https://paperswithcode.com/](https://paperswithcode.com/)

# Blogs

* [Ethan Mollick's Substack](https://www.oneusefulthing.org/)
* [https://lilianweng.github.io/](https://lilianweng.github.io/)
* [https://llm-utils.org/Home](https://llm-utils.org/Home)
* [https://yaofu.notion.site/Yao-Fu-s-Blog-b536c3d6912149a395931f1e871370db](https://yaofu.notion.site/Yao-Fu-s-Blog-b536c3d6912149a395931f1e871370db)
* [https://vinija.ai/](https://vinija.ai/)
* [https://kaiokendev.github.io/](https://kaiokendev.github.io/)


## Misc
* https://www.reddit.com/r/LocalLLaMA/comments/14le4ti/tree_of_thoughts_build_in_opensource_model/
* https://www.reddit.com/r/LocalLLaMA/comments/14fvht9/new_pruning_method_wanda_can_prune_llms_to_50/
* https://github.com/openlm-research/open_llama/issues/40
* https://github.com/openlm-research/open_llama/issues/63
https://github.com/openlm-research/open_llama/issues/65