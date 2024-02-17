[Large Language Models](https://en.wikipedia.org/wiki/Large_language_model) (LLM) are a type of [generative AI](https://en.wikipedia.org/wiki/Generative_artificial_intelligence) that power chatbot systems like [ChatGPT](https://openai.com/blog/chatgpt).

You can try many of these for free if you've never tried one (although most of this site is aimed at those with more familiarity with these types of systems).

All of these have free access, although many may require user registration, and remember, all data is being sent online to third parties so don't say anything you'd want to keep very private.

# Hosted Services
## Proprietary Model Chat
- [OpenAI ChatGPT](https://chat.openai.com/) - the free version uses 3.5, and is not as smart, but good for dipping your toe in. 
- [Anthropic Claude](https://claude.ai/) - this is an interesting alternative, has super long context window (short-term memory) so can interact w/ very long files for summarization, etc.
- [Google Bard](https://bard.google.com/) - now running Gemini Pro, which is about ChatGPT 3.5 level
- [Perplexity.ai](https://www.perplexity.ai/) - the best free service w/ web search capabilities 
- [You.com](https://you.com/) - another search and chat provider
- [Phind](https://www.phind.com/) - Proprietary coding model ([fine-tuned off of Code Llama 34B](https://news.ycombinator.com/item?id=38088538))

## Open Model Chat
- [HuggingFace Chat](https://huggingface.co/chat/) - chat for free w/ some of the best open models
- [ChatNBX](https://chat.nbox.ai/) - talk to some of the best open models and compare
- [Perplexity Labs](https://labs.perplexity.ai/) - hosts some open models to try as well
- [DeepSeek Coder](https://chat.deepseek.com/coder) - an Open Source coding model developed by a Chinese company, https://deepseekcoder.github.io/ - currently one of the strongest coding models
## Open Model Testing
- [Hugging Face Spaces](https://huggingface.co/spaces) - there's a huge amount of "spaces" with hosted models for testing. These are typically running on small/slow instances with queues so mainly good for the most simple testing
- [Vercel SDK Playground](https://sdk.vercel.ai/) - lets you easily A/B test a small selection of models
- [nat.dev Playground](https://nat.dev/) - a better selection of models


- Comparisons 
    - https://app.composo.ai/playground
    - [Anyscale Aviary Explorer](https://aviary.anyscale.com/)
    - [h2oGPT](https://gpt.h2o.ai/)


## Paid Open Models
This site tracks quality, throughput, price of various API hosting providers: https://artificialanalysis.ai/

https://openrouter.ai/
https://www.anyscale.com/endpoints
https://replicate.com/
https://www.together.ai/
https://www.fireworks.ai/
https://octo.ai/pricing/

# GPU Pricing
- https://computewatch.llm-utils.org/
- https://gpus.llm-utils.org/alternative-gpu-clouds/
- H100s: https://gpus.llm-utils.org/h100-gpu-cloud-availability-and-pricing/
	- https://gpus.llm-utils.org/nvidia-h100-gpus-supply-and-demand/

H100's
- https://www.coreweave.com/products/hgx-h100
- 



You can also run LLMs locally on most modern computers (although larger models require strong GPUs).

The easiest (virtually one-click, no command line futzing) way to test out some models is with [LM Studio](https://lmstudio.ai/) (Linux, Mac, Windows). Other alternatives include:

- [Nomic's GPT4All](https://gpt4all.io/) ([Github](https://github.com/nomic-ai/gpt4all)) - Windows, Mac, Linux
- [Ollama](https://ollama.ai/) ([Github](https://github.com/jmorganca/ollama)) - Mac, Linux
- https://jan.ai/

If you are more technical:

- [oobabooga](https://github.com/oobabooga/text-generation-webui) - think of it as the automatic1111 of LLMs
- [koboldcpp](https://github.com/LostRuins/koboldcpp) - more oriented for character/roleplay, see also [SillyTavern](https://sillytavernai.com/)
- [openplayground](https://github.com/nat/openplayground)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [exllama](https://github.com/turboderp/exllama)

Most of the guides in this HOWTO section will assume:

- On a UNIXy platform (Linux, macOS, WSL)
- [Familiarity w/ the command line](https://mostlyobvious.org/?link=/Reference%2FSoftware%2FDevelopment%2FLearn%20to%20Code%2FLearn%20to%20Command%20Line), comfortable installing apps etc
- Some [familiarity with Python, git](https://mostlyobvious.org/?link=/Reference%2FSoftware%2FDevelopment%2FLearn%20to%20Code%2FLearn%20to%20Program)

Global Recommendations:

[Install Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) and create a new conda environment anytime you are installing a package which have many dependencies. eg, create a separate `exllama` and `autogptq` or `lm-eval` environment.

# Other Resources

- [LocalLLaMA Wiki](https://www.reddit.com/r/LocalLLaMA/wiki/index/)
- [https://github.com/underlines/awesome-marketing-datascience/blob/master/README.md](https://github.com/underlines/awesome-marketing-datascience/blob/master/README.md)



https://github.com/LykosAI/StabilityMatrix
https://github.com/comfyanonymous/ComfyUI