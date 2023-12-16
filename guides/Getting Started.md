[Large Language Models](https://en.wikipedia.org/wiki/Large_language_model) (LLM) are a type of [generative AI](https://en.wikipedia.org/wiki/Generative_artificial_intelligence) that power chatbot systems like [ChatGPT](https://openai.com/blog/chatgpt).

You can try many of these for free if you've never tried one (although most of this site is aimed at those with more familiarity with these types of systems).

All of these have free access, although many may require user registration, and remember, all data is being sent online to third parties so don't say anything you'd want to keep very private:

- Chat Services 
  - [OpenAI ChatGPT](https://chat.openai.com/) - the free version uses 3.5, and is not as smart, but good for dipping your toe in. 
    - [chat.getgpt.world](https://chat.getgpt.world/) - HK (Alibaba Cloud) based server w/ gpt-3.5-turbo access
  - [Anthropic Claude](https://claude.ai/) - this is an interesting alternative, has super long context window (short-term memory) so can interact w/ very long files for summarization, etc.
  - [HuggingFace Chat](https://huggingface.co/chat/) - chat for free w/ some of the best open models
  - [ChatNBX](https://chat.nbox.ai/) - talk to some of the best open models and compare
  - [Perplexity.ai](https://www.perplexity.ai/) - the best free service w/ web search capabilities 
    - [Perplexity Labs](https://labs.perplexity.ai/) - hosts some open models to try as well
-  More 
  - [Bing Chat](https://www.bing.com/chat) (requires Microsoft Edge lol) - also has web capabilities 
    - [Bing Chat History extension](https://chrome.google.com/webstore/detail/bing-chat-history/hjhpahdglfjddhhecnjlhckicdpcdhpg) ([Github](https://github.com/benf2004/Bing-Chat-History))  

  - [Google Bard](https://bard.google.com/)
  - [You.com](https://you.com/) - 100 free AI chat searches
  - [DeepAI Chat](https://deepai.org/chat)  

  - [Hugging Face Spaces](https://huggingface.co/spaces)  

- Comparisons 
  - [Vercel SDK Playground](https://sdk.vercel.ai/) 
  - [nat.dev Playground](https://nat.dev/)
  - [Anyscale Aviary Explorer](https://aviary.anyscale.com/)
  - [h2oGPT](https://gpt.h2o.ai/)

You can also run LLMs locally on most modern computers (although larger models require strong GPUs).

The easiest (virtually one-click, no command line futzing) way to test out some models is with [LM Studio](https://lmstudio.ai/) (Windows, Mac). Other alternatives include:

- [Nomic's GPT4All](https://gpt4all.io/) ([Github](https://github.com/nomic-ai/gpt4all)) - Windows, Mac, Linux  

- [Ollama](https://ollama.ai/) ([Github](https://github.com/jmorganca/ollama)) - Mac

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