[Large Language Models](https://en.wikipedia.org/wiki/Large_language_model) (LLM) are a type of [generative AI](https://en.wikipedia.org/wiki/Generative_artificial_intelligence) that power chatbot systems like [ChatGPT](https://openai.com/blog/chatgpt).

You can try many of these for free if you've never tried one (although most of this site is aimed at those with more familiarity with these types of systems). All of these have free access, although many may require user registration:

- Chat Services 
  - [OpenAI ChatGPT](https://chat.openai.com/)
    - [chat.getgpt.world](https://chat.getgpt.world/) - HK (Alibaba Cloud) based server w/ gpt-3.5-turbo access
  - [Bing Chat](https://www.bing.com/chat) (requires Microsoft Edge lol)
  - [Google Bard](https://bard.google.com/)
  - [You.com](https://you.com/) - 100 free AI chat searches
  - [DeepAI Chat](https://deepai.org/chat)  

  - [Hugging Face Spaces](https://huggingface.co/spaces)
    - [Falcon-Chat demo](https://huggingface.co/spaces/HuggingFaceH4/falcon-chat) - for a real trip, take a look at this Wikipedia article, [Human rights in the United Arab Emirates](https://en.wikipedia.org/wiki/Human_rights_in_the_United_Arab_Emirates) and ask it questions of legal facts about the UAE. Falcon will consistently lie to you about it. 
- Comparisons 
  - [Vercel SDK Playground](https://sdk.vercel.ai/) 
  - [nat.dev Playground](https://nat.dev/)
  - [Anyscale Aviary Explorer](https://aviary.anyscale.com/)
  - [h2oGPT](https://gpt.h2o.ai/)

You can also run LLMs locally on most modern computers (although larger models require strong GPUs).

The easiest (virtually one-click, no command line futzing) way to test out some models is with [Nomic's GPT4All](https://gpt4all.io/) ([Github](https://github.com/nomic-ai/gpt4all)).

If you are more technical:

- [oobabooga](https://github.com/oobabooga/text-generation-webui)
- [koboldcpp](https://github.com/LostRuins/koboldcpp)
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