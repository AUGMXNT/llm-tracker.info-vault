Most inferencing packages have their own REST API, but having an OpenAI compatible API is useful for using a variety of clients, or to be able to easily switch between providers.

Are there any with full support: assistants, function calling, chat completions?


Recent:
* https://github.com/1b5d/llm-api
* https://github.com/janhq/nitro
* https://github.com/BerriAI/litellm

[https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

- General llama.cpp Python wrapper
- Has a good enough OpenAI API server as well

[https://github.com/c0sogi/llama-api](https://github.com/c0sogi/llama-api)

- Python
- llama.cpp and exllama compatibility
- on-demand model loading

[https://localai.io/](https://localai.io/)
https://github.com/mudler/LocalAI

- Docker-focused 
    - [https://localai.io/basics/getting\_started/](https://localai.io/basics/getting_started/)
- Runs its own llama.cpp

[https://github.com/hyperonym/basaran](https://github.com/hyperonym/basaran)

- Docker-focused
- For HF models

[https://github.com/xorbitsai/inference](https://github.com/xorbitsai/inference)

- Python
- Uses various backends (CTransformers, llama-cpp-python, not well documented)

Some clients provide an OpenAI API compatibility layer:

[https://github.com/oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

- Uses conda

[https://lmstudio.ai/](https://lmstudio.ai/)

- Mac or Windows GUI App
- But w/ an OpenAI API layer
- Not open source but free to use