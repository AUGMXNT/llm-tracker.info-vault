While some inferencing engines have their own API as well, most have standardized on [OpenAI API](https://github.com/openai/openai-openapi) compatibility.
- llama.cpp - has `llama-server` which is compatible OOTB
	- If you want Python bindings: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- mlc-llm - run `mlc_llm server` for OpenAI server
- ExLlamaV2 - not OOTB but run [TabbyAPI](https://github.com/theroyallab/tabbyAPI) for compatibility
- vLLM - comes OOTB with `vllm serve`
- [LiteLLM](https://github.com/BerriAI/litellm) - useful as a single middleware layer for lots of model types

Most inferencing packages have their own REST API, but having an OpenAI compatible API is useful for using a variety of clients, or to be able to easily switch between providers.

Are there any with full support: assistants, function calling, chat completions?




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