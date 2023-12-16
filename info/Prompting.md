# Prompting  

* [Prompt Engineering Guide
](https://www.promptingguide.ai/)
* https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api
* https://oneusefulthing.substack.com/p/power-and-weirdness-how-to-use-bing
* [Prompt Engineering for 7b LLMs](https://www.reddit.com/r/LocalLLaMA/comments/18e929k/prompt_engineering_for_7b_llms/)

## Prompt Format
Most instruct/chat fine tunes use their own formatting (which can wildly differ). These can usually be dug out of model card/repos/localllama discussion/discords, but the best single reference I've found is oobabooga's Instruction Templates: 
* [https://github.com/oobabooga/text-generation-webui/tree/main/instruction-templates](https://github.com/oobabooga/text-generation-webui/tree/main/instruction-templates)

If using HF, see also the documentation on [chat templating](https://huggingface.co/docs/transformers/main/chat_templating)

## Jailbreaks
Most of these will probably work on open models: https://www.jailbreakchat.com/

See:
* https://twitter.com/abacaj/status/1734450551381754063

## System Prompts
* https://github.com/spdustin/ChatGPT-AutoExpert/blob/main/System%20Prompts.md
* https://news.ycombinator.com/item?id=37879077