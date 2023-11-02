# Prompting  

* https://www.promptingguide.ai/
* https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api
* https://oneusefulthing.substack.com/p/power-and-weirdness-how-to-use-bing

## Prompt Format
Most instruct/chat fine tunes use their own formatting (which can wildly differ). These can usually be dug out of model card/repos/localllama discussion/discords, but the best single reference I've found is oobabooga's Instruction Templates: 
* [https://github.com/oobabooga/text-generation-webui/tree/main/instruction-templates](https://github.com/oobabooga/text-generation-webui/tree/main/instruction-templates)

If using HF, see also the documentation on [chat templating](https://huggingface.co/docs/transformers/main/chat_templating)