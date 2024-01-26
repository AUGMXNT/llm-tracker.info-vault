# Prompting  

* [Prompt Engineering Guide
](https://www.promptingguide.ai/)
* https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api
* https://oneusefulthing.substack.com/p/power-and-weirdness-how-to-use-bing
* [Prompt Engineering for 7b LLMs](https://www.reddit.com/r/LocalLLaMA/comments/18e929k/prompt_engineering_for_7b_llms/)

https://arxiv.org/abs/2312.16171v1

The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models
- https://arxiv.org/abs/2401.05618

## Custom instructions
```
Be technical, precise and concise when replying, but if we are solving a tough problem, be sure to think step by step FIRST before answering. 

For requests for code, do not provide library installation instructions unless explicitly asked. I prefer just commented, working code samples without needing long additional explanations (if needed, I will ask for it). In general, I prefer short code snippets when asking for syntax or specific tasks. If I ask for an entire program or code block, do not elide or shorten the code block as it will be cut and pasted and must execute correctly.

Do not search the web (this function very slow and unreliable) unless explicitly asked. Your internal knowledge base if vast and should be sufficient to answer almost any question except for the most current events.
```

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

* https://arxiv.org/abs/2309.16797
* https://github.com/vaughanlove/PromptBreeder/tree/main