We will eventually be hosting an API and list of LLMs.  In the meantime:

- [Open LLMs](https://github.com/eugeneyan/open-llms)
  - The GitHub repo is the currently most actively maintained list of open models (that can be used commercially) along with some other resources
- [awesome-marketing-datascience/Open LLM Models List](https://github.com/underlines/awesome-marketing-datascience/blob/master/llm-model-list.md)  

  - Also available [as a Google Sheet](https://docs.google.com/spreadsheets/d/1PtrPwDV8Wcdhzh-N_Siaofc2R6TImebnFvv0GuCCzdo/edit?usp=sharing), this is a curated list of fine tunes and quantized models
- [CRFM Ecosystem Graphs Table](https://crfm.stanford.edu/ecosystem-graphs/index.html?mode=table)
  - Maintained by CRFM, this tracks a large number of models (beyond LLMs) along with metadata
- [Viktor Garske's AI / ML / LLM / Transformer Models Timeline and List](https://ai.v-gar.de/ml/transformer/timeline/)  

  - A timeline of select papers and models
- [lhl's LLM Worksheet](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit?usp=sharing)
  - A Google Sheet that tracks foundational models, as well as some fine tunes, datasets, and evals
- [Reddit r/LocalLLaMA "New Model" flair](https://www.reddit.com/r/LocalLLaMA/?f=flair_name%3A%22New%20Model%22)
  - This is a good place if you're looking for the latest fine tunes of open foundational models
- [Hugging Face: Text Generation Models (sorted by recently modified)](https://huggingface.co/models?pipeline_tag=text-generation&sort=modified)
  - This is where almost every public fine-tune, quantize, as well as foundational model lives, so it's a bit much and not where to start, but where you might end up.
- [2023 LifeArchitect.ai data](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs/edit#gid=1158069878)
  - A Google Sheet w/ mixed list of various models, datasets, and lots of random data
- [Constellation: An Atlas of 15,000 Large Language Models](https://llmconstellation.olafblitz.repl.co/)

See [Evals](https://llm-tracker.info/books/evals/page/list-of-evals) for potentially more models and how they compare.

### Which model to use?  
There is a constant stream of new model releases (both fine tunes and foundational models) so any list will be inevitably out of date w/o constant attention. Instead of a list, here's how to pick a model:

- If you're just getting started, just use the [Getting Started guide](https://llm-tracker.info/books/howto-guides/page/getting-started) and seeing which ones you can test out. nbox and perplexity labs tend to cycle in the latest hot models.
- If you're looking to download a model, I'd compare [TheBloke's latest quantizes](https://huggingface.co/TheBloke?sort_models=modified#models) with your preferred [Eval for your task](https://llm-tracker.info/books/evals/page/list-of-evals).
- You can visit [r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/) and see the latest models that the community is talking about.