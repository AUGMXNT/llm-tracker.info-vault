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

See Evals for potentially more models and how they compare.

### lhl's opinionated list of local llms

May move this out if anyone else ever starts contributing and just give a recommended starting list. I don't do RP or require character writing/fiction. My main uses are for factual q&a, and ideally coding/tech support.

Last updated: 2023-07-24

- Current best local model (any size) 
  - [Stability AI FreeWilly2](https://huggingface.co/stabilityai/FreeWilly2) (llama2-70b fine tune) - unquantized will take 160-200GB of VRAM, 4-bit GPTQ on exllama in 41-48GB
- Current best local model for 24GB GPU (eg, 3090, 4090) 
  - There are a bunch of very strong llama-30b fine tunes. Expect this to be replaced by llama2-34b fine tunes when that is released...
  - Use exllama w/ GPTQ as you will want all the VRAM as you can get for context
  - [TheBloke/upstage-llama-30b-instruct-2048-GPTQ](https://huggingface.co/TheBloke/upstage-llama-30b-instruct-2048-GPTQ)  

  - [lilloukas/GPlatty-30B](https://huggingface.co/lilloukas/GPlatty-30B)
  - [lilloukas/Platypus-30B](https://huggingface.co/lilloukas/Platypus-30B)
  - For tool use 
    - [arielnlee/SuperPlatty-30B](https://huggingface.co/arielnlee/SuperPlatty-30B)
- Current best local model for 16GB GPU or Apple Silicon Mac 
  - [NousResearch/Nous-Hermes-Llama2-13b](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b) (GPTQ on GPU, GGML on Mac)
- Current best local coding model 
  - [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) (maybe [LoupGarou/WizardCoder-Guanaco-15B-V1.1](https://huggingface.co/LoupGarou/WizardCoder-Guanaco-15B-V1.1), untested)