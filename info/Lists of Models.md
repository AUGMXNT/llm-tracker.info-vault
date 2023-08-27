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

Note, llama.cpp just updated to GGUF format, so you should try to find or convert to one of those (or TheBloke may have converted by the time you read this). It should make using non-llama and extended context window models much easier moving forward. GPTQs remain the same.

My general preference for bang/bit quants is either 4-bit 32g actorder=True GPTQ w/ ExLlama or q4_K_M GGML w/ llama.cpp.

Last updated: 2023-08-23

- Current best local model (any size) 
  - All the top ranked models are currently llama2-70b fine tunes. While I didn't try them all, I recently tested most of the top leaderboard models and Pankaj Mathur's [Orca Mini V3](https://huggingface.co/psmathur/orca_mini_v3_70b) did the best at instruction following for a [basic text manipulation task](https://github.com/AUGMXNT/transcribe#readme).
- Current best local model for 24GB GPU (eg, 3090, 4090) 
  - While the llama2-34b has yet to be released, new llama2-13b models have largely overtaken llama-30b in the leaderboards. Due to their extended (4K vs 2K token) native context window, for most usage, I think the llama2-13bs should probably be preferred. I don't have strong opinions on the "best" models atm, but I'd give a few a try: 
    - [Open-Orca/OpenOrca-Platypus2-13B](https://huggingface.co/Open-Orca/OpenOrca-Platypus2-13B) - for instruction following, tasks
    - [MythoMax-L2-13b](https://huggingface.co/Gryphe/MythoMax-L2-13b) - for anything creative or chatting w/ personas
    - [totally-not-an-llm/EverythingLM-13b-V2-16k](https://huggingface.co/totally-not-an-llm/EverythingLM-13b-V2-16k) - this is an extended context model that's worth playing around with. The q4_K_M GGML should just barely fit into exactly 24GB of VRAM. Be sure to use `-c 16384 --rope-freq-base 10000 --rope-freq-scale 0.25` to get it to inference sensically.
- Current best local model for 16GB GPU or Apple Silicon Mac 
  - You can try any llama2-13b fine tune  

- Current best local coding model 
  - [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) (maybe [LoupGarou/WizardCoder-Guanaco-15B-V1.1](https://huggingface.co/LoupGarou/WizardCoder-Guanaco-15B-V1.1), untested)
  - There have also been [recent developments on the SQL generation front](https://www.reddit.com/r/LocalLLaMA/comments/15y6pfm/sqlcoder_new_15b_oss_llm_claims_better/)