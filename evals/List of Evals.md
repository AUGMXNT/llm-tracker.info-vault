2024-01: Towards the end of 2023, the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) has driven a trend of benchmark gaming that makes it (and other synthetic benchmarks) largely unuseful for testing model capabilities.

The current consensus is that [LMSys Chatbot Arena](https://chat.lmsys.org/?leaderboard) which allows users to compare responses and choose a winner (and does ELO style ranking) is the current gold standard for ranking general model performance. Obviously, this is has its own weaknesses (biased by audience and use case, colored by refusals or other considerations), but it's seems to be the best we've got atm.
* LMSys is also [collecting datasets from this chat data](https://huggingface.co/lmsys) which could be extremely useful for training
* u/DontPlanToEnd posted a [benchmark correlation analysis](https://www.reddit.com/r/LocalLLaMA/comments/18u0tu3/benchmarking_the_benchmarks_correlation_with/) on 2023-12-30 which showed MT-Bench (GPT-4 judged benchmark) had 0.89 correlation with Chatbot Arena, making it probably the 2nd best score. (MMLU has 0.85 correlation)
* u/WolframRavenwolf has been posting his own [LLM Comparison/Tests](https://www.reddit.com/user/WolframRavenwolf/submitted/) of new models which is pretty interesting - it tests in German (almost assuredly out of distribution) and focuses on instruction following, but is a good sanity check
# General
- [MosaicML Model Gauntlet](https://www.mosaicml.com/llm-evaluation) - 34 benchmarks in 6 categories
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
    - warning, their MMLU results are wrong, throwing off the whole ranking: [https://twitter.com/Francis\_YAO\_/status/1666833311279517696](https://twitter.com/Francis_YAO_/status/1666833311279517696)
- [LMSys Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard) - ELO style ranking
- [LLM-Leaderboard](https://llm-leaderboard.streamlit.app/)
- [Gotzmann LLM Score v2](https://docs.google.com/spreadsheets/d/1ikqqIaptv2P4_15Ytzro46YysCldKY7Ub2wcX5H1jCQ/edit#gid=0) ([discussion](https://www.reddit.com/r/LocalLLaMA/comments/13wvd0j/llm_score_v2_modern_models_tested_by_human/))
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)
- [C-Eval Leaderboard](https://cevalbenchmark.com/static/leaderboard.html)
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [YearZero's LLM Logic Tests](https://docs.google.com/spreadsheets/d/1NgHDxbVWJFolq8bLvLkuPWKC7i_R6I6W/edit#gid=1278290632)
- [HELM Core Scenarios](https://crfm.stanford.edu/helm/latest/?group=core_scenarios)
- [TextSynth Server](https://bellard.org/ts_server/)
- [llm-jeopardy](https://github.com/aigoopy/llm-jeopardy) - automated quiz show answering
- [Troyanovsky/Local-LLM-comparison](https://github.com/Troyanovsky/Local-LLM-comparison/tree/main) - one person's testing on his own standardized eval against different community models (fine-tuned quants)
- [LLM Logic Tests](https://docs.google.com/spreadsheets/d/1NgHDxbVWJFolq8bLvLkuPWKC7i_R6I6W/htmlview)
- [Asking 60+ LLMs a set of 20 questions](https://benchmarks.llmonitor.com/)
- [paperswithcode](https://paperswithcode.com/) (based off of numbers published in paper, not independently verified or standardized) 
    - [MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)
    - [ARC-c](https://paperswithcode.com/sota/common-sense-reasoning-on-arc-challenge)
    - [Hellaswag](https://paperswithcode.com/sota/sentence-completion-on-hellaswag)
    - [Winogrande](https://paperswithcode.com/sota/common-sense-reasoning-on-winogrande)
    - [BoolQ](https://paperswithcode.com/sota/question-answering-on-boolq)
    - [HumanEval](https://paperswithcode.com/sota/code-generation-on-humaneval)
- YALL
	- 
	- https://github.com/mlabonne/llm-autoeval

See also:

- [OpenAI Evals](https://github.com/openai/evals)
- [Papers and resources for LLMs evaluation](https://github.com/mlgroupjlu/llm-eval-survey)
- [2023-08-17 HN Discussion on evals](https://news.ycombinator.com/item?id=37157323)

# Hallucination
[Vectara Hallucination Leaderboard
](https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard)

# Code

- [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) - HF style leaderboard
- [Zero-Shot Replication Framework](https://github.com/emrgnt-cmplxty/zero-shot-replication) - replicate HumanEval, LeetCodeSparks, LeetCode100
- [code-eval](https://github.com/abacaj/code-eval) - scripts for running/reproducing human-eval scores on models
- [llm-humaneval-benchmarks](https://github.com/my-other-github-account/llm-humaneval-benchmarks) - HuggingFace models evald vs HumanEval+
- Multilingual Code Models Evaluation - base multilingual code generation models
- [CanAiCode Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) - using [Can AI Code? eval](https://github.com/the-crypt-keeper/can-ai-code)
- [airate](https://github.com/catid/supercharger/tree/main/airate) - C++ bug catching test
- [phi-1 prompt tests](https://twitter.com/khandelia1000/status/1675939866389934097)
    - [https://colab.research.google.com/drive/1mSb2t8NDz0o\_Cc8VgTMbhOg8kIh-cRIu?usp=sharing](https://colab.research.google.com/drive/1mSb2t8NDz0o_Cc8VgTMbhOg8kIh-cRIu?usp=sharing)

# Roleplay  


- [Another LLM Roleplay Rankings](https://rentry.co/ALLMRR)
- [The 'Ayumi' Inofficial LLM ERP Model Rating](https://rentry.org/ayumi_erp_rating)

# Context
## InfiniteBench
* https://github.com/OpenBMB/InfiniteBench 
* https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench 
* https://www.reddit.com/r/LocalLLaMA/comments/18ct9xh/infinitebench_100k_longcontext_benchmark/

# Japanese
* 

# New
* https://www.reddit.com/r/LocalLLaMA/comments/1945tfv/challenge_llms_to_reason_about_reasoning_a/
	* DiagGSM8K
* EvalPlus 0.2 - https://github.com/evalplus/evalplus/releases/tag/v0.2.0
* MMMU
  * https://huggingface.co/papers/2311.16502
  * https://arxiv.org/pdf/2311.16502.pdf
  * https://github.com/MMMU-Benchmark/MMMU
  * https://mmmu-benchmark.github.io/
  * https://twitter.com/xiangyue96/status/1729698316554801358
* GAIA: a benchmark for General AI Assistants
  * https://arxiv.org/abs/2311.12983
* GPQA: A Graduate-Level Google-Proof Q&A Benchmark
  * https://arxiv.org/abs/2311.12022
* AgentBench
  * https://github.com/THUDM/AgentBench
  * https://github.com/THUDM/AgentBench#leaderboard
* Skill-Mix
  * https://arxiv.org/abs/2310.17567
* Flash HELM
  * https://arxiv.org/abs/2308.11696
* ARB: Advanced Reasoning Benchmark for Large Language Models
	* https://arxiv.org/abs/2307.13692
* FLASK
  * https://twitter.com/SeonghyeonYe/status/1682209670302408705
  * https://arxiv.org/abs/2307.10928
  * https://kaistai.github.io/FLASK/