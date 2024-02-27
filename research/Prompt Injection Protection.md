We should have a tool to be able to track Github project activity...

Papers collection:
- https://huggingface.co/collections/leonardlin/prompt-injection-65dd93985012ec503f2a735a

Techniques:
- Input
	- heuristics
	- fine-tuned models
		- chunked detection of inputs?
	- prompt rewriting
- Output filtering
	- heuristics
	- canary tokens
	- fine-tuned models
- Logging
	- embeddings/vector db storage of attacks

# [rebuff](https://github.com/protectai/rebuff)
- Apache 2.0
- Heuristics: Filter out potentially malicious input before it reaches the LLM.
- LLM-based detection: Use a dedicated LLM to analyze incoming prompts and identify potential attacks.
- VectorDB: Store embeddings of previous attacks in a vector database to recognize and prevent similar attacks in the future.
- Canary tokens: Add canary tokens to prompts to detect leakages, allowing the framework to store embeddings about the incoming prompt in the vector database and prevent future attacks.

```
pip install rebuff
```
# [vigil](https://github.com/deadbits/vigil-llm)
`Vigil` is a Python library and REST API for assessing Large Language Model prompts and responses against a set of scanners to detect prompt injections, jailbreaks, and other potential threats. This repository also provides the detection signatures and datasets needed to get started with self-hosting.
- Analyze LLM prompts for common injections and risky inputs
- [Use Vigil as a Python library](https://github.com/deadbits/vigil-llm#using-in-python) or [REST API](https://github.com/deadbits/vigil-llm#running-api-server)
- Scanners are modular and easily extensible
- Evaluate detections and pipelines with **Vigil-Eval** (coming soon)
- Available scan modules
    - [x]  Vector database / text similarity
        - [Auto-updating on detected prompts](https://vigil.deadbits.ai/overview/use-vigil/auto-updating-vector-database)
    - [x]  Heuristics via [YARA](https://virustotal.github.io/yara)
    - [x]  Transformer model
    - [x]  Prompt-response similarity
    - [x]  Canary Tokens
    - [x]  Sentiment analysis
    - [ ]  Relevance (via [LiteLLM](https://docs.litellm.ai/docs/))
    - [ ]  Paraphrasing
- Supports [local embeddings](https://www.sbert.net/) and/or [OpenAI](https://platform.openai.com/)
- Signatures and embeddings for common attacks
- Custom detections via YARA signatures
- [Streamlit web UI playground](https://vigil.deadbits.ai/overview/use-vigil/web-server/web-ui-playground)
# [llm-guard](https://github.com/protectai/llm-guard)
- MIT
- Input/Output scanning, eg:
	- Prompt scanners
		- [Anonymize](https://llm-guard.com/input_scanners/anonymize/)
		- [BanCompetitors](https://llm-guard.com/input_scanners/ban_competitors/)
		- [BanSubstrings](https://llm-guard.com/input_scanners/ban_substrings/)
		- [BanTopics](https://llm-guard.com/input_scanners/ban_topics/)
		- [Code](https://llm-guard.com/input_scanners/code/)
		- [Gibberish](https://llm-guard.com/input_scanners/gibberish/)
		- [InvisibleText](https://llm-guard.com/input_scanners/invisible_text/)
		- [Language](https://llm-guard.com/input_scanners/language/)
		- [PromptInjection](https://llm-guard.com/input_scanners/prompt_injection/)
		- [Regex](https://llm-guard.com/input_scanners/regex/)
		- [Secrets](https://llm-guard.com/input_scanners/secrets/)
		- [Sentiment](https://llm-guard.com/input_scanners/sentiment/)
		- [TokenLimit](https://llm-guard.com/input_scanners/token_limit/)
		- [Toxicity](https://llm-guard.com/input_scanners/toxicity/)
	- Output scanners
		- [BanCompetitors](https://llm-guard.com/output_scanners/ban_competitors/)
		- [BanSubstrings](https://llm-guard.com/output_scanners/ban_substrings/)
		- [BanTopics](https://llm-guard.com/output_scanners/ban_topics/)
		- [Bias](https://llm-guard.com/output_scanners/bias/)
		- [Code](https://llm-guard.com/output_scanners/code/)
		- [Deanonymize](https://llm-guard.com/output_scanners/deanonymize/)
		- [JSON](https://llm-guard.com/output_scanners/json/)
		- [Language](https://llm-guard.com/output_scanners/language/)
		- [LanguageSame](https://llm-guard.com/output_scanners/language_same/)
		- [MaliciousURLs](https://llm-guard.com/output_scanners/malicious_urls/)
		- [NoRefusal](https://llm-guard.com/output_scanners/no_refusal/)
		- [ReadingTime](https://llm-guard.com/output_scanners/reading_time/)
		- [FactualConsistency](https://llm-guard.com/output_scanners/factual_consistency/)
		- [Gibberish](https://llm-guard.com/output_scanners/gibberish/)
		- [Regex](https://llm-guard.com/output_scanners/regex/)
		- [Relevance](https://llm-guard.com/output_scanners/relevance/)
		- [Sensitive](https://llm-guard.com/output_scanners/sensitive/)
		- [Sentiment](https://llm-guard.com/output_scanners/sentiment/)
		- [Toxicity](https://llm-guard.com/output_scanners/toxicity/)
		- [URLReachability](https://llm-guard.com/output_scanners/url_reachability/)

```
pip install llm-guard
```

# [langkit](https://github.com/whylabs/langkit)

LangKit is an open-source text metrics toolkit for monitoring language models. It offers an array of methods for extracting relevant signals from the input and/or output text, which are compatible with the open-source data logging library [whylogs](https://whylogs.readthedocs.io/en/latest).
- Apache 2.0

The out of the box metrics include:
- [Text Quality](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/quality.md)
    - readability score
    - complexity and grade scores
- [Text Relevance](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/relevance.md)
    - Similarity scores between prompt/responses
    - Similarity scores against user-defined themes
- [Security and Privacy](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/security.md)
    - patterns - count of strings matching a user-defined regex pattern group
    - jailbreaks - similarity scores with respect to known jailbreak attempts
    - prompt injection - similarity scores with respect to known prompt injection attacks
    - hallucinations - consistency check between responses
    - refusals - similarity scores with respect to known LLM refusal of service responses
- [Sentiment and Toxicity](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/sentiment.md)
    - sentiment analysis
    - toxicity analysis

```
pip install langkit[all]
```
# [promptmap](https://github.com/utkusen/promptmap)
promptmap is a tool that automatically tests prompt injection attacks on ChatGPT instances. It analyzes your ChatGPT rules to understand its context and purpose. This understanding is used to generate creative attack prompts tailored for the target. promptmap then run a ChatGPT instance with the system prompts provided by you and sends attack prompts to it. It can determine whether the prompt injection attack was successful by checking the answer coming from your ChatGPT instance.
- MIT
# [llm-security](https://github.com/dropbox/llm-security)
This repository contains scripts and related documentation that demonstrate attacks against large language models using repeated character sequences. These techniques can be used to execute prompt injection on content-constrained LLM queries.
# **[Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection)**
- No license
For attacks, clients can use one of the following key words: naive, escape, ignore, fake_comp, and combine. Each of they corresponds one attack strategy mentioned in the paper.

For defenses, specifying the following key words when creating the app:

1. By default, "no" is used, meaning that there is no defense used.
2. Paraphrasing: "paraphrasing"
3. Retokenization: "retokenization"
4. Data prompt isolation: "delimiters", "xml", or "random_seq"
5. Instructional prevention: "instructional"
6. Sandwich prevention: "sandwich"
7. Perplexity-based detection: use "ppl-\[window_size\]-\[threshold\]". When this is for non-windowed PPL detection, use "ppl-all-\[threshold\]". For example, "ppl-all-3.0" means the PPL detector without using windows when the threshold is 3.0. Another example is that "ppl-5-3.5" means to use a windowed PPL detector with threshold being 3.5.
8. LLM-based detection: "llm-based"
9. Response-based detection: "response-based"
10. Proactive detection: "proactive"

Clients are recommended to navigate to ./configs/model_configs/ to check the supported LLMs. Clients should also enter their own PaLM2 API keys in the corresponding areas in the model config. Supports for other models will be added later.
# [Prompt-adversarial collections](https://github.com/yunwei37/prompt-hacker-collections)
- MIT
This repository serves as a comprehensive resource on the study and practice of prompt-injection attacks, defenses, and interesting examples. It contains a collection of examples, case studies, and detailed notes aimed at researchers, students, and security professionals interested in this topic.