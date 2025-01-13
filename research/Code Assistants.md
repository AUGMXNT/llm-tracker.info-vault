We'll probably move this somewhere else, but I figure it might be useful to put this in public somewhere since I'm researching Coding Assistants to help w/ a refactor of a largish code base.

As of 2025-01, all the top models have improved tremendously in coding. The best coding leaderboards to look at:
- https://www.swebench.com/ - almost all the top scorers use Claude-3.5 Sonnet (20241022)
	- https://www.blackbox.ai/
	- https://gru.ai/
	- https://www.kodu.ai/
	- https://docs.all-hands.dev/
	- https://github.com/ComposioHQ/composio/tree/master/python/swe/agent
	- https://github.com/OpenAutoCoder/Agentless

# Cursor
# Windsurf

# Cline / Roo Cline

# Aider







I'm looking for practical tools for production use here, and less of the neat toys that's popular atm.

See also: 
* [https://github.com/ErikBjare/are-copilots-local-yet](https://github.com/ErikBjare/are-copilots-local-yet)

# Hosted Services

## [Cursor](https://www.cursor.so/)
GPT-4 backed VSCode-like editor

## Sourcegraph Cody
* [Desktop app](https://sourcegraph.com/get-cody) + official [neovim plugin](https://github.com/sourcegraph/sg.nvim) and [Visual Studio Code plugin](https://marketplace.visualstudio.com/items?itemName=sourcegraph.cody-ai)
* Free version for individual use (technically [Cody open sourced](https://news.ycombinator.com/item?id=35339010) but uses Sourcegraph APIs
* [No model training on your data](https://about.sourcegraph.com/terms/cody-notice)
* Uses Anthropic Claude 2

Phind
https://marketplace.visualstudio.com/items?itemName=phind.phind
https://news.ycombinator.com/item?id=39471388
https://github.com/Lomusire/gpt4-unlimited-phin


https://continue.dev/
https://github.com/continuedev/continue

https://github.com/huggingface/llm-vscode


## [bloop](https://bloop.ai/)
bloop is a developer assistant that uses GPT-4 to answer questions about your codebase. The agent searches both your local and remote repositories with natural language, regex and filtered queries.
* YC Startup
* Standalone GUI App (Tauri)
* Github Access (OAuth)
* [Tantivy](https://github.com/quickwit-oss/tantivy) local search and [Qdrnt](https://github.com/qdrant/qdrant) local vector DB
* Free version for individual use
  * Free version updated to only **10 uses per day**! ($20/mo)
* Uses OpenAI GPT-4, not possible to use your own API key

## [Mutable.ai](https://mutable.ai/)
Supposedly will let you talk to your code base, create tests, review PRs, refactor folders, auto-document, etc...
* Free version doest no have multifile codegen or code search
* [Discussion on HN](https://news.ycombinator.com/item?id=30458465)

# OpenAI API

## Code Interpreter
https://github.com/haseeb-heaven/code-interpreter

## [Instrukt](https://github.com/blob42/Instrukt)
Coding AI: A coding assistant. Create indexes over any code base and attach it to the agent to do RAG (Retrieval Augmented Generation)

## [Maccarone](https://github.com/bsilverthorn/maccarone)
VSCode plugin that uses GPT-4 to write stubbed code snippets

## [Mentat](https://github.com/biobootloader/mentat)
Mentat is an interactive command-line tool that can load and coordinate edits from your command line. It can directly edit files without copy and pasting (Y/n/i)
* GPT-4 API
* Shows tokens and costs for operations

## [GPT-4 Coding Assistant](https://github.com/alfiedennen/GPT-4-coding-assistant)
GPT-4 Coding Assistant is a web application that leverages the power of OpenAI's GPT-4 to help developers with their coding tasks. The application serves as an interactive chatbot that assists in code generation, understanding, and troubleshooting. It also utilizes embeddings and the Annoy library to search for similar code snippets in the provided codebase, offering more contextually relevant responses.
* Uses OpenAI Embeddings

## [aider](https://github.com/paul-gauthier/aider)
aider is a command-line chat tool that allows you to write and edit code with OpenAI's GPT models. You can ask GPT to help you start a new project, or modify code in your existing git repo. Aider makes it easy to git commit, diff & undo changes proposed by GPT without copy/pasting. It also has features that help GPT-4 understand and modify larger codebases.
* Doesn't have full embeddings or search, [uses ctags](https://aider.chat/docs/ctags.html)

## [Chapyter](https://github.com/chapyter/chapyter)
Chat w/ your Jupyter Lab code

## [CodeGPT](https://github.com/appleboy/codegpt)
Write commit messages/does code review via git hook w/ OpenAI API

## Local Code Interpreter
* https://github.com/ricklamers/gpt-code-ui
* https://github.com/iamgreggarcia/codesherpa
* https://github.com/KillianLucas/open-interpreter

## [OpenAI ChatGPT-4 Code Interpreter](https://chat.openai.com/?model=gpt-4-code-interpreter)
You get a VM sandbox and GPT-4 that knows how to run Python with it.
* Limit of 50 exchanges/3 hours
* VM will be recycled after a timeout period


# Local Models
With the release of [Meta's Code Llama](https://about.fb.com/news/2023/08/code-llama-ai-for-coding/) there is finally a model that is competitive with GPT-4 for code generation:
* [Beating GPT-4 on HumanEval with a Fine-Tuned CodeLlama-34B](https://www.phind.com/blog/code-llama-beats-gpt4)
* [WizardCode 34B](https://twitter.com/WizardLM_AI/status/1695396881218859374) also just dropped.

Also, it's worth pointing out there was another model, a Llama2 70B full fine tune, [Lemur](https://github.com/OpenLemur/Lemur) that also performs quite well.

Note there are some API providers for those that can't run these locally:
* [https://codellama.lepton.run/](https://codellama.lepton.run/) and https://codellama.lepton.run/api/v1
* [https://labs.perplexity.ai/](https://labs.perplexity.ai/)
* [https://openrouter.ai/](https://openrouter.ai/)
* [https://app.endpoints.anyscale.com/](https://app.endpoints.anyscale.com/)
* [https://api.together.xyz/playground](https://api.together.xyz/playground)
* [https://github.com/TheBlokeAI/dockerLLM/](https://github.com/TheBlokeAI/dockerLLM/) - RunPod One-Click

# Local Apps

## [Refact](https://github.com/smallcloudai/refact)
Refact is an open-source Copilot alternative available as a self-hosted or cloud option.
* VS Code plugin
* Cloud or self-hosted (docker container)
* Uses AutoGPTQ/HF - supports WizardCoder for code, Llama2 for chat
* Does not embed/search through code base yet


## [Continue](https://github.com/continuedev/continue)
Continue is the open-source autopilot for software development—an IDE extension that brings the power of ChatGPT to VS Code and JetBrains

## [localpilot](https://github.com/danielgross/localpilot)
Slick menubar interface to allow switching Github Copilot to use local models (built for Macs). Uses llama-cpp-python so should support any GGUF.

## [Code Llama for VSCode](https://github.com/xNul/code-llama-for-vscode)
An API which mocks llama.cpp to enable support for Code Llama with the Continue Visual Studio Code extension.

## [ggml p1](https://github.com/ggml-org/p1/discussions/1)
This project is an attempt to implement a local code completion engine utilizing large language models (LLM).
Think of it as an open-source alternative to Github Copliot that runs on your device.
* No code yet

## (deprecated) [TurboPilot](https://github.com/ravenscroftj/turbopilot)
TurboPilot is a self-hosted copilot clone which uses GGML,the library behind llama.cpp, to run local code completion models. It was originally designed to run Salesforce codegen models but has recently been updated to provide support for [Starcoder](https://huggingface.co/blog/starcoder), [Wizardcoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) and most recently, [StableCode Instruct](https://stability.ai/blog/stablecode-llm-generative-ai-coding) from StabilityAI. It is heavily based and inspired by on the [fauxpilot](https://github.com/fauxpilot/fauxpilot) project.

* Provides GPU inference via CUDA for Nvidia devices and OpenCL for Intel/AMD CPUS
* Used to depend on a custom ggml fork but now tracks the [ggerganov/ggml](https://github.com/ggerganov/ggml) project.

## [localGPT](https://github.com/PromtEngineer/localGPT)
Index and search through local files w/ local models
* Uses LangChain, Chroma, AutoGPTQ, llama.cpp

# Other
Maybe useful, but not going to organize...

Code Interpreters
* https://github.com/KillianLucas/open-interpreter

* https://docs.lmql.ai/en/latest/quickstart.html
  * https://lmql.ai/playground/?snippet=gist:lbeurerkellner/24d68046a3c88c43cf09185f0f5c3680/raw/eval-and-call.json


* https://www.codium.ai/ - write tests
* https://writer.mintlify.com/ Auto comment/document
* https://readable.so/ - auto comment
* https://useadrenaline.com/ - talk to repo
* https://www.grit.io/ - auto upgrades/tech-debt
* https://sweep.dev/ - bug reports, pull requests
* https://whatthediff.ai/ - auto code review
* https://deepnote.com/ - data analysis tool
* https://stepsize.com/ - (CollabGPT) team updates and collab
* https://www.lindy.ai/ - personal assistant
* https://www.hyperwriteai.com/personal-assistant - personal assistant
* [Codesee](https://www.codesee.io/) - A general code-mapping/refactoring tool that for $29/mo has an AI chat feature (but has a [waitlist](https://www.codesee.io/ai)?) 

Toys
* [Dev-GPT](https://github.com/jina-ai/dev-gpt)
Can write and launch microservices. Neat, but maybe more of a toy than something useful...
* [GPT Researcher](https://github.com/assafelovic/gpt-researcher) - built to research specific queries autonomously on the web. Neat.
* https://github.com/smol-ai/developer
* https://github.com/AntonOsika/gpt-engineer