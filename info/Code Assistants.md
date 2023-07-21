We'll probably move this somewhere else, but I figure it might be useful to put this in public somewhere since I'm researching Coding Assistants to help w/ a refactor of a largish code base.

I'm looking for practical tools for production use here, and less of the neat toys that's popular atm.

# Hosted Services

## [Sourcegraph Cody]()
* [Desktop app](https://sourcegraph.com/get-cody) + official [neovim plugin](https://github.com/sourcegraph/sg.nvim) and [Visual Studio Code plugin](https://marketplace.visualstudio.com/items?itemName=sourcegraph.cody-ai)
* Free version for individual use (technically [Cody open sourced](https://news.ycombinator.com/item?id=35339010) but uses Sourcegraph APIs
* [No model training on your data](https://about.sourcegraph.com/terms/cody-notice)
* Uses Anthropic Claude 2

## [bloop](https://bloop.ai/)
bloop is a developer assistant that uses GPT-4 to answer questions about your codebase. The agent searches both your local and remote repositories with natural language, regex and filtered queries.
* YC Startup
* Standalone GUI App (Tauri)
* Github Access (OAuth)
* [Tantivy](https://github.com/quickwit-oss/tantivy) local search and [Qdrnt](https://github.com/qdrant/qdrant) local vector DB
* Free version for individual use
* Uses OpenAI GPT-4

## [Mutable.ai]
Supposedly will let you talk to your code base, create tests, review PRs, refactor folders, auto-document, etc...
* Free version doest no have multifile codegen or code search
* [Discussion on HN](https://news.ycombinator.com/item?id=30458465)


# OpenAI API

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


# Local

## [Refact](https://github.com/smallcloudai/refact)
Refact is an open-source Copilot alternative available as a self-hosted or cloud option.
* VS Code plugin
* Cloud or self-hosted (docker container)
* Uses AutoGPTQ/HF - supports WizardCoder for code, Llama2 for chat
* Does not embed/search through code base yet

## [TurboPilot](https://github.com/ravenscroftj/turbopilot)
TurboPilot is a self-hosted copilot clone which uses the library behind llama.cpp to run the 6 Billion Parameter Salesforce Codegen model in 4GiB of RAM. It is heavily based and inspired by on the fauxpilot project.
* Not updated very recently
* Uses [ggml fork](https://github.com/ravenscroftj/ggml/tree/6c4fe0ef5e50b76dd2539130c109e12179da0bd2) - probably could be much better if updated to [master](https://github.com/ggerganov/ggml)
* Could be updated to Codegen25-7b presumably or WizardCoder-13b

## [ggml p1](https://github.com/ggml-org/p1/discussions/1)
This project is an attempt to implement a local code completion engine utilizing large language models (LLM).
Think of it as an open-source alternative to Github Copliot that runs on your device.
* No code yet

## [localGPT](https://github.com/PromtEngineer/localGPT)
Index and search through local files w/ local models
* Uses LangChain, Chroma, AutoGPTQ, llama.cpp

# Other
Maybe useful, but not going to organize...
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

Toys
* [Dev-GPT](https://github.com/jina-ai/dev-gpt)
Can write and launch microservices. Neat, but maybe more of a toy than something useful...
* [GPT Researcher](https://github.com/assafelovic/gpt-researcher) - built to research specific queries autonomously on the web. Neat.
* https://github.com/smol-ai/developer
* https://github.com/AntonOsika/gpt-engineer