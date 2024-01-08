Replit has trained a very strong [3B parameter code completion foundational model](https://huggingface.co/replit/replit-code-v1-3b) on [The Stack](https://arxiv.org/abs/2211.15533). One fine tune beats [WizardCoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) ([StarCoder](https://huggingface.co/blog/starcoder) fine tune) [in human-eval](https://github.com/abacaj/code-eval), making it probably the strongest open code-completion model as of July 2023.

2023-07-12: Sadly, it appears that replit-code-instruct-glaive's extremely strong HumanEval performance may be mostly due to training data contamination: [https://huggingface.co/sahil2801/replit-code-instruct-glaive/discussions/3](https://huggingface.co/sahil2801/replit-code-instruct-glaive/discussions/3) (also, I noticed a [v2](https://huggingface.co/sahil2801/replit-code-instruct-glaive-v2) in progress...)

# Setup
```
### Environment
conda create -n replit
mamba install pip

```

# Running Replit HF
First let's see if we can run the included code. Install any libs if it complains
```
git clone https://huggingface.co/sahil2801/replit-code-instruct-glaive
pip install einops sentencepiece transformers torch
```

Our `test.py`:
```
# Code from https://huggingface.co/replit/replit-code-v1-3b#generation
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/data/ai/models/llm/replit/replit-code-instruct-glaive'

# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)

PROMPT = '# Python function to call OpenAI Completion API'

x = tokenizer.encode(PROMPT, return_tensors='pt')
y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer
.eos_token_id)

# decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_code)
```

Running:
```
❯ time python test.py
You are using config.init_device='cpu', but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.05s/it]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
...
Executed in   70.49 secs    fish           external
```
* Fine, but slow, let's ggml


# Convert to GGML
```
git clone https://github.com/ggerganov/ggml
mkdir build && cd build
# using system CUDA is fine
cmake -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
# or -j`nproc`
make -j32 all
pip install -r ../requirements.txt
pip install pygments

# 0 for fp32, 1 for fp16
python ./examples/replit/convert-h5-to-ggml.py [replit_model_folder] 1
# outputs ggml-model-f16.bin in folder

# Optional quantize - for me fp16 is 105ms/tok, q8_0 is 60ms/tok, q5_1 is 50ms/tok
build/bin/replit-quantize ggml-model-f16.bin q8_0.bin 7
build/bin/replit-quantize ggml-model-f16.bin q5_1.bin 9
```

# Test GGML
```
time build/bin/replit -m /data/ai/models/llm/replit/replit-code-instruct-glaive/ggml-model-f16.bin -p "# Python function to call OpenAI Completion API" -n 100
...
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090
replit_model_load: memory_size =   640.00 MB, n_mem = 65536
...
main:  predict time = 22038.78 ms / 105.45 ms per token
...
Executed in   12.97 secs    fish           external
```
* [replit-code-instruct-glaive](https://huggingface.co/sahil2801/replit-code-instruct-glaive) ([announcement](https://twitter.com/csahil28/status/1676018856047853568)) scores [63.5% on human-eval pass@1](https://github.com/abacaj/code-eval)

# C Transformers
We want [C Transformers](https://github.com/marella/ctransformers), a Python GGML wrapper since it will allow us to [use w/ LangChain](https://python.langchain.com/docs/ecosystem/integrations/ctransformers) and other Python projects:
```
CT_CUBLAS=1 pip install -U ctransformers --no-binary ctransformers
```

And our test script based off of the [usage docs](https://github.com/marella/ctransformers#usage):
```
from ctransformers import AutoModelForCausalLM
import time

MODEL = '/data/ai/models/llm/replit/replit-code-instruct-glaive/ggml-model-f16.bin'
PROMPT = '# Python function to call OpenAI Completion API'

start = time.time()
print('Loading... ', end='')
llm = AutoModelForCausalLM.from_pretrained(MODEL, model_type='replit', gpu_layers=99)
t = time.time() - start
print(f'{t:.2f}s')

tokens = llm.tokenize(PROMPT)

n = 0
start = time.time()
for token in llm.generate(tokens):
    print(llm.detokenize(token), end='')

    # 100 tokens pls
    n += 1
    if n >= 100:
        break
tps = (time.time() - start)/100
print('\n\n')
print(f'*** {t:.3f} s/t')
```

Output:
```
❯ time python test-ggml.py
Loading... 0.79s
...
*** 0.786 s/t
...
Executed in   13.22 secs    fish           external
```

# ChatDocs
We can test how CTransformers works with [ChatDocs](https://github.com/marella/chatdocs).

Our `chatdocs.yml`:
```
ctransformers:
  model: /data/ai/models/llm/replit/replit-code-instruct-glaive
  model_file: ggml-model-f16.bin
  model_type: replit
  config:
    context_length: 2048
```

Setup:
```
pip install chatdocs

# note you need to make sure chatdocs is using your conda Python
# you can either run: python `which chatdocs` [command]
# or you can modify the chatdocs bin

chatdocs download
chatdocs add /path/to/documents
chatdocs ui
```
* I ran into some issues w/ ChromaDB's indexing. Will need to debug later

# replit-3b-inference
For just testing out simple interactive usage, adapting the [inference.py](https://github.com/abacaj/replit-3B-inference/blob/main/inference.py) worked well (just replace the model and ggml path).