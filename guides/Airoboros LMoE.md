Here we experiment w/ getting a local mixture of experts.

Released 2023-08-23: [https://x.com/jon_durbin/status/1694360998797250856](https://x.com/jon_durbin/status/1694360998797250856)

Code: [https://github.com/jondurbin/airoboros#lmoe](https://github.com/jondurbin/airoboros#lmoe)

# Setup
```
# env
conda create -n airoboros
mamba env config vars set CUDA_VISIBLE_DEVICES=0
conda create -n airoboros
mamba install pip
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia


# dl models - raw llama2 models token gated
cd [/models/path]
hfdownloader -t <hf_token> -m meta-llama/Llama-2-7b-hf -s ./
hfdownloader -m jondurbin/airoboros-lmoe-7b-2.1 -s ./

# code
cd [~/airoboros]
git clone https://github.com/jondurbin/airoboros
cd airoboros
pip install .

# alternatively, this should work:
# pip install --upgrade airoboros 
```

# Run
Uses 17.54GB VRAM
```
python -m airoboros.lmoe.api \
  --base-model /models/llm/hf/meta-llama_Llama-2-7b-hf \
  --lmoe /models/llm/lora/jondurbin_airoboros-lmoe-7b-2.1 \
  --router-max-samples 1000 \
  --router-k 25 \
  --port 7777 \
  --host 127.0.0.1
```

And test:

```
❯ curl -H 'content-type: application/json' http://127.0.0.1:7777/v1/chat/completions -d '
{
  "model": "meta-llama_Llama-2-7b-hf",
  "temperature": 0.7,
  "max_tokens": 2048,
  "messages": [
    {
      "role": "system",
      "content": "A chat."
    },
    {
      "role": "user",
      "content": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    }
  ]
}'
{"id":"cmpl-589cfd58-628d-493d-a348-9b49344ed325","object":"chat.completion","created":1692807132,"duration":1.069636,"routing_duration":0.023938,"model":"meta-llama_Llama-2-7b-hf","expert":"creative","choices":[{"index":0,"message":{"role":"assistant","content":"100% of a woodchuck's weight."},"finish_reason":"stop"}],"usage":{"prompt_tokens":33,"completion_tokens":48,"total_tokens":81}}
```