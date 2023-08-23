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

# Client
The current version of the API is quite picky and I couldn't find anything compatible... here's a simple client that ChatGPT-4 CI helped me write:

```
import requests
import json

SYSTEM_PROMPT = 'A chat with a helpful assistant.'

HOST = 'http://127.0.0.1:7777'
MODEL = 'meta-llama_Llama-2-7b-hf' 
MAX_CONTEXT = 4096

def send_request(messages):
    url = f'{HOST}/v1/chat/completions'
    headers = {'content-type': 'application/json'}
    payload = {
        'model': MODEL,
        'temperature': 0.7,
        'max_tokens': 2048,
        'messages': messages
    }


    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def get_assistant_reply(response):
    return response['choices'][0]['message']['content']

def interactive_chat():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        response = send_request(messages)
        print(response)
        assistant_reply = get_assistant_reply(response)
        messages.append({"role": "assistant", "content": assistant_reply})
        print("Assistant:", assistant_reply)
        if user_input.lower() == 'exit':
            break

interactive_chat()
```