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
* This is a super simple client, you'd want to add token counting and message log truncation if you were going to use it seriously
* For me on a fast system (NVMe SSD, 5950X, 4090), takes 2-3min to load, maybe shortened w/ bitsandbytes...
* The routing works but obvious not well, it's just a POC and could be improved tremendously
* llama2-7b is dumb as a box of rocks, lol

To test the routing, I recommend some simple queries like:
```
# function/code
Write me Python "hello world" FastAPI script.

# creative
Write me a haiku.

# reasoning
There are two ducks in front of a duck, two ducks behind a duck and a duck in the middle. How many ducks are there?
``

Part of these were bugs that I reported and got stamped out:
* `/v1/models` endpoint bug
* CORS errors

Just as an FYI, here are the clients I tried that didn't work:
* [mckaywrigley/chatbot-ui](https://github.com/mckaywrigley/chatbot-ui)
* https://github.com/blrchen/chatgpt-minimal
* https://github.com/kierangilliam/chatwithme.chat
* https://github.com/chatgptui/desktop
* https://github.com/patrikzudel/PatrikZeros-ChatGPT-API-UI
* https://github.com/Yisus7u7/openai-cli-client
* https://github.com/peterdemin/openai-cli