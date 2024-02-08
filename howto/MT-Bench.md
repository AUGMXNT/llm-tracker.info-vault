Runpod w/ 
```
apt update
apt install byobu -y
apt install nvtop -y


# hfdownloader
cd /workspace
bash <(curl -sSL https://g.bodaay.io/hfd) -h
# get model

# Install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
/root/miniforge/bin/mamba init

byobu

# mt-bench
mamba create -n mt-bench
mamba activate mt-bench

pip install vllm
# multiGPU
pip install ray
pip install pandas

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"

cd fastchat/llm_judge
time python3 gen_model_answer.py --model-path /workspace/ShinojiResearch_Senku-70B-Full --model-id Senku-70B-Full --num-gpus-per-model 4 --num-gpus-total 4

# (mt-bench) root@6f01a12c4b6b:/workspace/FastChat/fastchat/llm_judge# time python3 gen_model_answer.py --model-path /workspace/ShinojiResearch_Senku-70B-Full --model-id Senku-70B-Full --num-gpus-per-model 4 --num-gpus-total 4
Output to data/mt_bench/model_answer/Senku-70B-Full.jsonl



# Use VLLM
python3 -m fastchat.serve.controller --host 0.0.0.0
python3 -m fastchat.serve.vllm_worker --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
pip install pydantic-settings
# vi /workspace/FastChat/fastchat/serve/openai_api_server.py
# from pydantic_settings import BaseSettings
python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8000
```


```
from vllm import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
import time
from vllm import LLM
llm = LLM(model="/workspace/ShinojiResearch_Senku-70B-Full", trust_remote_code=True, tensor_parallel_size=4)

def make_chat_prompt(user_input, sys_msg = None):
    messages = [{"role": "user", "content": user_input}]
    if sys_msg is not None:
        messages = [{"role": "system", "content": sys_msg}] + messages
    # prompt = llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    tokenizer = get_tokenizer(tokenizer_name="/workspace/ShinojiResearch_Senku-70B-Full")
    prompt = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    return prompt

import json
import pandas as pd

prompt_path = "data/mt_bench/question.jsonl"
with open(prompt_path, "r") as f:
    prompts = [json.loads(x) for x in f.readlines()]

prompt_df = pd.DataFrame(prompts)

sys_msg = "You are a helpful assistant."
prompt_df["first_prompts"] = [make_chat_prompt(x["turns"][0], sys_msg) for x in prompts]

# Default parameters from original repo
temperature_config = {
    "writing": 0.5,
    "roleplay": 0.5,
    "extraction": 0.2,
    "math": 0.1,
    "coding": 0.1,
    "reasoning": 0.1,
    "stem": 0.2,
    "humanities": 0.2,
}

output_dfs = []

for category, cat_df in prompt_df.groupby("category"):
    print(category)
    temp = temperature_config[category]
    top_p = 0.9
    repetition_penalty = 1.1
    do_sample = temp > 0
    max_new_token = 1024
    stops = ["<|im_start|>", "<|im_end|>", "<s>", "</s>"]
    sampling_params = SamplingParams(max_tokens=max_new_token, temperature=temp, top_p=top_p, repetition_penalty=repetition_penalty, stop=stops)

    input_prompts = cat_df["first_prompts"].tolist()
    outputs = llm.generate(input_prompts, sampling_params)
    output_texts = [o.outputs[0].text for o in outputs]
    cat_df["first_output"] = output_texts
    cat_df["tstamp"] = time.time()
    cat_df["generate_params"] = [{"do_sample": do_sample, "max_new_token": max_new_token, "temperature": temp, "top_p": top_p, "repetition_penalty": repetition_penalty}] * cat_df.shape[0]
    output_dfs.append(cat_df)

    print("\n\n\n\n".join([i + "\n\n" + o for i, o in zip(input_prompts, output_texts)]))
```

Output
```
time python vllm-gen.py > out.jsonl
```


```
/workspace/FastChat/fastchat/llm_judge# time python3 gen_model_answer.py --model-path /workspace/ShinojiResearch_Senku-70B-Full --model-id Senku-70B-Full --num-gpus-per-model 4 --num-gpus-total 4
```