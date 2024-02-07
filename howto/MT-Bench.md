Runpod w/ 
```
apt update
apt install byobu
apt install nvtop


# hfdownloader
cd /workspace
bash <(curl -sSL https://g.bodaay.io/hfd) -h
# get model

# Install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
mamba init

# mt-bench
mamba create -n mt-bench
mamba activate mt-bench

pip install vllm

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"


# Use VLLM
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.vllm_worker --model-path [MODEL-PATH]
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```