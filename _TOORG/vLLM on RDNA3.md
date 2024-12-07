# 2024-12-7
See: https://embeddedllm.com/blog/vllm-now-supports-running-gguf-on-amd-radeon-gpu

Build
```
paru -S docker docker-compose docker-buildx

git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 sudo docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
```

You can check on the model here:
```
sudo docker images
```

Run the docker instance (mount your HF and models folder)
```
sudo docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /models:/app/model \
   -v /home/lhl/.cache/huggingface:/root/.cache/huggingface \
   docker.io/library/vllm-rocm \
   bash
```


# llama.cpp Comparions
## llama-bench
```

```

## benchmark_serving.py

Run server:
```
~/ai/llama.cpp/build/bin/llama-server -m /models/gguf/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf -ngl 99 -a 'llama3.1-8b' -sp
```
- `-sp` special token output required

Run benchmark:
```
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
pip install numpy datasets Pillow tqdm transformers

python benchmark_serving.py --backend openai-chat --model "llama3.1:8b-instruct-q5_K_M" --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --port 11434 --endpoint /v1/chat/completions


python benchmark_serving.py --backend openai-chat --model 'llama3.1-8b' --base-url 'http://localhost:8080' --host localhost --port 8080 --endpoint='/v1/chat/completions' --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```