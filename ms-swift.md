https://www.notion.so/Ascend_Doc-2180dc3bf51680989af4cd4eee46acdd


# Training (ms-swift)

[https://swift.readthedocs.io/en/latest/BestPractices/NPU-support.html](https://swift.readthedocs.io/en/latest/BestPractices/NPU-support.html)

- Training with Megatron is not supported by Ascend, but training with Deepseed is supported. Now the R&D team is working on training with Megatron and it will be available soon.
- A conda env has been created with the latest vllm-ascend. Use `conda activate swift-npu` to enter and try the following command line for training:

```
NPROC_PER_NODE=4 \\
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \\
swift sft \\
    --model Qwen/Qwen2-7B-Instruct \\
    --dataset AI-ModelScope/blossom-math-v2 \\
    --num_train_epochs 5 \\
    --train_type lora \\
    --output_dir output \\
    --deepspeed zero3 \\
    --learning_rate 1e-4 \\
    --gradient_accumulation_steps 16 \\
    --save_steps 100 \\
    --eval_steps 100
```

# Inference (vllm-ascend)

[https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html)

- A container has been launched and can be accessed through `docker exec -it vllm-ascend-env bash`. The vllm command line can be used after entering the container.
- To run a new container:

```
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.1rc1
docker run -itd \\
    --name vllm-ascend-env \\
    --device=/dev/davinci0 \\
    --device=/dev/davinci1 \\
    --device=/dev/davinci2 \\
    --device=/dev/davinci3 \\
    --device=/dev/davinci4 \\
    --device=/dev/davinci5 \\
    --device=/dev/davinci6 \\
    --device=/dev/davinci7 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/dcmi:/usr/local/dcmi \\
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \\
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \\
    -v /etc/ascend_install.info:/etc/ascend_install.info \\
    -v /etc/localtime:/etc/localtime  \\
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
    -v /var/log/npu/:/usr/slog \\
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \\
    --net=host \\
    -it $IMAGE bash 
```

- Sometimes, downloading models from Hugging Face is slow or throttled, even when using `hf_transfer`, despite having sufficient bandwidth.

Softbank:

--env "TRANSFORMERS_CACHE=/mnt/sfs_turbo/huggingface_cache" \

# NPU Cards Needed for Models

[https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5901037.html](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5901037.html)

# Whisper on Ascend

`pip install torch-npu decorator`

```
import torch
import whisper
import urllib.request

# Download the audio if needed
url = "<https://cdn.openai.com/API/examples/data/upfirstpodcastchunkthree.wav>"
audio_path = "sample_audio.wav"
urllib.request.urlretrieve(url, audio_path)

# Set NPU device
device = torch.device("npu:4")  # switch to your NPU index
print("Using device:", device)

# Load and move model
model = whisper.load_model("base")
model = model.to(device)
print("Model loaded on:", next(model.parameters()).device)

# Transcribe
print("Starting transcription...")
result = model.transcribe(audio_path)
print("Transcription result:", result["text"])
```

## Questions By Leonard