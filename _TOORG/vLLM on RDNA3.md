2024-12-7

Install
```
paru -S docker docker-compose docker-buildx

git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 sudo docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
```

```
sudo docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /path/to/hfmodels:/app/model \  # if you have pre-downloaded the model weight, else ignore
   ghcr.io/embeddedllm/vllm-rocm:navi-gguf-690c57c \
   bash
```