075 - TRL + Custom megablocks-hip fork
- gets to step 0 at least


Sample Blog - GPT2 training:
https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html
https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/megablocks


```
drwxr-xr-x  5 root root  4096 Sep 25 02:31 ..
-rw-r--r--  1 root root  2599 Sep 25 02:31 megablocks_moe_gpt2_125m_8gpu.sh
-rwxr-xr-x  1 root root  1794 Sep 25 02:31 megablocks_gpt2_125m_8gpu.sh
-rwxr-xr-x  1 root root   902 Sep 25 02:31 data_preprocessing.sh
drwxr-xr-x 10 root root  4096 Sep 25 04:30 iter_0002000
drwxr-xr-x  3 root root  4096 Sep 25 04:30 .
-rw-r--r--  1 root root     4 Sep 25 04:31 latest_checkpointed_iteration.txt
-rw-r--r--  1 root root 53225 Sep 25 04:31 train.log
root@amd2:/app/src# du -sh *
4.0K    data_preprocessing.sh
55G     iter_0002000
4.0K    latest_checkpointed_iteration.txt
4.0K    megablocks_gpt2_125m_8gpu.sh
4.0K    megablocks_moe_gpt2_125m_8gpu.sh
52K     train.log

```

Context
```
docker run -it --rm \
--privileged -v ./:/app \
--network=host --device=/dev/kfd \
--device=/dev/dri --group-add video \
--name=my_megablocks --cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--ipc=host --shm-size 16G \
megablocks
```