
Build llama.cpp-hip w/ RPC
```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"     cmake -S . -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release     && cmake --build build --config Release -- -j 32
```


```
```