# Install
To install MLC, first we will install the prebuild MLC-LLM package:
- https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages
```
# conda/mamba env
mamba create -n mlc python=3.11
mamba activate mlc

# Install the version you want
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu122 mlc-ai-nightly-cu122

# We'll want this
conda install -c conda-forge libgcc-ng

# Verify
python -c "import mlc_chat; print(mlc_chat)"
# Prints out: <module 'mlc_chat' from '/path-to-env/lib/python3.11/site-packages/mlc_chat/__init__.py'>

# Verify
mlc_chat --help
```

We'll also want to install TVM Unity to be able to build new models:
- https://llm.mlc.ai/docs/install/tvm.html#option-1-prebuilt-package
```
# Install the version you want
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu122

# Locate TVM package
python -c "import tvm; print(tvm.__file__)"
/some-path/lib/python3.11/site-packages/tvm/__init__.py

# Confirm which TVM library is used
python -c "import tvm; print(tvm._ffi.base._LIB)"
<CDLL '/some-path/lib/python3.11/site-packages/tvm/libtvm.dylib', handle 95ada510 at 0x1030e4e50>

# Reflect TVM build option
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```

# Make an MLC model
This is a two ste process:
- Convert weights from HF weights
	- https://llm.mlc.ai/docs/compilation/convert_weights.html
- Compile the model so inference can be run
	- https://llm.mlc.ai/docs/compilation/compile_models.html

```
mkdir -p dist/libs

# Convert model (quantize as well) - need to d/l path
mlc_chat convert_weight /models/shisa-7b-v1 -o dist/shisa-7b-v1-q4f16_1 --quantization q4f16_1

# gen_config: generate mlc-chat-config.json and process tokenizers
# Note you need to pick one of their prebuilt of a "custom" conv-template
# doesn't seem to support chat_template?
mlc_chat gen_config /models/shisa-7b-v1 -o dist/shisa-7b-v1-q4f16_1 --quantization q4f16_1 --conv-template llama-2

# 2. compile: compile model library with specification in mlc-chat-config.json
mlc_chat compile  dist/shisa-7b-v1-q4f16_1/mlc-chat-config.json --device cuda -o dist/libs/shisa-7b-v1-q4f16_1-cuda.so
```

# Run Model
In Python
```
from mlc_chat import ChatModule
cm = ChatModule(model="dist/shisa-7b-v1-q4f16_1", model_lib_path="dist/libs/shisa-7b-v1-q4f16_1-cuda.so")
cm.generate("hi")
```

Note, if you run without the library path, it will not JIT compile the module!:
```
cm = ChatModule(model="dist/shisa-7b-v1-q4f16_1")
```

Or you can run with `mlc_chat`
```
mlc_chat chat dist/shisa-7b-v1-q4f16_1 --model-lib-path dist/libs/shisa-7b-v1-q4f16_1-cuda.so
```
# Benchmark
MLC's benchmarking is a bit unfortunate. It doesn't support specifying a length for the prefill/prompt generation and for generation length separately...
```
mlc_chat bench dist/shisa-7b-v1-q4f16_1 --model-lib-path dist/libs/shisa-7b-v1-q4f16_1-cuda.so --prompt "Hello" --generate-length 3968
```
