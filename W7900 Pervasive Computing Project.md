In January 2024 I applied for the [Hackster.io AMD Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023) and a W7900 card was delivered on 2024-04-19.
- Application: https://www.hackster.io/contests/amd2023/hardware_applications/16885
- Project: https://www.hackster.io/lhl/ultra-low-latency-local-voice-assistant-avatar-4c48f2
- Repo:

I will be keeping a log here for now...


# 2024-04-20

## Flash Attention - Not Working
Last update 2024-04-08 - FA 2.5.5 being worked on internally
- https://github.com/ROCm/flash-attention/issues/35#issuecomment-2042391285

## xformers
We need to compile from code
```
# Make sure we have the ROCm version of PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0

pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
$ python -m xformers.info
Unable to find python bindings at /usr/local/dcgm/bindings/python3. No data will be captured.
xFormers 0.0.26+f663712.d20240420
memory_efficient_attention.ckF:                    unavailable
memory_efficient_attention.ckB:                    unavailable
memory_efficient_attention.ck_decoderF:            unavailable
memory_efficient_attention.ck_splitKF:             unavailable
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.decoderF:               available
memory_efficient_attention.flshattF@0.0.0:         unavailable
memory_efficient_attention.flshattB@0.0.0:         unavailable
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.triton_splitKF:         unavailable
indexing.scaled_index_addF:                        unavailable
indexing.scaled_index_addB:                        unavailable
indexing.index_select:                             unavailable
sequence_parallel_fused.write_values:              available
sequence_parallel_fused.wait_values:               available
sequence_parallel_fused.cuda_memset_32b_async:     available
sp24.sparse24_sparsify_both_ways:                  available
sp24.sparse24_apply:                               available
sp24.sparse24_apply_dense_output:                  available
sp24._sparse24_gemm:                               available
sp24._cslt_sparse_mm@0.4.0:                        available
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               False
pytorch.version:                                   2.2.2+cu121
pytorch.cuda:                                      not available
dcgm_profiler:                                     unavailable
build.info:                                        available
build.cuda_version:                                None
build.hip_version:                                 None
build.python_version:                              3.11.7
build.torch_version:                               2.2.2+cu121
build.env.TORCH_CUDA_ARCH_LIST:                    None
build.env.PYTORCH_ROCM_ARCH:                       None
build.env.XFORMERS_BUILD_TYPE:                     None
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              None
build.env.XFORMERS_PACKAGE_FROM:                   None
source.privacy:                                    open source

```