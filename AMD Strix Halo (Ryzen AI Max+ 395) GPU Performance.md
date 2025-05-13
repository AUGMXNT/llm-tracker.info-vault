I've been doing some (ongoing) testing on a Strix Halo system and with a bunch of desktop systems coming out soon, I figured it might be worth sharing a few notes of on the current performance.

In terms of raw compute specs, the Ryzen AI Max 395's Radeon 8060S has 40 RDNA3.5 CUs. At a max clock of 2.9GHz this should have a peak of 59.4 FP16/BF16 TFLOPS:

    512 * 40 * 2.9e9 / 1e12 = 59.392 FP16 TFLOPS

This peak value requires either WMMA or wave32 VOPD otherwise the max is halved.

Using [mamf-finder](https://github.com/shisa-ai/mamf-finder) to test, without hipBLASLt, it takes about 35 hours to run and only gets to 5.1 BF16 TFLOPS (<9% max theoretical). When run with hipBLASLt, this goes up to 36.9 TFLOPS (>60% max theoretical) which is [comparable to AMD CDNA3 efficiency](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-flops).

On the memory bandwidth (MBW) front, [rocm\_bandwidth\_test](https://github.com/ROCm/rocm_bandwidth_test) gives about 212 GB/s peak bandwidth (DDR5-8000 on a 256-bit bus gives a theoretical peak MBW of 256 GB/s). This is roughly in line with [max MBW tested by ThePhawx](https://youtu.be/yiHr8CQRZi4?t=1191), [jack stone](https://youtu.be/UXjg6Iew9lg?t=239), and others.

CPU to GPU memory copies is \~84 GB/s.

The system I am using is set to almost all of its memory dedicated to GPU - 8GB GART and 110 GB GTT.

Raw perf is fine, but probably what most people in the sub are interested is how it performs for llama.cpp inference. First I'll test with the standard [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) (Q4\_0) so you can compare vs say some of my [prior efficiency testing](https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/) or the [official llama.cpp Apple Silicon M-series performance thread](https://github.com/ggml-org/llama.cpp/discussions/4167).

I ran with a few different backends, and this was a bit surprising to me:

|Run|pp512 (t/s)|tg128 (t/s)|Max Mem (MiB)|
|:-|:-|:-|:-|
|CPU|294.64 ± 0.58|28.94 ± 0.04||
|CPU + FA|294.36 ± 3.13|29.42 ± 0.03||
|HIP|348.96 ± 0.31|48.72 ± 0.01|4219|
|HIP + FA|331.96 ± 0.41|45.78 ± 0.02|4245|
|HIP + WMMA|322.63 ± 1.34|48.40 ± 0.02|4218|
|HIP + WMMA + FA|343.91 ± 0.60|50.88 ± 0.01|4218|
|Vulkan|881.71 ± 1.71|52.22 ± 0.05|**3923**|
|Vulkan + FA|**884.20 ± 6.23**|**52.73 ± 0.07**|**3923**|

The HIP version performs far below what you'd expect in terms of tok/TFLOP efficiency for prompt processing even vs other RDNA3 architectures:

* `gfx1103` Radeon 780M iGPU gets 14.51 tok/TFLOP. At that efficiency you'd expect the about 850 tok/s that the Vulkan backend delivers.
* `gfx1100` Radeon 7900 XTX gets 25.12 tok/TFLOP. At that efficiency you'd expect almost 1500 tok/s, almost double what the Vulkan backend delivers, and >4X what the current HIP backend delivers.
* HIP pp512 barely beats out CPU backend numbers. I don't have an explanation for this.
* Just for a reference of how bad the HIP performance is, an 18CU M3 Pro has \~12.8 FP16 TFLOPS (4.6X less compute than Strix Halo) and delivers about the same pp512. Lunar Lake Arc 140V has 32 FP16 TFLOPS (almost 1/2 Strix Halo) and has a pp512 of 657 tok/s (1.9X faster)
* With the Vulkan backend pp512 is about the same as an M4 Max and tg128 is about equivalent to an M4 Pro

Testing a similar system with Linux 6.14 vs 6.15 showed a 15% performance difference so it's possible future driver/platform updates will improve/fix Strix Halo's ROCm/HIP compute efficiency problems.

So that's a bit grim, but I did want to point out one silver lining. With the recent fixes for Flash Attention with the llama.cpp Vulkan backend, I did some higher context testing, and here, the HIP + rocWMMA backend actually shows some strength. It has basically no decrease in either pp or tg performance at 8K context and uses the least memory to boot:

|Run|pp8192 (t/s)|tg8192 (t/s)|Max Mem (MiB)|
|:-|:-|:-|:-|
|HIP|245.59 ± 0.10|12.43 ± 0.00|6+10591|
|HIP + FA|190.86 ± 0.49|30.01 ± 0.00|7+8089|
|HIP + WMMA|230.10 ± 0.70|12.37 ± 0.00|6+10590|
|HIP + WMMA + FA|368.77 ± 1.22|**50.97 ± 0.00**|**7+8062**|
|Vulkan|487.69 ± 0.83|7.54 ± 0.02|7761+1180|
|Vulkan + FA|**490.18 ± 4.89**|32.03 ± 0.01|7767+1180|

* You need to have `rocmwmma` installed - many distros have packages but you need gfx1151 support is very new ([PR 538](https://github.com/ROCm/rocWMMA/pull/538) from last week) so you will probably need to build your own from source: [https://github.com/ROCm/rocWMMA](https://github.com/ROCm/rocWMMA)
* You should then rebuild llama.cpp with `-DGGML_HIP_ROCWMMA_FATTN=ON`
* HIP + WMMA + FA is not only the fastest for tg8192, but also uses the least memory

If you mostly do 1-shot inference, then the Vulkan + FA backend is actually probably the best. If you frequently have longer conversations then HIP + WMMA + FA is probalby the way to go, even if prompt processing is slower than it should be atm.

I also ran some tests with Qwen3-30B-A3B (UD-Q4\_K\_XL). Larger MoEs is where these large unified memory APUs really shine. Here are Vulkan results. One thing worth noting, and this is particular to the Qwen3 MoE and Vulkan backend, but using `-b 256` significantly improves the pp512 performance:

|Run|pp512 (t/s)|tg128 (t/s)|
|:-|:-|:-|
|Vulkan|70.03 ± 0.18|75.32 ± 0.08|
|Vulkan b256|118.78 ± 0.64|74.76 ± 0.07|

While the pp512 is slow, tg128 is as speedy as you'd expect for 3B activations.

This is still only a 16.5 GB model though. Let's go bigger. The Llama 4 Scout is 109B parameters and 17B activations and the UD-Q4\_K\_XL is 57.93 GiB.

|Run|pp512 (t/s)|tg128 (t/s)|
|:-|:-|:-|
|Vulkan|102.61 ± 1.02|20.23 ± 0.01|
|HIP|GPU Hang|GPU Hang|

While Llama 4 has gotten some guff, this is a model that performs at about as well as Llama 3.3 70B, but tg is 4X faster.

I've also been able to successfully RPC llama.cpp to test some truly massive models, but I'll leave that for a future discussion.

Besides romWMMA, I was able to build a ROCm 6.4 image for Strix Halo (gfx1151) using u/scottt's dockerfiles: https://github.com/scottt/rocm-TheRock. This builds hipBLASLt for gfx1151.

I was able to build AOTriton without too much hassle (takes about 1h wall time on Strix Halo for just the gfx1151 GPU\_TARGET).

Composable Kernel (CK) has gfx1151 support now as well and builds in about 15 minutes.

PyTorch was a huge PITA to build, but with a fair amount of elbow grease, I was able to get HEAD (2.8.0a0) compiling, however it still has problems with Flash Attention not working even with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` set.

There's a lot of active work ongoing for PyTorch. For those interested, I'd recommend following along:

* [https://github.com/ROCm/ROCm/issues/4499](https://github.com/ROCm/ROCm/issues/4499)
* [https://github.com/ROCm/TheRock/discussions/244](https://github.com/ROCm/TheRock/discussions/244)

I won't bother testing training or batch inference engines until at least PyTorch FA is sorted. Current testing shows fwd/bwd pass to be in the \~1 TFLOPS ballpark...