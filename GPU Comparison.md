A M4 Pro has 273 GB/s of MBW and roughly 7 FP16 TFLOPS. A 5090 has 1.8TB/s of MBW and likely somewhere around 200 FP16 Tensor TFLOPS (for llama.cpp inference this is even more stark as it is doing roughly 90% INT8 for its CUDA backend and the 5090 likely has >800 INT8 dense TOPS). So we're looking at a device with about 5.6X more memory bandwidth for token generation, and >100X more processing power for prefill/prompt processing or diffusion workloads. The performance and cost/perf are really not on the same planet.




https://www.reddit.com/r/LocalLLaMA/comments/191srof/amd_radeon_7900_xtxtx_inference_performance/
https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/
https://www.reddit.com/r/LocalLLaMA/comments/1gheslj/testing_llamacpp_with_intels_xe2_igpu_core_ultra/


[](https://www.reddit.com/r/LocalLLaMA/comments/1hvqydy/comment/m5vkqdw/)

Actually there **is** exact info on the MBW. The linked writeup says the HP device using (the max suppoprted) LPDDR5x-8000 and per the [official AMD specsheet](https://www.amd.com/en/products/processors/laptop/ryzen/300-series/amd-ryzen-ai-max-plus-395.html), it is "256-bit LPDDR5x". A little multiplication, you get 256GB/s of MBW.

[Per Apple](https://support.apple.com/en-us/121553), 16/20CU M4 Pros have 273 GB/s MBW, 32CU M4 Max has 410GB/s, and 40CU M4 Max has 546GB/s (they use LPDDRx-8533). You need to use an M4 Max to get 128GB of memory, and the starting price for that is $4700, but you get >2X the MBW. The main weakness of the Apple Silicon remains compute. The M4 Max has almost half the raw TFLOPS vs Strix Halo (important for prompt processing, batching, any diffusion or image generation).

See:

Just a clarification, per the Ampere GA 102 whitepaper (Appendix A, Table 9) for the 3090 (328 3rd gen Tensor Cores):

- Peak FP16 Tensor TFLOPS with FP32 (dense/sparse): 71/142
    
    - So for most training (unless you're DeepSeek) you're looking at 71 TFLOPS
        
- Peak INT8 Tensor TOPS (dense/sparse): 284/568
    
    - llama.cpp's CUDA backed is doing 90% dense INT8
        

And from the Ada GPU architecture whitepaper (Appendix A, Table 2) for the 4090 (512 4th gen Tensor Cores):

- Peak FP16 Tensor TFLOPS with FP32 (dense/sparse): 165.2/330.4
    
    - So for most training you're looking at 165 TFLOPS
        
- Peak INT8 Tensor TOPS (dense/sparse): 660.6/1321.2
    
    - llama.cpp's CUDA backed is doing 90% dense INT8
        

AFAIK, Nvidia has yet to publicly publish a Blackwell/GB102 technical architecture doc.


Here's chart I made. The GB10 announcement seems very light on details atm. Based on Nvidia's recent technical marketing, I'll assuming the 1 PFLOPS FP4 mentioned is sparse, so dense would be 500 TFLOPS. From there I use the Blackwell datasheet to back-calculate the dense FP16 and INT8 ratios based on the Blackwell fovd: [https://resources.nvidia.com/en-us-blackwell-architecture](https://resources.nvidia.com/en-us-blackwell-architecture)

|Specification|Apple M4 Max|AMD Ryzen AI Max Plus 395|NVIDIA GB10 Digits|
|---|---|---|---|
|Release Date|November 8, 2024|Spring 2025|May 2025|
|Price|$4,699 (MBP 14)|$1200+|$3,000|
|Memory|128GB LPDDR5X-8533|128GB LPDDR5X-8000|128GB LPDDR5X|
|Memory Bandwidth|546 GB/s|256 GB/s|Unknown, 256GB/s or 512GB/s|
|FP16 TFLOPS|34.08|59.39|125|
|INT8 TOPS (GPU)|34.08|59.39|250|
|INT8 TOPS (NPU)|38|50||
|Storage|1TB (non-upgradable) + 3 x TB5 (120Gbps)|2 x NVMe PCIe 4.0 x 4 + 2 x TB4 (40Gbps)|NVMe?|

- There's no M4 Max Mac Mini or Mac Studio, so the MBP 14 is the cheapest M4 Max config you can get
    
- HP is the only one I've seen showing off a [mini-PC workstation w/ the HP Z2 Mini G1a](https://liliputing.com/hp-z2-mini-g1a-is-a-workstation-class-mini-pc-with-amd-strix-halo-and-up-to-96gb-graphics-memory/). I'm using Liliputing's reporting on pricing, I'd guess the lower RAM models would be the starting price.
    
- Interested people might want to refer to my recent [llama.cpp efficiency tests](https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/) to see how that might translate to real world performance.
    

If the GB10 has a 512-bit bus (and hence 512GB/s of MBW) it's big FLOPS/TOPS advantage definitely puts it in a class of its own. If it merely matches Strix Halo on MBW, then it becomes a lot less interesting for the price...


Mac testing (Llama 7B Q4_0): [https://github.com/ggerganov/llama.cpp/discussions/4167](https://github.com/ggerganov/llama.cpp/discussions/4167)

- M2 Ultra (60 CU) does pp512/tg128 1013.81/88.64
    
- M2 Ultra (76 CU) does pp512/tg128 1238.48/94.27
    

So there apepars to be a tg advantage w/ more cores.

As for a 7900 XT, here's some testing I did a while back on a similar Llama2 7B Q4_0: [https://www.reddit.com/r/LocalLLaMA/comments/191srof/amd_radeon_7900_xtxtx_inference_performance/](https://www.reddit.com/r/LocalLLaMA/comments/191srof/amd_radeon_7900_xtxtx_inference_performance/)

tg is 96.6 t/s, so about on par. Note the RTX 3090 tg speed though. It has 936.2GB/s of MBW (+17%) but is +41% faster, so theoretical MBW doesn't tell the whole story (Nvidia cards have gotten even faster on the llama.cpp CUDA backend since then).

You can more recent analysis I did of t/TFLOP and theoretical MBW% efficiency numbers here: [https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/](https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/)


Power Efficiency
https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/comment/m4r84nl/