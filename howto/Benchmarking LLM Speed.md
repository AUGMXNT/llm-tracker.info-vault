We don't use "performance" to avoid performance since in ML research the term performance typically refers to mearusing model quality/capabilities.

But what if you want to compare "drag race" performance?  Here's a cheat sheet.

# Background and Definitions

What we need to know about LLMs:
- LLMs are basically a big pile of numbers (matrices) They have different sizes, which is **parameter** count - When you see 7B, 8B, 14B, this is an approximate count of how many parameters (in billions).
- Quantization - Models weights (the parameters) used to be stored as FP32 (4 bytes), then FP16/BF16. Commercially, FP8 and INT8 are quite common, and FP4 and INT4 are emerging. At home, "Q4", which is roughly ~4-bit is used most often used, but there are even smaller quants that are usable these days (down to ~1.58b). Note, performance loss is not linear - a good Q4 quant can be close to or even
- Weights vs Activations - for quants, normally we are talking about the size and memory of weights, but new accelearators 

What we want to measure/compare
- Memory usage



