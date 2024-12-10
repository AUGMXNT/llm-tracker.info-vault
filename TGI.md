

Always neat to see competition. A couple notes:

- there's still no RDNA3/gfx1100 support sadly: https://github.com/huggingface/text-generation-inference/issues/2641
- I noticed that for speculative decode, only Medusa (requires training, if going that route maybe EAGLE-2 is better?) and ngram appear supported - in practice, vLLM's speculative decoding support might out the general speed advantage TGI might have (refer to https://github.com/hemingkx/Spec-Bench and https://github.com/hemingkx/SpeculativeDecodingPapers ; training-free methods like Ouroboros and SWIFT are particularly interesting)
- 