## [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models<svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewbox="0 0 16 16" width="16"></svg>](https://github.com/yl4579/StyleTTS2#styletts-2-towards-human-level-text-to-speech-through-style-diffusion-and-adversarial-training-with-large-speech-language-models)
* Samples: [https://styletts2.github.io/](https://styletts2.github.io/) 
* Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691) 
* Repo: [https://github.com/yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)

StyleTTS 2 is very appealing since the quality is very high and it's also flexible, supporting multi-speaker, zero-shot speaker adaptation, speech expressiveness, and style transfer (speech and style vectors are separated).

It also turns out the inferencing code appears to be very fast, beating out [TTS VITS](https://tts.readthedocs.io/en/latest/models/vits.html) by a big margin (and [XTTS](https://coqui.ai/blog/tts/open_xtts) by an even bigger margin). Note, all of these generate faster than real-time on an RTX 4090, although for StyleTTS 2, I'm seeing up to 95X and XTTS is barely faster at about 1.4X.

This write-up is done on the first day after release, and only adapting [the LJSpeech inferencing ipynb code](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb) to a Python script. The instructions [weren't in too bad a state](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb). You can see this post also for [a quick comparison of StyleTTS 2 vs TTS VITS vs TTS XTTS output](https://fediverse.randomfoo.net/notice/AaOgprU715gcT5GrZ2).

Recommended System Pre-requisites
* `espeak-ng` - you need this
* [CUDA](https://llm-tracker.info/books/howto-guides/page/nvidia-gpus#bkmrk-cuda-version-hell) - you could probably use CPU or ROCm but idk
* [Mamba](https://github.com/conda-forge/miniforge#mambaforge) - not required but will make your life a lot easier

Environment setup:
```
# you may need 3.10, depends on your pytorch version
mamba create -n styletts2 python=3.11
mamba activate styletts2

# pytorch - current nightly works w/ Python 3.11 but not 3.12
# pick your version here: https://pytorch.org/get-started/locally/
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

# reqs - torch stuff already installed 
pip install SoundFile munch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing typing-extensions git+https://github.com/resemble-ai/monotonic_align.git

# checkout codebase
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```

Get models:
```
# way useful for servers
pip install gdown
gdown 'https://drive.google.com/file/d/1K3jt1JEbtohBLUA0X75KLw36TW7U1yxq/view?usp=sharing'
unzip Models.zip
```

Inferencing
* Well, just use [Inference_LJSpeech.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb) basically. Latest version should work.

My changes, mainly include doing file output:
```
## No
# import IPython.display as ipd
# display(ipd.Audio(wav, rate=24000))

## Yes
import soundfile as sf
sf.write('output.df5.wav', wav, 24000)
```

Oh, and I like to output some more timing stuff, eg:
```
end = time.time()
rtf = (end - start) / (len(wav) / 24000)
print(f"Clip duration: {len(wav)/24000:.2f}s")
print(f"Inference time: {end-start:.2f}s")
print(f"RTF = {rtf:5f}")
print(f"RTX = {1/rtf:.2f}")
```

Personally, I find RT X multiple more intuitive, especially once you get to higher multiples.

To be continued when I have a chance to get to training...