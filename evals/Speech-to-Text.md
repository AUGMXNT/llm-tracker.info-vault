# WhisperX
[WhisperX](https://github.com/m-bain/whisperX) is the current best version of Whisper.
```
conda create --name whisperx python=3.10
conda activate whisperx
pip install git+https://github.com/m-bain/whisperx.git

# Test File
wget https://github.com/ggerganov/whisper.cpp/blob/master/samples/jfk.wav

# Run
❯ time whisperx --no_align --output_format txt --task transcribe --language en jfk.wav
torchvision is not available - cannot save figures
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.0.5. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint --file ../../.cache/torch/whisperx-vad-segmentation.bin`
Model was trained with pyannote.audio 0.0.1, yours is 2.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.0.0. Bad things might happen unless you revert torch to 1.x.
>>Performing transcription...

real	0m6.604s
user	0m5.285s
sys	0m5.718s

❯ cat jfk.txt 
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

# SeamlessM4T
* [Bringing the world closer together with a foundational multimodal model for speech translation](https://ai.meta.com/blog/seamless-m4t/)
* [https://github.com/facebookresearch/seamless_communication/](https://github.com/facebookresearch/seamless_communication/)

The installation instructions are a bit incomplete. Here's how I got it running:
```
conda create -n seamless
git clone https://github.com/facebookresearch/seamless_communication
cd seamless_communication
pip install fairseq2==0.1
pip install .

# Missing
pip install torchaudio
mamba install libsndfile

# seamlessM4T_large is 10GB d/l

# Test File
wget https://github.com/ggerganov/whisper.cpp/blob/master/samples/jfk.wav

# Run
❯ time python scripts/m4t/predict/predict.py jfk.wav s2tt eng
INFO:__main__:Running inference on the GPU.
Using the cached checkpoint of the model 'seamlessM4T_large'. Set `force=True` to download again.
Using the cached tokenizer of the model 'seamlessM4T_large'. Set `force=True` to download again.
Using the cached checkpoint of the model 'vocoder_36langs'. Set `force=True` to download again.
INFO:__main__:Translated text in eng: And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

real	0m8.825s
user	0m15.527s
sys	0m9.225s

❯ time python scripts/m4t/predict/predict.py jfk.wav asr eng
INFO:__main__:Running inference on the GPU.
Using the cached checkpoint of the model 'seamlessM4T_large'. Set `force=True` to download again.
Using the cached tokenizer of the model 'seamlessM4T_large'. Set `force=True` to download again.
Using the cached checkpoint of the model 'vocoder_36langs'. Set `force=True` to download again.
INFO:__main__:Translated text in eng: And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

real	0m8.740s
user	0m15.413s
sys	0m9.285s
```