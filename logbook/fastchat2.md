Install
```
mamba create -n fastchat2 python=3.11
pip install pipx
pipx install yt-dlp
pipx install insanely-fast-whisper
```

Measure Memory usage:
```bash
max=0; while : ; do usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}' | sort -nr | head -n 1); if [ "$usage" -gt "$max" ]; then max=$usage; fi; echo $max; sleep 1; done
```

Get files
```
# 10 Minute 2 speaker JA
yt-dlp -f 'bestaudio' -x 'https://www.youtube.com/watch?v=_GZTDjcRyR0' -o 'ja-sample'

# 10 Minute multi-segment EN
yt-dlp -f 'bestaudio' -x 'https://www.youtube.com/watch?v=pT-Lm4DEqaw' -o 'en-sample'
```

# 4090

## insanely-fast-whisper (v3-large)
```
❯ time insanely-fast-whisper --file-name ja-sample.opus 
/home/local/.local/lib/python3.10/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
/home/local/.local/lib/python3.10/site-packages/torch_audiomentations/utils/io.py:27: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
🤗 Transcribing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:12
Voila!✨ Your file has been transcribed go check it out over here 👉 output.json

real    0m21.834s
user    0m38.873s
sys     1m0.199s
```
* 11243 MB

```
❯ time insanely-fast-whisper --file-name en-sample.opus 
/home/local/.local/lib/python3.10/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
/home/local/.local/lib/python3.10/site-packages/torch_audiomentations/utils/io.py:27: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
🤗 Transcribing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:08
Voila!✨ Your file has been transcribed go check it out over here 👉 output.json

real    0m17.828s
user    0m33.981s
sys     0m53.873s
```
- 10539 MB

## WhisperX
Install
```
pip install git+https://github.com/m-bain/whisperx.git
```

```
❯ time whisperx --model large-v3 --language ja ja-sample.opus 
/home/local/.conda/envs/fastchat2/lib/python3.11/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
/home/local/.conda/envs/fastchat2/lib/python3.11/site-packages/torch_audiomentations/utils/io.py:27: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
torchvision is not available - cannot save figures
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.1.3. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../.cache/torch/whisperx-vad-segmentation.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.1.2+cu121. Bad things might happen unless you revert torch to 1.x.
>>Performing transcription...
Ignored unknown kwarg option normalize
Ignored unknown kwarg option normalize
Ignored unknown kwarg option normalize
Ignored unknown kwarg option normalize
Some weights of the model checkpoint at jonatasgrosman/wav2vec2-large-xlsr-53-japanese were not used when initializing Wav2Vec2ForCTC: ['wav2vec2.encoder.pos_conv_embed.conv.weight_g', 'wav2vec2.encoder.pos_conv_embed.conv.weight_v']
- This IS expected if you are initializing Wav2Vec2ForCTC from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing Wav2Vec2ForCTC from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at jonatasgrosman/wav2vec2-large-xlsr-53-japanese and are newly initialized: ['wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1', 'wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
>>Performing alignment...

real    0m19.488s
user    0m22.797s
sys     0m8.183s
```
- 6625 MB

```
❯ time whisperx --model large-v3 --language en en-sample.opus 
/home/local/.conda/envs/fastchat2/lib/python3.11/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
/home/local/.conda/envs/fastchat2/lib/python3.11/site-packages/torch_audiomentations/utils/io.py:27: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
torchvision is not available - cannot save figures
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.1.3. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../.cache/torch/whisperx-vad-segmentation.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.1.2+cu121. Bad things might happen unless you revert torch to 1.x.
>>Performing transcription...
>>Performing alignment...

real    0m13.672s
user    0m14.951s
sys     0m7.084s

```
- 6273

## StyleTTS2
* 1723

# 7840HS
ROCm
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 -U --force-reinstall

# Put this in your .bashrc ideally
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## ROCm Support
insanely-fast-whisper - 

WhisperX: no https://github.com/OpenNMT/CTranslate2/issues/1072

distil:
- https://github.com/m-bain/whisperX/issues/558
- https://github.com/SYSTRAN/faster-whisper/issues/533
- https://github.com/SYSTRAN/faster-whisper/pull/557

# MLX
https://owehrens.com/whisper-nvidia-rtx-4090-vs-m1pro-with-mlx/
https://news.ycombinator.com/item?id=38628184
