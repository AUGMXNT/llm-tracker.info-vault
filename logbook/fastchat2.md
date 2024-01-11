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


# MLX
https://owehrens.com/whisper-nvidia-rtx-4090-vs-m1pro-with-mlx/