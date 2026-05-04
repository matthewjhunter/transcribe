# transcribe

CLI that produces a speaker-labeled transcript from an audio or video file.

Designed to run end-to-end on a local workstation: Whisper transcription via an OpenAI-compatible HTTP backend (e.g. [Lemonade](https://lemonade-server.ai/)), speaker diarization via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), and word-level alignment of the two into a single output.

## Status

Early — see `CLAUDE.md` for the design plan.

## Usage (planned)

```bash
transcribe path/to/recording.mkv
# → path/to/recording.txt   (SPEAKER_00..SPEAKER_NN labels, like WhisperX)
```

## Why

The previous pipeline ran [WhisperX](https://github.com/m-bain/whisperX) on a CUDA host (rainbow, RTX 3080) over SSH. It works, but it pins the workflow to a remote machine, occupies VRAM that competes with Ollama, and ties speaker diarization to a PyTorch+pyannote stack that's awkward on AMD hardware.

This tool replaces that pipeline with components that run natively on a Strix Halo workstation (or any machine with a Lemonade-compatible Whisper endpoint and CPU/ONNX-capable diarization):

- **Transcription**: Lemonade's `whispercpp` recipe (XDNA2 NPU acceleration on Strix Halo, GPU/CPU elsewhere) via the OpenAI `/v1/audio/transcriptions` endpoint.
- **Diarization**: sherpa-onnx with pyannote segmentation + 3D-Speaker embedding ONNX models, CPU or DirectML.
- **Alignment**: in-process Go logic that merges word-timestamped Whisper output with speaker segments.

No PyTorch, no Python runtime at runtime, single binary (with sherpa-onnx shared library).
