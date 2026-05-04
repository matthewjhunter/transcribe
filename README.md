# transcribe

CLI that produces a speaker-labeled transcript from an audio or video file.

The tool runs end-to-end on a local workstation: Whisper transcription via an OpenAI-compatible HTTP backend (e.g. [Lemonade](https://lemonade-server.ai/)), speaker diarization via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), and word-level alignment of the two — all in a single Go binary. No PyTorch, no Python at runtime.

## Install

```bash
go install github.com/matthewjhunter/transcribe/cmd/transcribe@latest
```

CGO is required (sherpa-onnx wraps a C library; the prebuilt `libonnxruntime.so` and `libsherpa-onnx-c-api.so` ship with the bindings and are linked via rpath). Linux/amd64 is the only target wired up today.

`ffmpeg` and `ffprobe` must be on `$PATH` for input decoding.

## Usage

`--num-speakers` is required and intentionally has no default. The expected workflow is to count speakers from the recording (Roll20 attendees, Discord participants, etc.) and pass the exact number — wrong defaults produce wrong attribution, and clustering quality depends entirely on giving sherpa the right K.

```bash
# Standard run: writes <input>.txt with [HH:MM:SS] [SPEAKER_NN]: text
transcribe --num-speakers 4 path/to/recording.mkv

# Pin to a specific Lemonade host
transcribe --whisper-url http://halo:13305/api/v1 --num-speakers 4 session.mkv

# Match the historical WhisperX `[SPEAKER_NN]: text` format byte-for-byte
transcribe --output-format wxtxt --num-speakers 4 session.mkv

# Stream to stdout
transcribe --num-speakers 4 -o - session.mkv

# Skip diarization entirely (no --num-speakers needed)
transcribe --no-diarize session.mkv

# Structured output for downstream pipelines
transcribe --num-speakers 4 --output-format json session.mkv
```

Run `transcribe -h` for the full flag list.

## Models

On first run with diarization enabled, the tool downloads two ONNX files into `${XDG_CACHE_HOME:-$HOME/.cache}/transcribe/models/`:

| File | Source | Purpose |
|---|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0.onnx` | sherpa-onnx releases (extracted from `.tar.bz2`) | Pyannote 3.0 speaker segmentation |
| `nemo_en_titanet_large.onnx` | sherpa-onnx releases | NeMo TitaNet (large) English speaker embeddings, default |

The default uses TitaNet *large* because the *small* variant fails to distinguish similar adult voices on conversational audio (e.g. two male players are often merged into one cluster). Large is ~95 MB and runs ~3× slower than small, but produces materially better attribution. Switch back with `--embedding-preset titanet_small` if speed matters more than accuracy.

Override individual paths with `--segmentation-model` / `--embedding-model`.

## Sherpa knobs

All sherpa-onnx diarization parameters are exposed for experimentation:

| Flag | Default | Maps to |
|---|---|---|
| `--num-speakers` | (required) | `FastClusteringConfig.NumClusters`. Number of distinct speakers in the recording. |
| `--min-speech-duration` | 0 | `OfflineSpeakerDiarizationConfig.MinDurationOn` (seconds). Drops short voice-activity segments at the segmenter. |
| `--min-silence-duration` | 0 | `OfflineSpeakerDiarizationConfig.MinDurationOff` (seconds). Merges adjacent speech across short silences. |
| `--diarize-threads` | 0 (NumCPU) | Threadpool size for sherpa's segmentation and embedding stages. |
| `--diarize-provider` | `""` (cpu) | ONNX execution provider (`cpu`, `cuda`, ...). |
| `--embedding-preset` | `titanet_large` | `titanet_small` (~22 MB, fast, low accuracy on similar voices) or `titanet_large` (~95 MB, default). |
| `--segmentation-model` | auto-cache | Path to pyannote segmentation ONNX. |
| `--embedding-model` | auto-cache | Path to speaker-embedding ONNX. Bypasses `--embedding-preset` when set. |

## Backends

`--whisper-url` accepts any OpenAI-compatible `/v1` base. Tested with:

- **Lemonade** (`whispercpp` recipe): default. NPU-accelerated on Strix Halo when the `whisper-large-v3-encoder-vitisai.rai` cache is loaded.
- **whisper.cpp server**: works as-is.
- **OpenAI Whisper API**: pass `--whisper-url https://api.openai.com/v1` and `--whisper-api-key $OPENAI_API_KEY`.

The Whisper response shape is normalized: top-level `words[]` (OpenAI) and nested `segments[].words[]` (Lemonade) are both accepted.

## Output format

| Format | Default | Layout |
|---|---|---|
| `tstxt` | yes | `[HH:MM:SS] [SPEAKER_NN]: text` |
| `wxtxt` | | `[SPEAKER_NN]: text` (WhisperX byte-for-byte) |
| `json` | | `[{start, end, speaker, text}, ...]` |

## Performance

Indicative timing on Matthew's workstation against Lemonade-on-halo with `Whisper-Large-v3-Turbo` and the default `titanet_large` embedding:

| Input length | End-to-end |
|---|---|
| 30 s | ~5 s |
| 5 min | ~23 s |

Whisper runs in parallel with diarization. Whisper time is dominated by network + backend; diarization runs CPU-locally and is the slower of the two on long inputs.

## Why this exists

Replaces a `~/bin/transcribe` shell script that ssh's into a remote CUDA box to run WhisperX. The new tool keeps everything on the local workstation, removes the PyTorch+CUDA dependency, runs offline-first, and produces output that drops directly into the existing OSG session-notes pipeline.

## License

MIT.
