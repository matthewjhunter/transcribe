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

```bash
# Default: writes <input>.txt next to the input, with [HH:MM:SS] [SPEAKER_NN]: text
transcribe path/to/recording.mkv

# Pass --num-speakers when you know the count — clustering quality is much better
# than auto-discovery on real conversational audio
transcribe --num-speakers 4 path/to/recording.mkv

# Pin to a specific Lemonade host
transcribe --whisper-url http://halo:13305/api/v1 path/to/recording.mkv

# Match the historical WhisperX `[SPEAKER_NN]: text` format byte-for-byte
transcribe --output-format wxtxt path/to/recording.mkv

# Stream to stdout
transcribe -o - path/to/recording.mkv

# Skip diarization entirely
transcribe --no-diarize path/to/recording.mkv

# Structured output for downstream pipelines
transcribe --output-format json path/to/recording.mkv
```

Run `transcribe -h` for the full flag list.

## Models

On first run with diarization enabled, the tool downloads two ONNX files into `${XDG_CACHE_HOME:-$HOME/.cache}/transcribe/models/`:

| File | Source | Purpose |
|---|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0.onnx` | sherpa-onnx releases (extracted from `.tar.bz2`) | Pyannote 3.0 speaker segmentation |
| `nemo_en_titanet_small.onnx` | sherpa-onnx releases | NeMo TitaNet English speaker embeddings |

Override either with `--segmentation-model` / `--embedding-model` to use different models without touching the cache.

## Speaker count

Sherpa-onnx's default threshold-based clustering over-segments real-world conversational audio — a session with 4 actual people can produce 20+ "speakers" because the embedder distinguishes voice within the same speaker too aggressively when audio is noisy. To absorb that, the tool soft-targets a cluster count and folds low-duration "speakers" into their temporal neighbors after diarization.

Defaults:

| Flag | Default | Meaning |
|---|---|---|
| `--num-speakers` | 0 | Force exactly N. 0 = soft target. |
| `--target-speakers` | 4 | Aim for ~N distinct speakers. 0 disables the post-merge. |
| `--speaker-tolerance` | 2 | Allow up to `target+tolerance` distinct speakers. Anything past that cap gets merged. |
| `--speaker-threshold` | 0.5 | Sherpa clustering threshold (only used when `--num-speakers=0`). |

Tuning:

- **Different group size?** `transcribe --target-speakers 6 session.mkv` (or any other count).
- **Need exact N speakers?** `transcribe --num-speakers 4 session.mkv` — bypasses soft-merge entirely.
- **Want the raw clustering output?** `transcribe --target-speakers 0 session.mkv`.

The per-run log shows the merge: `diarization complete turns=75 raw_speakers=28` followed by `merged sparse speakers before=28 after=6`.

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

Indicative timing on Matthew's workstation against Lemonade-on-halo with `Whisper-Large-v3-Turbo`:

| Input length | End-to-end |
|---|---|
| 30 s | 3 s |
| 5 min | 15 s |

Whisper time is dominated by network + backend; diarization runs CPU-locally.

## Why this exists

Replaces a `~/bin/transcribe` shell script that ssh's into a remote CUDA box to run WhisperX. The new tool keeps everything on the local workstation, removes the PyTorch+CUDA dependency, runs offline-first, and produces output that drops directly into the existing OSG session-notes pipeline.

## License

MIT.
