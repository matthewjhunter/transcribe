# transcribe

CLI that produces a speaker-labeled transcript from an audio or video file.

The tool runs end-to-end on a local workstation: Silero VAD pre-chunking, Whisper transcription via an OpenAI-compatible HTTP backend (e.g. [Lemonade](https://lemonade-server.ai/)), speaker diarization, and word-level alignment of the two — all in a single Go binary using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for the speech models. No PyTorch, no Python at runtime.

## Install

```bash
go install github.com/matthewjhunter/transcribe/cmd/transcribe@latest
```

CGO is required (sherpa-onnx wraps a C library; the prebuilt `libonnxruntime.so` and `libsherpa-onnx-c-api.so` ship with the bindings and are linked via rpath). Linux/amd64 is the only target wired up today.

`ffmpeg` and `ffprobe` must be on `$PATH` for input decoding.

## Usage

`--num-speakers` is required and intentionally has no default. The expected workflow is to count speakers from the recording (Roll20 attendees, Discord participants, etc.) and pass the exact number — wrong defaults produce wrong attribution, and clustering quality depends entirely on giving sherpa the right K.

```bash
# Standard run on a video:
#   - writes <input>.txt with [HH:MM:SS] [SPEAKER_NN]: text
#   - writes <input>.m4a (lossless audio stream copy from the source)
transcribe --num-speakers 4 path/to/recording.mkv

# Pin to a specific Lemonade host
transcribe --whisper-url http://halo:13305/api/v1 --num-speakers 4 session.mkv

# Match the historical WhisperX `[SPEAKER_NN]: text` format byte-for-byte
transcribe --output-format wxtxt --num-speakers 4 session.mkv

# Stream the transcript to stdout (suppresses the sibling audio file)
transcribe --num-speakers 4 -o - session.mkv

# Skip diarization entirely (no --num-speakers needed)
transcribe --no-diarize session.mkv

# Structured output for downstream pipelines
transcribe --num-speakers 4 --output-format json session.mkv

# Disable VAD pre-chunking (single whole-file POST; not recommended for
# long-form Whisper-Large-v3 due to hallucination loops)
transcribe --no-vad --num-speakers 4 session.mkv
```

Run `transcribe -h` for the full flag list.

## Side outputs

For video inputs, `transcribe` writes a sibling lossless audio file next to the transcript using `ffmpeg -vn -acodec copy` — no re-encoding, just a stream copy into a container matching the source codec (aac → m4a, opus → opus, mp3 → mp3, vorbis → ogg, flac → flac, alac → m4a, ac3/eac3 → ac3/eac3, wmav → wma, pcm → wav). Existing siblings are skipped, so re-runs are idempotent. Use `--no-extract-audio` to disable.

The sibling tracks the transcript's location: alongside the input by default, alongside `--output` when set, suppressed entirely for stdout output.

## VAD pre-chunking

Whisper-Large-v3 has a well-known long-form failure mode where the decoder's `condition_on_previous_text` bias latches onto a phrase and emits it for tens of minutes straight. The default pipeline pre-chunks audio at silence boundaries with [Silero VAD](https://github.com/snakers4/silero-vad) and submits each chunk as an independent request, so each chunk gets fresh decoder state. A loop on one chunk cannot poison adjacent chunks.

| Flag | Default | Behavior |
|---|---|---|
| `--no-vad` | off | Single whole-file POST (the historical behavior). |
| `--vad-min-silence` | 0.5 s | Merge speech regions whose silence gap is shorter than this. |
| `--vad-max-chunk` | 28 s | Hard cap on chunk length sent to the backend. Whisper's encoder window is 30 s. |
| `--vad-min-chunk` | 1 s | Drop chunks shorter than this after merging. |
| `--vad-model` | auto-cache | Path to `silero_vad.onnx`. |
| `--whisper-concurrency` | 1 | Parallel transcription requests. Lemonade's whispercpp serializes server-side, so >1 mostly helps against api.openai.com. |

## Models

On first run, the tool downloads ONNX files into `${XDG_CACHE_HOME:-$HOME/.cache}/transcribe/models/`:

| File | Source | Purpose | When fetched |
|---|---|---|---|
| `silero_vad.onnx` | sherpa-onnx releases | Silero VAD for pre-chunking | Default; skipped with `--no-vad` |
| `sherpa-onnx-pyannote-segmentation-3-0.onnx` | sherpa-onnx releases (extracted from `.tar.bz2`) | Pyannote 3.0 speaker segmentation | Default; skipped with `--no-diarize` |
| `nemo_en_titanet_large.onnx` | sherpa-onnx releases | NeMo TitaNet (large) English speaker embeddings, default | Default; skipped with `--no-diarize` |

The default uses TitaNet *large* because the *small* variant fails to distinguish similar adult voices on conversational audio (e.g. two male players are often merged into one cluster). Large is ~95 MB and runs ~3× slower than small, but produces materially better attribution. Switch back with `--embedding-preset titanet_small` if speed matters more than accuracy.

Override individual paths with `--vad-model` / `--segmentation-model` / `--embedding-model`.

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

These were measured before VAD pre-chunking landed; per-chunk submission adds some HTTP overhead but is offset (or beaten) by `--whisper-concurrency > 1` on backends that support concurrent requests. Whisper still runs in parallel with diarization. Whisper time is dominated by network + backend; diarization runs CPU-locally and is the slower of the two on long inputs.

## Why this exists

Replaces a `~/bin/transcribe` shell script that ssh'd into a remote CUDA box to run WhisperX. The new tool keeps everything on the local workstation, removes the PyTorch+CUDA dependency, runs offline-first, and produces output that drops directly into the existing OSG session-notes pipeline. The Silero-VAD pre-chunking and lossless audio sibling output mirror behaviors the old WhisperX + `extract-audio` workflow had — neither was free, and skipping them turned out to be a regression.

## License

MIT.
