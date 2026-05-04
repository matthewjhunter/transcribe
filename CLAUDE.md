# transcribe

CLI that produces a speaker-labeled transcript from an audio or video file by combining a Whisper backend (transcription + word timestamps) with sherpa-onnx (diarization + speaker segments), then aligning the two.

## Goal

Replace the existing `~/bin/transcribe` shell script (which shells out to WhisperX over SSH) with a single Go binary that runs locally on Strix Halo and produces output in the same format the OSG session pipeline already consumes.

## Pipeline

```
input file
   │
   ├─ ffprobe: video? → ffmpeg extract WAV (16 kHz mono PCM)
   │
   ├─ Whisper backend (Lemonade /v1/audio/transcriptions)
   │     → segments + word-level timestamps (verbose_json)
   │
   ├─ sherpa-onnx offline speaker diarization
   │     pyannote-segmentation-3.0 ONNX + 3D-Speaker embedding ONNX
   │     → list of (start, end, speaker_id) segments
   │
   └─ aligner: assign each Whisper word/segment to the speaker whose
              diarization segment overlaps it most (majority vote per
              transcript segment), emit `[HH:MM:SS.sss --> HH:MM:SS.sss]
              (SPEAKER_NN) text` lines compatible with WhisperX `--output_format txt`
```

## Backend choice

Whisper backend is OpenAI-compatible HTTP, configurable via flag/env:
- `--whisper-url` (default `http://localhost:13305/api/v1`)
- `--whisper-model` (default `Whisper-Large-v3`)
- `--whisper-api-key` (default empty; Lemonade does not require auth)

This means the same binary works against:
- **Lemonade on halo** (NPU when the `whispercpp` recipe loads the `*-vitisai.rai` cache)
- **whisper.cpp server** anywhere (`./server` mode)
- **OpenAI Whisper API** (just point the URL at `https://api.openai.com/v1` with a key)

Sherpa-onnx runs in-process via Go bindings (`github.com/k2-fsa/sherpa-onnx-go`). Models are downloaded on first run to `${XDG_CACHE_HOME:-$HOME/.cache}/transcribe/models/`.

## Non-goals (v0.1)

- No live/streaming mode (offline files only).
- No alternative ASR backends besides OpenAI-compatible HTTP.
- No GPU acceleration of the diarization stage; CPU sherpa-onnx is the target. Diarization is the bottleneck on Strix Halo CPU but acceptable.
- No internal HTTP API or job queue — that's a separate service-shaped project (`asrclient`-style) that could later wrap this CLI.

## Deferred / open questions

- Whether to enable Lemonade's NPU recipe is a server-side config decision, not a client one — out of scope here.
- Output format parity with WhisperX: the current consumer (OSG) uses the `.txt` form, which is `[time --> time] (SPEAKER_NN) text` per line. Match that exactly.
- VAD: WhisperX uses Silero VAD before transcription. Lemonade's whispercpp does its own internal VAD. For v0.1 we trust the backend; revisit if accuracy regresses.
- Word timestamps: required for alignment. Lemonade `whispercpp` returns segment + word timestamps with `response_format=verbose_json` — need to confirm and handle if any field is missing.

## Build

```bash
go build ./...
# or: task build
```

CGO is required because sherpa-onnx wraps a C++ library. The binary needs `libonnxruntime.so` (and the sherpa shared lib) on `LD_LIBRARY_PATH` at runtime, or set `CGO_LDFLAGS` for a static link if upstream supports it.

## Test

```bash
go test -race ./...
# or: task test
```

Unit tests for the aligner and the Whisper-response parser should not require external services. An integration test that hits a real Lemonade endpoint is opt-in via `-tags=integration` and reads `WHISPER_URL` from env.

## Conventions

- Single-dash short flags, double-dash long flags (Matthew's preference).
- Taskfile.yml is the build entrypoint, not Makefile.
- `cmd/transcribe/main.go` stays thin — orchestration only. All real logic in `internal/`.
- Match the existing WhisperX `.txt` output line format byte-for-byte where feasible so the OSG pipeline doesn't notice a backend swap.

## Related work

- `~/bin/transcribe` — the bash script this replaces (still in use; don't remove until parity is proven).
- `~/.local/share/whisperx/` — the Python WhisperX install on rainbow and halo. Reference for output format.
- `github.com/matthewjhunter/asrclient` — sibling Go module abstracting ASR backends for `dicta`. If a Whisper-HTTP backend lands there first, this tool should consume it rather than reimplementing.
