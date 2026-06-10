# transcribe

CLI that produces a speaker-labeled transcript from an audio or video file by combining a Whisper backend (transcription + word timestamps) with sherpa-onnx (diarization + speaker segments), then aligning the two.

## Goal

Replace the existing `~/bin/transcribe` shell script (which shells out to WhisperX over SSH) with a single Go binary that runs locally on Strix Halo and produces output in the same format the OSG session pipeline already consumes.

## Pipeline

```
input file
   │
   ├─ ffprobe → ffmpeg
   │     │
   │     ├─ video? stream-copy audio to sibling <input>.m4a/.opus/...
   │     │       (lossless, codec→container map; --no-extract-audio to skip)
   │     └─ extract canonical WAV (16 kHz mono PCM) into temp dir
   │
   ├─ Silero VAD (sherpa-onnx, in-process)
   │     → speech regions → chunk plan
   │       (greedy-merge gap < 0.5s, hard-cap 28s, drop < 1s)
   │       --no-vad falls back to whole-file submission
   │
   ├─ Whisper backend (Lemonade /v1/audio/transcriptions)
   │     one POST per VAD chunk, fresh decoder state per chunk,
   │     bounded concurrency (--whisper-concurrency)
   │     → segments + word-level timestamps (verbose_json)
   │     → offset to global time, merged in chunk order
   │
   ├─ sherpa-onnx offline speaker diarization (parallel with Whisper)
   │     pyannote-segmentation-3.0 ONNX + 3D-Speaker embedding ONNX
   │     runs on the whole file, not per chunk
   │     → list of (start, end, speaker_id) segments
   │
   ├─ voice labeling (pure DSP, no model)
   │     YIN F0 estimation per cluster → median F0 → M / F / ?
   │     thresholds: < 155 Hz → M, > 180 Hz → F, in-between → ?
   │     --no-label-gender to skip
   │
   └─ aligner: assign each Whisper word/segment to the speaker whose
              diarization segment overlaps it most (majority vote per
              transcript segment), stamp each line with the cluster's
              voice label, emit via the `internal/output` package
              (formats: `tstxt` default, `wxtxt`, `json`)
```

VAD is on by default because Whisper-Large-v3 hallucinates on long-form
conversational input — its `condition_on_previous_text` decoder bias can
latch onto a phrase and emit it for tens of minutes straight. Per-chunk
submission resets decoder state at every silence gap, bounding the
blast radius of any single hallucination loop to one chunk.

Voice labeling is on by default because diarization clusters are
anonymous — `SPEAKER_00` carries no identity beyond "consistent voice."
Tagging each cluster M/F/? halves the search space when manually mapping
clusters to known players. F0 from a YIN estimator is enough for two-class
on adult voices (~92–96% on clean conversational audio); no ML model,
no Python sidecar, no extra dependency. The `wxtxt` format intentionally
strips the label to preserve byte-for-byte WhisperX compatibility.

## Backend choice

Whisper backend is OpenAI-compatible HTTP, configurable via flag/env:
- `--whisper-url` (default `http://localhost:13305/api/v1`)
- `--whisper-model` (default `Whisper-Large-v3`)
- `--whisper-api-key` (default empty; Lemonade does not require auth)

This means the same binary works against:
- **Lemonade on halo** (NPU when the `whispercpp` recipe loads the `*-vitisai.rai` cache)
- **whisper.cpp server** anywhere (`./server` mode)
- **OpenAI Whisper API** (just point the URL at `https://api.openai.com/v1` with a key)

Sherpa-onnx runs in-process via Go bindings (`github.com/k2-fsa/sherpa-onnx-go`) for both VAD and diarization. Three ONNX model files are downloaded on first run to `${XDG_CACHE_HOME:-$HOME/.cache}/transcribe/models/`: `silero_vad.onnx`, `sherpa-onnx-pyannote-segmentation-3-0.onnx`, and the configured speaker-embedding model (`nemo_en_titanet_large.onnx` by default).

## Non-goals (v0.1)

- No live/streaming mode (offline files only).
- No alternative ASR backends besides OpenAI-compatible HTTP.
- No GPU acceleration of the diarization stage; CPU sherpa-onnx is the target. Diarization is the bottleneck on Strix Halo CPU but acceptable.
- No internal HTTP API or job queue — that's a separate service-shaped project (`asrclient`-style) that could later wrap this CLI.

## Output formats

The `.txt` is read by humans and LLM agents (e.g. the OSG `osg-session-notes` skill treats it as committed source material), not parsed by any tool — `grep -r SPEAKER_ ~/git/osg` returns no matches outside vendored code. So format choice is about reader ergonomics, not byte-level compatibility.

- **`tstxt`** (default) — `[HH:MM:SS] [SPEAKER_NN (X)]: text\n` when voice labeling is on, `[HH:MM:SS] [SPEAKER_NN]: text\n` when off. Recommended: timestamps let a human or LLM jump back to the source video for a specific moment in a multi-hour session; the `(M)`/`(F)`/`(?)` tag halves the candidate set when manually mapping clusters to players.
- **`wxtxt`** — `[SPEAKER_NN]: text\n`. Byte-for-byte match for WhisperX `--output_format txt --diarize` (verified against `~/Shadowmaze 2026-02-09.txt`). Voice labels are intentionally dropped here — WhisperX has no slot for them. Note: the WhisperX TXT format has *no* timestamps — the `[time --> time]` form is SRT/VTT, not TXT. Keep `wxtxt` only if some specific consumer requires byte-identical output.
- **`json`** — array of `{start, end, speaker, label?, text}` for programmatic consumers. The `label` field uses `omitempty`, so unlabeled output stays clean.

## Deferred / open questions

- Whether to enable Lemonade's NPU recipe is a server-side config decision, not a client one — out of scope here.
- Word timestamps: required for alignment. Lemonade `whispercpp` returns segment + word timestamps with `response_format=verbose_json` — need to confirm and handle if any field is missing.
- VAD chunk overlap + dedup: chunks are currently cut at silences with no overlap, on the assumption that VAD boundaries don't bisect words. Add an overlap window with word-level dedup if real output shows boundary artifacts.
- Back-channel preservation: short utterances (e.g. "yeah", "right") with > 0.5 s of silence on either side are below the merge threshold and below `--vad-min-chunk`, so they're dropped from the transcript. Diarization still attributes the time region to a speaker, but the line disappears. Lower thresholds rescue them at the cost of more requests and a higher loop-hallucination surface area.
- Voice labeling beyond binary M/F: F0 alone can cleanly separate adult M / adult F / child but says nothing useful about specific adult age. Adult-vs-child detection is straightforward to add (children have higher F0 and shorter vocal tracts) if it ever matters; specific-age estimation is a much harder problem we don't need.

## Build

```bash
go build ./...
# or: task build
```

CGO is required because sherpa-onnx wraps a C++ library. The default build
links the shared libs that ship inside the `sherpa-onnx-go-linux` module via
an rpath into the Go module cache, so the installed binary breaks when the
cache is cleaned.

The static build avoids that: `task build:static` (or `task install:static`)
runs `scripts/prepare-static.sh`, which downloads the sha256-pinned
`linux-x64-static-lib` archive from the sherpa-onnx release matching the
go.mod version, forks the binding into `build/sherpa-onnx-go-linux-static/`
with static link flags, and generates `go.static.mod` with a replace
directive. The default `go.mod` build is untouched. The build:static task
fails if `ldd` still shows sherpa, onnxruntime, or libstdc++. When bumping
sherpa-onnx-go, update `SHERPA_SHA256` in the script.

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
- The `wxtxt` format must remain byte-for-byte identical to WhisperX `--output_format txt --diarize` (i.e. `[SPEAKER_NN]: text\n`, no timestamps). Don't "fix" it by adding timestamps — that would silently break drop-in compatibility for any consumer that depends on the WhisperX line shape. Add new format constants instead.

## Related work

- `~/bin/transcribe.old` and `~/bin/extract-audio` — the retired bash scripts this replaces (renamed from `~/bin/transcribe`; kept for reference, no longer on PATH). The lossless audio-stream sibling output preserves the side effect of the old `extract-audio` helper.
- `~/.local/share/whisperx/` — the Python WhisperX install on rainbow and halo. Reference for output format. Sample output: `~/Shadowmaze 2026-02-09.txt`.
- `github.com/matthewjhunter/asrclient` — sibling Go module abstracting ASR backends for `dicta`. If a Whisper-HTTP backend lands there first, this tool should consume it rather than reimplementing.
