# Cross-file speaker identification

## Problem

Diarization assigns anonymous, file-local cluster IDs (`SPEAKER_00`,
`SPEAKER_01`, …). The IDs are unstable across runs: the same person who
is `SPEAKER_02` in one session can be `SPEAKER_00` in the next, and the
`(M)` / `(F)` voice label only halves the search space. For OSG-style
recurring sessions with the same 4–5 participants, manually mapping
clusters → players every time is wasted work.

What we want: when the diarizer emits a cluster whose voice matches a
known player, the transcript line says `[matt]:` instead of
`[SPEAKER_02]:`. Unknown voices stay anonymous.

## Background: what diarization already gives us

`internal/diarize` runs sherpa-onnx with a pyannote segmentation model
plus a speaker-embedding model (NeMo TitaNet by default). The embedding
model produces a fixed-length vector per audio window — exactly the
input speaker *recognition* needs. Today we let the diarizer cluster
those vectors internally and then discard them; only `(start, end,
speaker_id)` tuples survive.

The sherpa-onnx Go binding (v1.13.0) exposes the embedding primitives
independently of the diarizer:

- `sherpa.SpeakerEmbeddingExtractor` — load the embedding ONNX once,
  call `Compute(samples) []float32` on any 16 kHz mono slice.
- `sherpa.SpeakerEmbeddingManager` — registers named vectors and does
  cosine-similarity search, with the same C++ implementation the
  diarizer uses internally.

So nothing here requires forking sherpa or reaching into private state.
We re-use what's already loaded and run a few extra forward passes per
file.

## Proposal

Add cross-file speaker identification as a post-diarization step,
gated behind enrolled voice prints stored on disk.

```
diarization → list of (start, end, cluster_id) turns
              │
              ├─ for each cluster:
              │     pick representative audio (concatenate cluster
              │     turns, cap at N seconds)
              │     → embedding vector
              │
              ├─ match each cluster embedding against enrolled prints
              │     (cosine similarity; argmax above threshold)
              │
              ├─ resolve collisions (two clusters → same name): keep
              │     the closer match, leave the other anonymous, log
              │     a warning (likely over-split cluster)
              │
              └─ produce a `cluster_id → display_name` map; aligner
                 uses display_name in output if present, falls back to
                 SPEAKER_NN if not
```

Voice labeling already runs per-cluster post-diarization, so the
plumbing pattern is established (`internal/voicelabel`). Speaker ID
follows the same shape.

### Why post-diarization, not replacing it

Tempting alternative: skip clustering, embed every segment, match each
one against the enrolled set. Don't do this:

- Loses sherpa's clustering quality on segments where the enrolled set
  doesn't match (visiting player, brief crosstalk).
- Embeddings on short segments (<1.5 s) are noisier than cluster
  centroids; clustering averages out that noise for free.
- We'd lose the `SPEAKER_NN` fallback for unknown voices.

Clustering first, then identifying clusters, is strictly more robust.

## New package: `internal/spkid`

Mirrors `internal/voicelabel` in shape:

```go
package spkid

type Profile struct {
    Name      string    // display name, e.g. "matt"
    Embedding []float32 // mean of enrollment-clip embeddings
    Created   time.Time
    Source    string    // path or note describing the enrollment audio
}

type Store struct { /* path-backed; JSON on disk */ }

func LoadStore(path string) (*Store, error)
func (s *Store) Add(p Profile) error
func (s *Store) List() []Profile
func (s *Store) Save() error

type Identifier struct { /* wraps sherpa.SpeakerEmbeddingExtractor */ }

func NewIdentifier(modelPath string, threads int) (*Identifier, error)
func (i *Identifier) Embed(samples []float32, sampleRate int) ([]float32, error)
func (i *Identifier) Close() error

type Match struct {
    Name       string  // empty if no match above threshold
    Similarity float64 // cosine, [-1, 1]
}

// IdentifyClusters mirrors voicelabel.ClassifyClusters.
// Returns map[clusterID]Match. Caller decides what to do with empty Name.
func IdentifyClusters(
    samples []float32,
    sampleRate int,
    turns []diarize.Turn,
    extractor *Identifier,
    profiles []Profile,
    opts IdentifyOptions,
) (map[int]Match, error)

type IdentifyOptions struct {
    Threshold       float64       // cosine threshold; default 0.5
    MaxClusterAudio time.Duration // cap input per cluster; default 30 s
    MinClusterAudio time.Duration // skip clusters with less; default 2 s
}
```

The store file path defaults to `${XDG_CONFIG_HOME:-$HOME/.config}/transcribe/speakers.json`.

### Storage format

```json
{
  "version": 1,
  "embedding_model": "nemo_en_titanet_large",
  "embedding_dim": 192,
  "profiles": [
    {
      "name": "matt",
      "vector": [/* 192 floats */],
      "created": "2026-05-06T14:30:00Z",
      "source": "matt-2026-04-12.wav (0:00-0:30)"
    }
  ]
}
```

Profiles are tied to a specific embedding model: vectors from TitaNet
and 3D-Speaker are not interchangeable. The `embedding_model` field
gates loading — mismatch is a hard error, not a silent re-run, because
silently producing garbage matches is worse than failing loudly.

## CLI surface

### Enrollment subcommand

```
transcribe enroll --name matt clip.wav [clip2.wav ...]
```

Behavior:

- Loads the configured embedding model (same `--embedding-preset` /
  `--embedding-model` flags as the main pipeline).
- For each clip: extract canonical 16 kHz mono PCM (re-use
  `internal/audio`), embed, accumulate.
- Mean-normalizes across clips, writes/updates the `Profile` in the
  store.
- `--store <path>` overrides the default location.
- `--replace` overwrites an existing profile of the same name; default
  is to error if the name exists.

Recommended enrollment: 10–30 s of clean speech per person, ideally
from the same session-recording setup (mic, room) you'll be
transcribing. A single clip is fine; the helper just makes
multi-sample averaging easy.

### Main pipeline flag

```
--identify-speakers          # opt-in; loads default store path
--speakers-file <path>       # override store path
--identify-threshold <f>     # cosine threshold; default 0.5
--no-identify-speakers       # explicit opt-out (e.g. when env enables it)
```

Off by default. If on and the store is missing or empty: warn, continue
with anonymous IDs. If on and `--embedding-preset` doesn't match the
store's `embedding_model`: hard error.

## Output format changes

The on-disk text formats already render speaker as a string field. The
change is: when an identified name is present, render the name; else
render `SPEAKER_NN`.

- **`tstxt`** — `[HH:MM:SS] [matt (M)]: text` when identified.
  Voice label still tags the cluster (sanity check; if matt's enrolled
  print is M but a misidentified cluster comes back F, the label warns
  the reader).
- **`wxtxt`** — `[matt]: text`. WhisperX's `--diarize` output also
  supports named speakers (it prints whatever string the alignment
  assigns), so this is still byte-shape compatible. Confirm against a
  WhisperX run before declaring victory.
- **`json`** — `align.SpeakerLine` already has `Speaker int`. Add
  `Name string` (omitempty). The integer cluster ID stays for tooling
  that wants the raw clustering result.

Concretely, `align.SpeakerLine` grows one field:

```go
type SpeakerLine struct {
    Start, End time.Duration
    Speaker    int          // cluster ID; unchanged
    Name       string       // identified name; empty if unmatched
    Label      voicelabel.Label
    Text       string
}
```

The aligner takes `map[int]string` (clusterID → name) and stamps
`Name` during line construction.

## Algorithm details

### Embedding a cluster

Reuse `voicelabel.collectClusterAudio`'s strategy: concatenate the
cluster's turns up to `MaxClusterAudio` (default 30 s). One forward
pass through the embedding model produces one vector. (TitaNet
internally pools across the input window, so passing ≥3 s of voiced
audio is fine; the model isn't fixed-length.)

### Matching

- Cosine similarity against every enrolled profile.
- Pick argmax; accept if `>= Threshold` (default 0.5).
- TitaNet on clean 16 kHz speech: same speaker pairs typically score
  0.55–0.85; cross-speaker pairs 0.05–0.35. 0.5 is a conservative
  floor; tune empirically against real session audio before locking
  it in.

### Collision resolution

If two clusters in one file both top-match `matt`:

1. Keep the higher-similarity match as `matt`.
2. The other cluster goes back to `SPEAKER_NN` — *not* renamed to the
   second-best name. Forcing a wrong name is worse than admitting
   uncertainty.
3. Log: `cluster 02 also matched matt (0.71); kept cluster 01 (0.78)`.
   Almost always a sign of an over-split cluster (sherpa created two
   clusters for one person); the operator can rerun with adjusted
   `--num-speakers` if they care.

We are *not* auto-merging the clusters in this pass. Merging would
require rewriting alignment outputs and changes the contract that
cluster IDs come straight from sherpa. Defer to a separate "smart
merge" feature if it ever proves necessary.

## Performance

Each diarization run today: one segmentation pass + N embedding passes
(once per pyannote-segmentation window, internally). Adding speaker ID:

- One extra `Embed` call per cluster (4–5 calls per session). TitaNet
  forward pass on ~30 s of audio is well under a second on CPU.
- Cosine similarity against ≤10 enrolled profiles is free.

Negligible compared to diarization itself, which already takes minutes
on Strix Halo CPU for a multi-hour recording.

## Test plan

Unit:

- `spkid.Identifier` round-trip: same audio → similar vectors
  (cosine > 0.95).
- `IdentifyClusters` with two known profiles and synthetic cluster
  audio (real recordings of the test author, checked in or downloaded
  on first test run from a git-LFS or HTTP cache — *not* committed
  raw to the repo).
- Threshold edge cases: similarity exactly at threshold; all matches
  below; collision where two clusters claim the same name.
- Store load/save round-trip; mismatched embedding model rejected.

Integration (build tag `integration`):

- End-to-end on a real OSG recording with 4 enrolled players. Assert
  cluster→name mapping is stable across two consecutive runs.

Hold off on a regression suite of "1000 voices, 95% accuracy" — we
don't need that scale, and benchmarking sherpa's embedder is upstream's
job.

## Risks and open questions

- **Mic/room change degrades matching.** TitaNet is reasonably robust
  but not invariant. Mitigation: re-enroll when the recording setup
  changes; document it. Not solving it in v0.1.
- **Enrollment audio quality.** Garbage in, garbage out. The enroll
  subcommand should reject < 3 s of input and warn on long silences.
- **Two-person overlap** within a single cluster (sherpa over-merges)
  produces a blended embedding that won't match either enrolled
  profile well. We accept this — falls back cleanly to `SPEAKER_NN`.
- **Embedding-model coupling.** Profiles enrolled against TitaNet are
  invalid if the user later switches to 3D-Speaker. Hard error on
  mismatch is the right call; alternative would be auto-re-enrollment,
  but we don't have the source clips after enrollment unless we save
  them, and we shouldn't.
- **Privacy (local CLI).** `speakers.json` contains voice-embedding
  vectors. Trivially portable within the TitaNet model family — the
  vector is the credential, no proprietary format. Within scope for
  v0.1: store at `~/.config/transcribe/speakers.json` mode 0600, brief
  note in README that the file is enrollment data and shouldn't be
  shared. The hosted-deployment threat model (below) is materially
  different and is not v0.1's problem.

## Out of scope (v0.1)

- Auto-enrollment from labeled past transcripts (cool, but needs a UI
  and a confidence model we haven't built).
- Federated / multi-user voice-print stores.
- Real-time / streaming identification.
- Active learning ("this cluster wasn't matched but the operator typed
  a name; remember it"). Easy follow-up; not blocking.

## Implementation phasing

1. **Spike**: write `spkid.Identifier` + a CLI throwaway that prints
   cosine similarity between two audio files. Validate that
   same-speaker > 0.5 and cross-speaker < 0.4 on real OSG audio
   *before* designing the rest. If TitaNet doesn't separate cleanly
   on Matthew's recording setup, the whole feature is moot.
2. `spkid` package: store, profile, identifier.
3. `transcribe enroll` subcommand.
4. Wire `IdentifyClusters` into the main pipeline behind
   `--identify-speakers`. Plumb `Name` through `align.SpeakerLine`
   and the three renderers.
5. Integration test on a real session recording.
6. Update README + SECURITY.md with privacy note.

Each phase is independently shippable. Phase 1 is the gate — if it
fails, stop and reconsider the whole approach.
