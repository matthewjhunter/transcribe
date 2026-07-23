package whisper

import (
	"fmt"

	"github.com/matthewjhunter/transcribe/internal/vad"
)

// ChunkStore persists and restores per-chunk transcription results so a
// long run survives a kill, OOM, or restart. Keys are opaque to the store;
// TranscribeChunks derives them from each chunk's time bounds via ChunkKey.
// Saved Results carry global (offset) timestamps, so a resumed chunk merges
// identically to a freshly transcribed one.
//
// Save is called concurrently -- one goroutine per in-flight chunk -- so
// implementations must be safe for concurrent use.
type ChunkStore interface {
	// Load returns a previously saved result for key, or (nil, false).
	Load(key string) (*Result, bool)
	// Save persists the completed, globally offset result for key.
	Save(key string, r *Result) error
}

// ChunkKey is the stable identity of a chunk within a run: its start and
// end offsets in nanoseconds. Guarding against reuse across incompatible
// runs (different VAD params, model, or language) is the checkpoint store's
// job, not the key's -- the store fingerprints the whole run.
func ChunkKey(ch vad.Chunk) string {
	return fmt.Sprintf("%d-%d", ch.Start.Nanoseconds(), ch.End.Nanoseconds())
}

type chunkOptions struct {
	store ChunkStore
}

// ChunkOption configures TranscribeChunks.
type ChunkOption func(*chunkOptions)

// WithChunkStore enables checkpoint/resume: each chunk is saved to the
// store as it completes, and any chunk already present is loaded instead of
// re-requested. A nil store is ignored, so callers can pass one
// unconditionally.
func WithChunkStore(s ChunkStore) ChunkOption {
	return func(o *chunkOptions) { o.store = s }
}
