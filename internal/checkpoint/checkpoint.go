// Package checkpoint persists per-chunk Whisper results to a sidecar file
// as they complete, so a long transcription that is killed, OOMs, or loses
// its backend part-way through can resume instead of starting over. A
// 3h20m session that ran for hours and produced nothing because every
// chunk lived only in process memory is the failure this prevents.
//
// The sidecar is JSONL: a header line pinning the run fingerprint, then one
// line per completed chunk. Each Save fsyncs, so a result that returned is
// on disk before the next chunk starts. On Open, a sidecar whose
// fingerprint does not match the current run is discarded rather than
// reused -- different VAD params or a different model would make the old
// chunk boundaries and transcripts wrong. A torn trailing line (the process
// died mid-append) is skipped; every complete line before it survives.
package checkpoint

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/matthewjhunter/transcribe/internal/whisper"
)

// maxLine bounds a single JSONL record. Chunk results are a few KB at most
// (one VAD chunk is capped at ~28s of speech); this is headroom, not a real
// expectation.
const maxLine = 8 * 1024 * 1024

// Fingerprint derives a stable run identity from the parameters that change
// what a chunk's transcript would be -- model, language, and the VAD
// chunk-plan inputs. A sidecar written under one fingerprint is never
// reused under another.
func Fingerprint(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		_, _ = h.Write([]byte(p))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// record is one JSONL line. The header line carries Fingerprint (and
// Version) with no Key; every other line carries a Key and Result. The
// omitempty tags keep each shape to just its own fields.
type record struct {
	Version     int             `json:"version,omitempty"`
	Fingerprint string          `json:"fingerprint,omitempty"`
	Key         string          `json:"key,omitempty"`
	Result      *whisper.Result `json:"result,omitempty"`
}

// Store is an append-only checkpoint sidecar. It satisfies
// whisper.ChunkStore, so TranscribeChunks can load and save chunks through
// it directly.
type Store struct {
	path string

	mu     sync.Mutex
	f      *os.File
	loaded map[string]*whisper.Result
}

// Open returns a Store backed by the sidecar at path. If the file exists
// and its header fingerprint matches, its chunks are available via Load and
// new saves append. Otherwise (missing, stale, or corrupt header) the file
// is truncated and started fresh with a header for the current fingerprint.
func Open(path, fingerprint string) (*Store, error) {
	loaded := map[string]*whisper.Result{}
	reuse := false

	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		reuse = parseInto(data, fingerprint, loaded)
		if !reuse {
			loaded = map[string]*whisper.Result{}
		}
	case errors.Is(err, os.ErrNotExist):
		// fresh sidecar
	default:
		return nil, fmt.Errorf("checkpoint: read %q: %w", path, err)
	}

	var f *os.File
	if reuse {
		f, err = os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	} else {
		f, err = os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	}
	if err != nil {
		return nil, fmt.Errorf("checkpoint: open %q: %w", path, err)
	}

	s := &Store{path: path, f: f, loaded: loaded}
	if !reuse {
		if err := s.appendLocked(record{Version: 1, Fingerprint: fingerprint}); err != nil {
			_ = f.Close()
			return nil, err
		}
	}
	return s, nil
}

// parseInto reads a sidecar body into out and reports whether it may be
// reused. Reuse requires a first record that is the header and whose
// fingerprint matches. Any unparseable line (a crash's torn trailing write)
// is skipped rather than fatal.
func parseInto(data []byte, fingerprint string, out map[string]*whisper.Result) bool {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)

	headerSeen := false
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec record
		if err := json.Unmarshal(line, &rec); err != nil {
			continue // torn/partial line -- skip it, keep what came before
		}
		if !headerSeen {
			// The first complete record must be the header for this run.
			if rec.Key != "" || rec.Fingerprint != fingerprint {
				return false
			}
			headerSeen = true
			continue
		}
		if rec.Key != "" && rec.Result != nil {
			out[rec.Key] = rec.Result
		}
	}
	return headerSeen
}

// Load returns a previously saved result for key, or (nil, false).
func (s *Store) Load(key string) (*whisper.Result, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.loaded[key]
	return r, ok
}

// Save appends the result for key and fsyncs before returning, so a chunk
// that completed is durable before the next one starts.
func (s *Store) Save(key string, r *whisper.Result) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.f == nil {
		return errors.New("checkpoint: store is closed")
	}
	return s.appendLocked(record{Key: key, Result: r})
}

// appendLocked writes one JSONL line and fsyncs. Callers hold s.mu.
func (s *Store) appendLocked(rec record) error {
	b, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("checkpoint: marshal record: %w", err)
	}
	b = append(b, '\n')
	if _, err := s.f.Write(b); err != nil {
		return fmt.Errorf("checkpoint: write %q: %w", s.path, err)
	}
	if err := s.f.Sync(); err != nil {
		return fmt.Errorf("checkpoint: sync %q: %w", s.path, err)
	}
	return nil
}

// Count is the number of chunks resumable from a prior run.
func (s *Store) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.loaded)
}

// Close releases the file handle, leaving the sidecar in place so a later
// run can resume from it. Safe to call after Discard.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.f == nil {
		return nil
	}
	err := s.f.Close()
	s.f = nil
	return err
}

// Discard removes the sidecar; call it once the transcript is written so a
// successful run leaves nothing behind (mirroring the idempotent re-run of
// the extracted-audio sibling).
func (s *Store) Discard() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.f != nil {
		_ = s.f.Close()
		s.f = nil
	}
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("checkpoint: remove %q: %w", s.path, err)
	}
	return nil
}
