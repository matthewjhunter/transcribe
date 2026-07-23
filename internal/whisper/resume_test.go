package whisper

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/vad"
)

// memStore is a trivial in-memory ChunkStore for exercising the
// resume/checkpoint path in TranscribeChunks without touching the disk.
type memStore struct {
	mu    sync.Mutex
	m     map[string]*Result
	saves int
}

func newMemStore() *memStore { return &memStore{m: map[string]*Result{}} }

func (s *memStore) Load(key string) (*Result, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.m[key]
	return r, ok
}

func (s *memStore) Save(key string, r *Result) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = r
	s.saves++
	return nil
}

func TestTranscribeChunks_ResumesStoredChunk(t *testing.T) {
	fs := &fakeServer{t: t, respond: func(c fakeCall) ([]byte, int) {
		return chunkRespond(indexFromName(c.fileName))
	}}
	srv := fs.start()
	defer srv.Close()

	const sr = 16000
	samples := make([]float32, sr*40)
	chunks := []vad.Chunk{
		{Start: 0, End: 5 * time.Second},
		{Start: 10 * time.Second, End: 15 * time.Second},
		{Start: 30 * time.Second, End: 35 * time.Second},
	}

	// Pretend a prior run already completed the middle chunk.
	store := newMemStore()
	store.m[ChunkKey(chunks[1])] = &Result{
		Text:     "resumed-1",
		Language: "english",
		Segments: []Segment{{Start: 10 * time.Second, End: 12 * time.Second, Text: "resumed-1"}},
		Words:    []Word{{Start: 10 * time.Second, End: 12 * time.Second, Text: "resumed-1"}},
	}

	c := New(Config{Endpoint: srv.URL})
	got, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 1, WithChunkStore(store))
	if err != nil {
		t.Fatalf("TranscribeChunks: %v", err)
	}

	// The resumed chunk must not have been fetched: 3 chunks, 2 calls.
	if len(fs.calls) != 2 {
		t.Fatalf("got %d HTTP calls, want 2 (middle chunk should resume)", len(fs.calls))
	}
	for _, cl := range fs.calls {
		if indexFromName(cl.fileName) == 1 {
			t.Errorf("chunk 1 was fetched (%q) but should have resumed from the store", cl.fileName)
		}
	}

	// Merged output threads the stored chunk in at its position.
	if got.Text != "text-0 resumed-1 text-2" {
		t.Errorf("merged Text = %q, want %q", got.Text, "text-0 resumed-1 text-2")
	}

	// The two freshly fetched chunks were persisted; the pre-existing one
	// was not re-saved.
	for i, ch := range chunks {
		if _, ok := store.Load(ChunkKey(ch)); !ok {
			t.Errorf("chunk %d (%s) missing from store after run", i, ChunkKey(ch))
		}
	}
	if store.saves != 2 {
		t.Errorf("store saw %d saves, want 2 (only the fetched chunks)", store.saves)
	}
}

func TestTranscribeChunks_NilStoreOptionIsIgnored(t *testing.T) {
	fs := &fakeServer{t: t, respond: func(c fakeCall) ([]byte, int) {
		return chunkRespond(indexFromName(c.fileName))
	}}
	srv := fs.start()
	defer srv.Close()

	const sr = 16000
	samples := make([]float32, sr*10)
	chunks := []vad.Chunk{{Start: 0, End: 2 * time.Second}}

	c := New(Config{Endpoint: srv.URL})
	if _, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 1, WithChunkStore(nil)); err != nil {
		t.Fatalf("TranscribeChunks with nil store: %v", err)
	}
	if len(fs.calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(fs.calls))
	}
}
