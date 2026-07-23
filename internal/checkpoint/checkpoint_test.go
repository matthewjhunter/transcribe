package checkpoint

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/whisper"
)

func sidecar(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "session.mkv.transcribe-progress.jsonl")
}

func sampleResult(text string) *whisper.Result {
	return &whisper.Result{
		Text:     text,
		Language: "english",
		Duration: 5 * time.Second,
		Segments: []whisper.Segment{{Start: 10 * time.Second, End: 12 * time.Second, Text: text}},
		Words:    []whisper.Word{{Start: 10 * time.Second, End: 11 * time.Second, Text: text}},
	}
}

func TestStore_SaveThenReopenLoads(t *testing.T) {
	path := sidecar(t)
	const fp = "fingerprint-A"

	s, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if s.Count() != 0 {
		t.Fatalf("fresh store Count = %d, want 0", s.Count())
	}
	if err := s.Save("0-5", sampleResult("hello")); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen with the same fingerprint: the saved chunk is restored,
	// timestamps and text intact.
	s2, err := Open(path, fp)
	if err != nil {
		t.Fatalf("reopen Open: %v", err)
	}
	defer s2.Close()
	if s2.Count() != 1 {
		t.Fatalf("reopened Count = %d, want 1", s2.Count())
	}
	got, ok := s2.Load("0-5")
	if !ok {
		t.Fatal("Load(0-5) not found after reopen")
	}
	if got.Text != "hello" {
		t.Errorf("Text = %q, want hello", got.Text)
	}
	if len(got.Words) != 1 || got.Words[0].Start != 10*time.Second {
		t.Errorf("word timestamps not round-tripped: %+v", got.Words)
	}
}

func TestStore_FingerprintMismatchDiscards(t *testing.T) {
	path := sidecar(t)

	s, err := Open(path, "fingerprint-A")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Save("0-5", sampleResult("stale")); err != nil {
		t.Fatalf("Save: %v", err)
	}
	s.Close()

	// A run with different VAD params / model presents a different
	// fingerprint; the old sidecar's entries must not be reused.
	s2, err := Open(path, "fingerprint-B")
	if err != nil {
		t.Fatalf("reopen Open: %v", err)
	}
	defer s2.Close()
	if s2.Count() != 0 {
		t.Fatalf("mismatched-fingerprint Count = %d, want 0 (stale discarded)", s2.Count())
	}
	if _, ok := s2.Load("0-5"); ok {
		t.Error("stale chunk from an incompatible run was loaded")
	}

	// The fresh run's own saves take effect on top of the discarded file.
	if err := s2.Save("0-5", sampleResult("fresh")); err != nil {
		t.Fatalf("Save after discard: %v", err)
	}
	s2.Close()
	s3, err := Open(path, "fingerprint-B")
	if err != nil {
		t.Fatalf("third Open: %v", err)
	}
	defer s3.Close()
	got, ok := s3.Load("0-5")
	if !ok || got.Text != "fresh" {
		t.Errorf("post-discard reload = %v/%q, want fresh", ok, textOf(got))
	}
}

func TestStore_TornTrailingLineTolerated(t *testing.T) {
	path := sidecar(t)
	const fp = "fingerprint-A"

	s, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Save("0-5", sampleResult("good-0")); err != nil {
		t.Fatalf("Save 0: %v", err)
	}
	if err := s.Save("5-10", sampleResult("good-1")); err != nil {
		t.Fatalf("Save 1: %v", err)
	}
	s.Close()

	// Simulate a crash mid-append: a partial JSON line with no newline.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("reopen for corruption: %v", err)
	}
	if _, err := f.WriteString(`{"key":"10-15","result":{"Text":"tor`); err != nil {
		t.Fatalf("write torn line: %v", err)
	}
	f.Close()

	// The two complete chunks must still load; the torn line is skipped.
	s2, err := Open(path, fp)
	if err != nil {
		t.Fatalf("reopen after corruption: %v", err)
	}
	defer s2.Close()
	if s2.Count() != 2 {
		t.Fatalf("Count after torn trailing line = %d, want 2", s2.Count())
	}
	if _, ok := s2.Load("0-5"); !ok {
		t.Error("good-0 lost after torn trailing line")
	}
	if _, ok := s2.Load("5-10"); !ok {
		t.Error("good-1 lost after torn trailing line")
	}
}

func TestStore_DiscardRemovesFile(t *testing.T) {
	path := sidecar(t)
	s, err := Open(path, "fp")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Save("0-5", sampleResult("x")); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := s.Discard(); err != nil {
		t.Fatalf("Discard: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("sidecar still present after Discard: stat err = %v", err)
	}
	// Close after Discard is a no-op, not a panic.
	if err := s.Close(); err != nil {
		t.Errorf("Close after Discard: %v", err)
	}
}

func TestStore_ConcurrentSaves(t *testing.T) {
	path := sidecar(t)
	const fp = "fp"
	s, err := Open(path, fp)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	const n = 32
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := keyFor(i)
			if err := s.Save(key, sampleResult(key)); err != nil {
				t.Errorf("Save(%s): %v", key, err)
			}
		}(i)
	}
	wg.Wait()
	s.Close()

	s2, err := Open(path, fp)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()
	if s2.Count() != n {
		t.Fatalf("Count after %d concurrent saves = %d", n, s2.Count())
	}
}

func textOf(r *whisper.Result) string {
	if r == nil {
		return ""
	}
	return r.Text
}

func keyFor(i int) string {
	return time.Duration(i).String() + "-k"
}
