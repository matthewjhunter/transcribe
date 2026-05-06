package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/vad"
)

// fakeServer captures every multipart upload so tests can assert per-chunk
// shape (file name, body size) and shape the canned response per chunk.
type fakeServer struct {
	t *testing.T

	mu     sync.Mutex
	calls  []fakeCall
	inFlt  atomic.Int32
	maxFlt atomic.Int32

	respond func(call fakeCall) ([]byte, int)
}

type fakeCall struct {
	fileName string
	bodyLen  int
}

// start spins up an httptest.Server backed by fs.
func (fs *fakeServer) start() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(fs.handle))
}

func (fs *fakeServer) handle(w http.ResponseWriter, r *http.Request) {
	cur := fs.inFlt.Add(1)
	for {
		mx := fs.maxFlt.Load()
		if cur <= mx || fs.maxFlt.CompareAndSwap(mx, cur) {
			break
		}
	}
	defer fs.inFlt.Add(-1)

	_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil {
		fs.t.Errorf("parse media type: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	mr := multipart.NewReader(r.Body, params["boundary"])
	var call fakeCall
	for {
		p, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			fs.t.Errorf("next part: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if p.FileName() != "" {
			call.fileName = p.FileName()
			b, _ := io.ReadAll(p)
			call.bodyLen = len(b)
		} else {
			_, _ = io.Copy(io.Discard, p)
		}
		_ = p.Close()
	}

	fs.mu.Lock()
	fs.calls = append(fs.calls, call)
	fs.mu.Unlock()

	body, status := fs.respond(call)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)

	// Brief delay so concurrent goroutines actually overlap when
	// concurrency > 1; without this the requests serialize naturally.
	time.Sleep(20 * time.Millisecond)
}

// chunkRespond returns a response with one segment and one word, each
// at chunk-relative time [0..0.5s], so callers can verify global-time
// offsets are applied correctly.
func chunkRespond(idx int) ([]byte, int) {
	resp := apiResponse{
		Text:     fmt.Sprintf("text-%d", idx),
		Language: "english",
		Duration: 0.5,
		Segments: []apiSegment{{Start: 0, End: 0.5, Text: fmt.Sprintf("text-%d", idx)}},
		Words:    []apiWord{{Start: 0, End: 0.5, Word: fmt.Sprintf("word-%d", idx)}},
	}
	body, _ := json.Marshal(resp)
	return body, http.StatusOK
}

// indexFromName parses the trailing digits from "chunk-NNNNN.wav".
func indexFromName(name string) int {
	var n int
	_, _ = fmt.Sscanf(name, "chunk-%05d.wav", &n)
	return n
}

func TestTranscribeChunks_NoChunks(t *testing.T) {
	c := New(Config{Endpoint: "http://unused"})
	got, err := c.TranscribeChunks(context.Background(), []float32{0, 0, 0}, 16000, nil, 1)
	if err != nil {
		t.Fatalf("TranscribeChunks(nil): %v", err)
	}
	if len(got.Words) != 0 || len(got.Segments) != 0 {
		t.Errorf("expected empty result, got %+v", got)
	}
}

func TestTranscribeChunks_OffsetsAndOrder(t *testing.T) {
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

	c := New(Config{Endpoint: srv.URL})
	got, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 1)
	if err != nil {
		t.Fatalf("TranscribeChunks: %v", err)
	}

	if len(got.Words) != 3 {
		t.Fatalf("got %d words, want 3", len(got.Words))
	}
	wantStart := []time.Duration{0, 10 * time.Second, 30 * time.Second}
	for i, w := range got.Words {
		if w.Start != wantStart[i] {
			t.Errorf("word[%d].Start = %v, want %v", i, w.Start, wantStart[i])
		}
		if w.End != wantStart[i]+500*time.Millisecond {
			t.Errorf("word[%d].End = %v, want %v", i, w.End, wantStart[i]+500*time.Millisecond)
		}
		if want := fmt.Sprintf("word-%d", i); w.Text != want {
			t.Errorf("word[%d].Text = %q, want %q", i, w.Text, want)
		}
	}
	if got.Text != "text-0 text-1 text-2" {
		t.Errorf("merged Text = %q", got.Text)
	}
	if got.Duration != 35*time.Second {
		t.Errorf("Duration = %v, want 35s", got.Duration)
	}
	if got.Language != "english" {
		t.Errorf("Language = %q", got.Language)
	}
}

func TestTranscribeChunks_FileNameAndWAVBody(t *testing.T) {
	fs := &fakeServer{t: t, respond: func(c fakeCall) ([]byte, int) {
		return chunkRespond(indexFromName(c.fileName))
	}}
	srv := fs.start()
	defer srv.Close()

	const sr = 16000
	samples := make([]float32, sr*10)
	chunks := []vad.Chunk{
		{Start: 0, End: 2 * time.Second},
		{Start: 3 * time.Second, End: 5 * time.Second},
	}

	c := New(Config{Endpoint: srv.URL})
	if _, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 1); err != nil {
		t.Fatalf("TranscribeChunks: %v", err)
	}

	if len(fs.calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(fs.calls))
	}
	sort.Slice(fs.calls, func(i, j int) bool {
		return fs.calls[i].fileName < fs.calls[j].fileName
	})
	if fs.calls[0].fileName != "chunk-00000.wav" {
		t.Errorf("call 0 name: got %q", fs.calls[0].fileName)
	}
	if fs.calls[1].fileName != "chunk-00001.wav" {
		t.Errorf("call 1 name: got %q", fs.calls[1].fileName)
	}
	// 2s of 16kHz mono int16 = 64000 bytes PCM + 44-byte header.
	wantSize := 44 + 2*sr*2
	for i, c := range fs.calls {
		if c.bodyLen != wantSize {
			t.Errorf("call %d body size: got %d want %d", i, c.bodyLen, wantSize)
		}
	}
}

func TestTranscribeChunks_RespectsConcurrency(t *testing.T) {
	fs := &fakeServer{t: t, respond: func(c fakeCall) ([]byte, int) {
		return chunkRespond(indexFromName(c.fileName))
	}}
	srv := fs.start()
	defer srv.Close()

	const sr = 16000
	samples := make([]float32, sr*30)
	chunks := make([]vad.Chunk, 6)
	for i := range chunks {
		chunks[i] = vad.Chunk{
			Start: time.Duration(i) * time.Second,
			End:   time.Duration(i+1) * time.Second,
		}
	}

	c := New(Config{Endpoint: srv.URL})
	if _, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 3); err != nil {
		t.Fatalf("TranscribeChunks: %v", err)
	}

	if got := fs.maxFlt.Load(); got > 3 {
		t.Errorf("max in-flight = %d, want <= 3", got)
	}
	if got := fs.maxFlt.Load(); got < 2 {
		t.Errorf("max in-flight = %d, want >= 2 (concurrency seems unused)", got)
	}
}

func TestTranscribeChunks_FailFast(t *testing.T) {
	fs := &fakeServer{t: t, respond: func(c fakeCall) ([]byte, int) {
		if indexFromName(c.fileName) == 1 {
			return []byte(`{"error":"boom"}`), http.StatusInternalServerError
		}
		return chunkRespond(indexFromName(c.fileName))
	}}
	srv := fs.start()
	defer srv.Close()

	const sr = 16000
	samples := make([]float32, sr*10)
	chunks := []vad.Chunk{
		{Start: 0, End: time.Second},
		{Start: time.Second, End: 2 * time.Second},
		{Start: 2 * time.Second, End: 3 * time.Second},
	}

	c := New(Config{Endpoint: srv.URL})
	_, err := c.TranscribeChunks(context.Background(), samples, sr, chunks, 1)
	if err == nil {
		t.Fatal("expected error from chunk 1, got nil")
	}
}
