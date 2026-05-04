package whisper

import (
	"context"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestParseResponse_LemonadeShape(t *testing.T) {
	body, err := os.ReadFile("testdata/lemonade-tone.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	got, err := parseResponse(body)
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
	}
	if got.Language != "english" {
		t.Errorf("Language: got %q want %q", got.Language, "english")
	}
	if got.Duration != 3*time.Second {
		t.Errorf("Duration: got %v want 3s", got.Duration)
	}
	if len(got.Segments) != 1 {
		t.Fatalf("Segments: got %d want 1", len(got.Segments))
	}
	if len(got.Words) != 1 {
		t.Fatalf("Words (flattened from segments[].words[]): got %d want 1", len(got.Words))
	}
	w := got.Words[0]
	if w.Text != " ." {
		t.Errorf("Word.Text: got %q want %q", w.Text, " .")
	}
	if w.Start != 0 || w.End != 2990*time.Millisecond {
		t.Errorf("Word time: got [%v..%v] want [0..2.99s]", w.Start, w.End)
	}
}

func TestParseResponse_OpenAIShape(t *testing.T) {
	body, err := os.ReadFile("testdata/openai-shape.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	got, err := parseResponse(body)
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
	}
	if got.Text != "Hello world. How are you?" {
		t.Errorf("Text: got %q", got.Text)
	}
	if len(got.Segments) != 2 {
		t.Errorf("Segments: got %d want 2", len(got.Segments))
	}
	if len(got.Words) != 7 {
		t.Errorf("Words (top-level): got %d want 7", len(got.Words))
	}
	if got.Words[0].Text != "Hello" {
		t.Errorf("first word: got %q want %q", got.Words[0].Text, "Hello")
	}
}

func TestParseResponse_TopLevelWordsWinsOverNested(t *testing.T) {
	// If both shapes are present (unusual), prefer the top-level array
	// because that's the documented OpenAI contract.
	body := []byte(`{
		"text": "x",
		"language": "english",
		"duration": 1.0,
		"words": [{"start": 0.0, "end": 0.5, "word": "top"}],
		"segments": [{"start": 0.0, "end": 0.5, "text": "x",
			"words": [{"start": 0.0, "end": 0.5, "word": "nested"}]}]
	}`)
	got, err := parseResponse(body)
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
	}
	if len(got.Words) != 1 || got.Words[0].Text != "top" {
		t.Errorf("expected top-level words to win; got %+v", got.Words)
	}
}

func TestEndpointNormalization(t *testing.T) {
	for _, tc := range []struct {
		in, want string
	}{
		{"http://localhost:13305/api/v1", "http://localhost:13305/api/v1/audio/transcriptions"},
		{"http://localhost:13305/api/v1/", "http://localhost:13305/api/v1/audio/transcriptions"},
		{"http://localhost:13305/api/v1/audio/transcriptions", "http://localhost:13305/api/v1/audio/transcriptions"},
		{"https://api.openai.com/v1", "https://api.openai.com/v1/audio/transcriptions"},
	} {
		c := New(Config{Endpoint: tc.in})
		if c.endpoint != tc.want {
			t.Errorf("endpoint %q: got %q want %q", tc.in, c.endpoint, tc.want)
		}
	}
}

func TestNew_Defaults(t *testing.T) {
	c := New(Config{})
	if !strings.HasPrefix(c.endpoint, DefaultEndpoint) {
		t.Errorf("default endpoint: got %q", c.endpoint)
	}
	if c.model != DefaultModel {
		t.Errorf("default model: got %q", c.model)
	}
}

// TestTranscribe_Success verifies the multipart body and the success path
// end-to-end against an httptest.Server returning a canned response.
func TestTranscribe_Success(t *testing.T) {
	canned, err := os.ReadFile("testdata/openai-shape.json")
	if err != nil {
		t.Fatalf("read canned: %v", err)
	}

	wavPath := filepath.Join(t.TempDir(), "in.wav")
	wavBytes := []byte("RIFF$\x00\x00\x00WAVEfmt fakeaudio")
	if err := os.WriteFile(wavPath, wavBytes, 0o644); err != nil {
		t.Fatalf("write wav: %v", err)
	}

	var captured struct {
		path     string
		auth     string
		fields   map[string][]string
		fileName string
		fileSize int
	}
	captured.fields = map[string][]string{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.Path
		captured.auth = r.Header.Get("Authorization")

		ct := r.Header.Get("Content-Type")
		_, params, err := mime.ParseMediaType(ct)
		if err != nil {
			t.Errorf("parse media type: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mr := multipart.NewReader(r.Body, params["boundary"])
		for {
			p, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Errorf("next part: %v", err)
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if p.FileName() != "" {
				captured.fileName = p.FileName()
				b, _ := io.ReadAll(p)
				captured.fileSize = len(b)
			} else {
				b, _ := io.ReadAll(p)
				captured.fields[p.FormName()] = append(captured.fields[p.FormName()], string(b))
			}
			_ = p.Close()
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(canned)
	}))
	defer srv.Close()

	c := New(Config{
		Endpoint: srv.URL,
		APIKey:   "test-key",
		Model:    "Whisper-Test",
		Language: "en",
	})

	got, err := c.Transcribe(context.Background(), wavPath)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got.Language != "english" || len(got.Words) != 7 {
		t.Errorf("unexpected result: %+v", got)
	}

	if captured.path != "/audio/transcriptions" {
		t.Errorf("path: got %q", captured.path)
	}
	if captured.auth != "Bearer test-key" {
		t.Errorf("auth: got %q", captured.auth)
	}
	if captured.fileName != "in.wav" {
		t.Errorf("file name: got %q", captured.fileName)
	}
	if captured.fileSize != len(wavBytes) {
		t.Errorf("file size: got %d want %d", captured.fileSize, len(wavBytes))
	}

	wantField := func(k, v string) {
		t.Helper()
		got := captured.fields[k]
		if len(got) == 0 || got[0] != v {
			t.Errorf("field %q: got %v want [%q]", k, got, v)
		}
	}
	wantField("model", "Whisper-Test")
	wantField("response_format", "verbose_json")
	wantField("language", "en")

	gran := captured.fields["timestamp_granularities[]"]
	sort.Strings(gran)
	if want := []string{"segment", "word"}; len(gran) != 2 || gran[0] != want[0] || gran[1] != want[1] {
		t.Errorf("timestamp_granularities[]: got %v want %v", gran, want)
	}
}

func TestTranscribe_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad model"}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	wavPath := filepath.Join(t.TempDir(), "in.wav")
	if err := os.WriteFile(wavPath, []byte("dummy"), 0o644); err != nil {
		t.Fatalf("write wav: %v", err)
	}

	c := New(Config{Endpoint: srv.URL})
	_, err := c.Transcribe(context.Background(), wavPath)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 400") || !strings.Contains(err.Error(), "bad model") {
		t.Errorf("error message: got %v", err)
	}
}

func TestTranscribe_NoAuthHeaderWhenKeyEmpty(t *testing.T) {
	canned, _ := os.ReadFile("testdata/openai-shape.json")
	gotAuth := "(unset)"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write(canned)
	}))
	defer srv.Close()

	wavPath := filepath.Join(t.TempDir(), "in.wav")
	_ = os.WriteFile(wavPath, []byte("dummy"), 0o644)

	c := New(Config{Endpoint: srv.URL}) // no APIKey
	if _, err := c.Transcribe(context.Background(), wavPath); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if gotAuth != "" {
		t.Errorf("Authorization header: got %q want empty", gotAuth)
	}
}
