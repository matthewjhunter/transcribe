// Package whisper is a small HTTP client for the OpenAI-compatible
// /v1/audio/transcriptions endpoint exposed by Lemonade, whisper.cpp's
// HTTP server, and api.openai.com.
//
// The client streams the audio file via multipart/form-data and asks for
// verbose_json with both segment and word timestamp granularities, so the
// caller can do its own alignment (e.g. against speaker diarization).
package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DefaultEndpoint is the assumed Lemonade URL when the caller doesn't override.
	DefaultEndpoint = "http://localhost:13305/api/v1"

	// DefaultModel is the Lemonade model id we ask for by default.
	DefaultModel = "Whisper-Large-v3"

	transcriptionsPath = "/audio/transcriptions"

	// maxResponseBytes caps response decoding. Real responses are KB-MB
	// not GB; this is paranoia, not a real expectation.
	maxResponseBytes = 64 * 1024 * 1024

	// maxErrorBodyBytes caps the slice of error bodies surfaced in error messages.
	maxErrorBodyBytes = 4 * 1024
)

// Config controls a Client.
type Config struct {
	// Endpoint is the OpenAI-compatible base URL. May be the bare
	// "http://host:port/api/v1" or include the trailing
	// "/audio/transcriptions" — both are accepted.
	Endpoint string

	// APIKey is sent as Bearer auth; empty omits the header.
	APIKey string

	// Model is the model name passed in the multipart form.
	Model string

	// Language is an optional ISO-639-1 hint. Empty means auto-detect.
	Language string

	// Timeout is forwarded to the default http.Client when HTTPClient is nil.
	// Zero means no timeout — appropriate for long audio files.
	Timeout time.Duration

	// HTTPClient lets callers inject a custom client (tests, proxies, TLS).
	// When non-nil, Timeout is ignored.
	HTTPClient *http.Client
}

// Client posts audio files to an OpenAI-compatible transcription endpoint.
type Client struct {
	endpoint string
	apiKey   string
	model    string
	language string
	hc       *http.Client
}

// New constructs a Client. Empty fields in cfg fall back to the package defaults.
func New(cfg Config) *Client {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	if !strings.HasSuffix(endpoint, transcriptionsPath) {
		endpoint = strings.TrimRight(endpoint, "/") + transcriptionsPath
	}

	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}

	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: cfg.Timeout}
	}

	return &Client{
		endpoint: endpoint,
		apiKey:   cfg.APIKey,
		model:    model,
		language: cfg.Language,
		hc:       hc,
	}
}

// Transcribe uploads the file at wavPath (or any container ffmpeg / the
// backend can decode) and returns the parsed verbose_json result.
func (c *Client) Transcribe(ctx context.Context, wavPath string) (*Result, error) {
	f, err := os.Open(wavPath)
	if err != nil {
		return nil, fmt.Errorf("whisper: open %q: %w", wavPath, err)
	}
	defer f.Close()

	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)

	go c.writeMultipart(pw, mw, f, filepath.Base(wavPath))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, pr)
	if err != nil {
		_ = pr.Close()
		return nil, fmt.Errorf("whisper: build request: %w", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("whisper: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return nil, fmt.Errorf("whisper: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("whisper: read response: %w", err)
	}
	return parseResponse(body)
}

// writeMultipart runs in a goroutine, writing the upload body into the
// pipe. It always closes the writer side so the request body terminates.
func (c *Client) writeMultipart(pw *io.PipeWriter, mw *multipart.Writer, f *os.File, name string) {
	var werr error
	defer func() {
		if cerr := mw.Close(); werr == nil && cerr != nil {
			werr = cerr
		}
		_ = pw.CloseWithError(werr)
	}()

	fields := []struct{ k, v string }{
		{"model", c.model},
		{"response_format", "verbose_json"},
		{"timestamp_granularities[]", "segment"},
		{"timestamp_granularities[]", "word"},
	}
	for _, kv := range fields {
		if werr = mw.WriteField(kv.k, kv.v); werr != nil {
			return
		}
	}
	if c.language != "" {
		if werr = mw.WriteField("language", c.language); werr != nil {
			return
		}
	}

	fw, err := mw.CreateFormFile("file", name)
	if err != nil {
		werr = err
		return
	}
	if _, err := io.Copy(fw, f); err != nil {
		werr = err
		return
	}
}

// apiResponse covers both shapes we've seen in the wild:
//   - OpenAI: top-level Words []apiWord, Segments without nested words.
//   - Lemonade: Words inside each Segment, no top-level Words.
//
// parseResponse normalizes both into Result.Words.
type apiResponse struct {
	Text     string       `json:"text"`
	Language string       `json:"language"`
	Duration float64      `json:"duration"`
	Segments []apiSegment `json:"segments"`
	Words    []apiWord    `json:"words"`
}

type apiSegment struct {
	Start float64   `json:"start"`
	End   float64   `json:"end"`
	Text  string    `json:"text"`
	Words []apiWord `json:"words"`
}

type apiWord struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Word  string  `json:"word"`
}

func parseResponse(body []byte) (*Result, error) {
	var ar apiResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, fmt.Errorf("whisper: decode response: %w", err)
	}

	res := &Result{
		Text:     ar.Text,
		Language: ar.Language,
		Duration: seconds(ar.Duration),
	}

	if len(ar.Segments) > 0 {
		res.Segments = make([]Segment, len(ar.Segments))
		for i, s := range ar.Segments {
			res.Segments[i] = Segment{
				Start: seconds(s.Start),
				End:   seconds(s.End),
				Text:  s.Text,
			}
		}
	}

	switch {
	case len(ar.Words) > 0:
		res.Words = make([]Word, len(ar.Words))
		for i, w := range ar.Words {
			res.Words[i] = Word{
				Start: seconds(w.Start),
				End:   seconds(w.End),
				Text:  w.Word,
			}
		}
	default:
		// flatten Lemonade-style nested words
		var n int
		for _, s := range ar.Segments {
			n += len(s.Words)
		}
		if n > 0 {
			res.Words = make([]Word, 0, n)
			for _, s := range ar.Segments {
				for _, w := range s.Words {
					res.Words = append(res.Words, Word{
						Start: seconds(w.Start),
						End:   seconds(w.End),
						Text:  w.Word,
					})
				}
			}
		}
	}

	return res, nil
}

func seconds(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}
