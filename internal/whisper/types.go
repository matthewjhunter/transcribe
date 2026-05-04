package whisper

import "time"

// Result is the structured outcome of one /v1/audio/transcriptions call.
//
// Words may be nil when the backend doesn't return word-level timestamps;
// callers should fall back to segment-level alignment in that case.
type Result struct {
	Text     string
	Language string
	Duration time.Duration
	Segments []Segment
	Words    []Word
}

// Segment is a chunk of the transcript with its time bounds.
type Segment struct {
	Start time.Duration
	End   time.Duration
	Text  string
}

// Word is a single decoded token with its time bounds. The Text field
// preserves the leading space the decoder emits (Whisper-style).
type Word struct {
	Start time.Duration
	End   time.Duration
	Text  string
}
