// Package output renders aligned speaker lines to one of three text
// formats: timestamped (default), WhisperX byte-for-byte compatible,
// and JSON for programmatic consumers.
package output

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/matthewjhunter/transcribe/internal/align"
)

// Format selects the rendering style.
type Format int

const (
	// FormatTimestampedTXT writes "[HH:MM:SS] [SPEAKER_NN]: text\n".
	// Default and recommended for new pipelines.
	FormatTimestampedTXT Format = iota

	// FormatWhisperXTXT writes "[SPEAKER_NN]: text\n", matching the
	// historical WhisperX --output_format txt + --diarize output
	// byte-for-byte. For drop-in compatibility with existing tooling.
	FormatWhisperXTXT

	// FormatJSON writes the lines slice as a JSON array suitable for
	// programmatic consumers.
	FormatJSON
)

// Render writes lines to w in the requested format.
func Render(lines []align.SpeakerLine, w io.Writer, f Format) error {
	switch f {
	case FormatTimestampedTXT:
		return renderTimestamped(lines, w)
	case FormatWhisperXTXT:
		return renderWhisperX(lines, w)
	case FormatJSON:
		return renderJSON(lines, w)
	default:
		return fmt.Errorf("output: unknown format %d", f)
	}
}

func renderTimestamped(lines []align.SpeakerLine, w io.Writer) error {
	for _, l := range lines {
		h, m, s := splitHMS(l.Start)
		if _, err := fmt.Fprintf(w, "[%02d:%02d:%02d] [SPEAKER_%02d]: %s\n",
			h, m, s, l.Speaker, l.Text); err != nil {
			return err
		}
	}
	return nil
}

func renderWhisperX(lines []align.SpeakerLine, w io.Writer) error {
	for _, l := range lines {
		if _, err := fmt.Fprintf(w, "[SPEAKER_%02d]: %s\n", l.Speaker, l.Text); err != nil {
			return err
		}
	}
	return nil
}

type jsonLine struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker int     `json:"speaker"`
	Text    string  `json:"text"`
}

func renderJSON(lines []align.SpeakerLine, w io.Writer) error {
	out := make([]jsonLine, len(lines))
	for i, l := range lines {
		out[i] = jsonLine{
			Start:   l.Start.Seconds(),
			End:     l.End.Seconds(),
			Speaker: l.Speaker,
			Text:    l.Text,
		}
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func splitHMS(d time.Duration) (h, m, s int) {
	total := int(d / time.Second)
	h = total / 3600
	m = (total % 3600) / 60
	s = total % 60
	return
}
