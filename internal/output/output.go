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

// AbsoluteLayout is the timestamp layout used when a recording start time is
// known. Chosen to sort lexicographically and to survive a session that runs
// past midnight, which a bare wall clock would not.
const AbsoluteLayout = "2006-01-02T15:04:05"

// Render writes lines to w in the requested format, with timestamps relative to
// the start of the recording.
func Render(lines []align.SpeakerLine, w io.Writer, f Format) error {
	return RenderFrom(lines, w, f, time.Time{})
}

// RenderFrom writes lines to w in the requested format. If start is non-zero,
// FormatTimestampedTXT emits absolute timestamps (start + each line's offset)
// instead of relative ones, so the transcript can be merged with other
// timestamped records of the same session by timestamp alone.
//
// start is ignored by the other formats: wxtxt has no timestamp slot, and json
// carries raw offsets for programmatic consumers.
func RenderFrom(lines []align.SpeakerLine, w io.Writer, f Format, start time.Time) error {
	switch f {
	case FormatTimestampedTXT:
		return renderTimestamped(lines, w, start)
	case FormatWhisperXTXT:
		return renderWhisperX(lines, w)
	case FormatJSON:
		return renderJSON(lines, w)
	default:
		return fmt.Errorf("output: unknown format %d", f)
	}
}

func renderTimestamped(lines []align.SpeakerLine, w io.Writer, start time.Time) error {
	for _, l := range lines {
		speaker := fmt.Sprintf("SPEAKER_%02d", l.Speaker)
		if l.Label != "" {
			speaker = fmt.Sprintf("SPEAKER_%02d (%s)", l.Speaker, l.Label)
		}

		var stamp string
		if start.IsZero() {
			h, m, s := splitHMS(l.Start)
			stamp = fmt.Sprintf("%02d:%02d:%02d", h, m, s)
		} else {
			stamp = start.Add(l.Start).Format(AbsoluteLayout)
		}

		if _, err := fmt.Fprintf(w, "[%s] [%s]: %s\n", stamp, speaker, l.Text); err != nil {
			return err
		}
	}
	return nil
}

// renderWhisperX intentionally ignores SpeakerLine.Label. The wxtxt
// format is mandated to be byte-for-byte compatible with WhisperX
// `--output_format txt --diarize`, which has no label slot.
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
	Label   string  `json:"label,omitempty"`
	Text    string  `json:"text"`
}

func renderJSON(lines []align.SpeakerLine, w io.Writer) error {
	out := make([]jsonLine, len(lines))
	for i, l := range lines {
		out[i] = jsonLine{
			Start:   l.Start.Seconds(),
			End:     l.End.Seconds(),
			Speaker: l.Speaker,
			Label:   l.Label,
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
