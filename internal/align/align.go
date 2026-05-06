// Package align merges Whisper transcription output with diarization
// turns to produce per-speaker lines.
//
// The preferred path is AssignSpeakers, which works at word granularity:
// each Whisper word is assigned to the diarization turn it overlaps most,
// then consecutive same-speaker words are grouped into one SpeakerLine.
//
// AssignFromSegments is the fallback for backends that don't return
// word-level timestamps; it groups consecutive same-speaker segments at
// segment granularity. Less accurate during overlapping or rapid
// turn-taking, but acceptable as a fallback.
package align

import (
	"strings"
	"time"

	"github.com/matthewjhunter/transcribe/internal/diarize"
	"github.com/matthewjhunter/transcribe/internal/whisper"
)

// SpeakerLine is one contiguous run of text attributed to a single speaker.
//
// Label is an optional voice-characterization tag (e.g. "M", "F", "?")
// the renderer can attach to the speaker. Empty string means no label
// should be shown.
type SpeakerLine struct {
	Start   time.Duration
	End     time.Duration
	Speaker int
	Label   string
	Text    string
}

// AssignSpeakers walks words in order, attributes each to the
// diarization turn with maximum overlap, and groups consecutive
// same-speaker words into SpeakerLines.
//
// Returns nil if either input is empty.
func AssignSpeakers(words []whisper.Word, turns []diarize.Turn) []SpeakerLine {
	if len(words) == 0 || len(turns) == 0 {
		return nil
	}

	var lines []SpeakerLine
	var cur *SpeakerLine

	for _, w := range words {
		sp := pickSpeaker(w.Start, w.End, turns)
		if cur == nil || cur.Speaker != sp {
			if cur != nil {
				lines = append(lines, *cur)
			}
			cur = &SpeakerLine{Start: w.Start, Speaker: sp}
		}
		cur.End = w.End
		cur.Text += w.Text
	}
	if cur != nil {
		lines = append(lines, *cur)
	}

	for i := range lines {
		lines[i].Text = strings.TrimLeft(lines[i].Text, " ")
	}
	return lines
}

// AssignFromSegments groups consecutive same-speaker segments. Each
// segment is attributed to the diarization turn with maximum overlap.
//
// Returns nil if either input is empty.
func AssignFromSegments(segments []whisper.Segment, turns []diarize.Turn) []SpeakerLine {
	if len(segments) == 0 || len(turns) == 0 {
		return nil
	}

	var lines []SpeakerLine
	var cur *SpeakerLine

	for _, s := range segments {
		sp := pickSpeaker(s.Start, s.End, turns)
		text := strings.TrimLeft(s.Text, " ")
		if cur == nil || cur.Speaker != sp {
			if cur != nil {
				lines = append(lines, *cur)
			}
			cur = &SpeakerLine{Start: s.Start, Speaker: sp, Text: text}
		} else {
			cur.Text += " " + text
		}
		cur.End = s.End
	}
	if cur != nil {
		lines = append(lines, *cur)
	}
	return lines
}

// pickSpeaker returns the Speaker of the turn with maximum overlap with
// [start, end]. If no turn overlaps, falls back to the turn whose center
// is nearest the interval's center.
func pickSpeaker(start, end time.Duration, turns []diarize.Turn) int {
	bestIdx := -1
	var bestOverlap time.Duration
	for i, t := range turns {
		a := maxDur(start, t.Start)
		b := minDur(end, t.End)
		if b <= a {
			continue
		}
		if d := b - a; d > bestOverlap {
			bestOverlap = d
			bestIdx = i
		}
	}
	if bestIdx >= 0 {
		return turns[bestIdx].Speaker
	}

	// no overlap; pick nearest by center distance
	mid := (start + end) / 2
	nearest := 0
	nearestDist := absDur((turns[0].Start+turns[0].End)/2 - mid)
	for i := 1; i < len(turns); i++ {
		t := turns[i]
		d := absDur((t.Start+t.End)/2 - mid)
		if d < nearestDist {
			nearestDist = d
			nearest = i
		}
	}
	return turns[nearest].Speaker
}

func maxDur(a, b time.Duration) time.Duration {
	if a > b {
		return a
	}
	return b
}

func minDur(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func absDur(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}
