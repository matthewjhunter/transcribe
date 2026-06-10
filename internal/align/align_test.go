package align

import (
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/diarize"
	"github.com/matthewjhunter/transcribe/internal/whisper"
)

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

func TestAssignSpeakers(t *testing.T) {
	tests := []struct {
		name  string
		words []whisper.Word
		turns []diarize.Turn
		want  []SpeakerLine
	}{
		{
			name:  "empty inputs return nil",
			words: nil,
			turns: nil,
			want:  nil,
		},
		{
			name:  "no turns returns nil",
			words: []whisper.Word{{Start: ms(0), End: ms(100), Text: "hi"}},
			turns: nil,
			want:  nil,
		},
		{
			name: "single speaker, multiple words, one line",
			words: []whisper.Word{
				{Start: ms(0), End: ms(500), Text: " Hello"},
				{Start: ms(500), End: ms(1000), Text: " world"},
				{Start: ms(1000), End: ms(1100), Text: "."},
			},
			turns: []diarize.Turn{{Start: ms(0), End: ms(2000), Speaker: 3}},
			want: []SpeakerLine{
				{Start: ms(0), End: ms(1100), Speaker: 3, Text: "Hello world."},
			},
		},
		{
			name: "two speakers strict alternation",
			words: []whisper.Word{
				{Start: ms(0), End: ms(500), Text: " Hi"},
				{Start: ms(1000), End: ms(1500), Text: " bye"},
			},
			turns: []diarize.Turn{
				{Start: ms(0), End: ms(800), Speaker: 0},
				{Start: ms(900), End: ms(1600), Speaker: 1},
			},
			want: []SpeakerLine{
				{Start: ms(0), End: ms(500), Speaker: 0, Text: "Hi"},
				{Start: ms(1000), End: ms(1500), Speaker: 1, Text: "bye"},
			},
		},
		{
			name: "word straddles boundary, attributed to dominant overlap",
			words: []whisper.Word{
				// word straddles 0..1000 with turn 0 owning 0..600 and turn 1 owning 600..1500
				{Start: ms(0), End: ms(1000), Text: " straddle"},
			},
			turns: []diarize.Turn{
				{Start: ms(0), End: ms(600), Speaker: 0},
				{Start: ms(600), End: ms(1500), Speaker: 1},
			},
			want: []SpeakerLine{
				{Start: ms(0), End: ms(1000), Speaker: 0, Text: "straddle"},
			},
		},
		{
			name: "word with no overlap goes to nearest turn",
			words: []whisper.Word{
				{Start: ms(2000), End: ms(2100), Text: " orphan"},
			},
			turns: []diarize.Turn{
				{Start: ms(0), End: ms(500), Speaker: 0},
				{Start: ms(1500), End: ms(1900), Speaker: 7},
			},
			want: []SpeakerLine{
				{Start: ms(2000), End: ms(2100), Speaker: 7, Text: "orphan"},
			},
		},
		{
			name: "consecutive same-speaker words across turn boundary stay one line",
			words: []whisper.Word{
				{Start: ms(0), End: ms(100), Text: " a"},
				{Start: ms(100), End: ms(200), Text: " b"},
				{Start: ms(200), End: ms(300), Text: " c"},
			},
			// two adjacent turns both labeled speaker 0 -- common when the
			// segmenter splits a long monologue.
			turns: []diarize.Turn{
				{Start: ms(0), End: ms(150), Speaker: 0},
				{Start: ms(150), End: ms(400), Speaker: 0},
			},
			want: []SpeakerLine{
				{Start: ms(0), End: ms(300), Speaker: 0, Text: "a b c"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AssignSpeakers(tc.words, tc.turns)
			equalLines(t, got, tc.want)
		})
	}
}

func TestAssignFromSegments(t *testing.T) {
	segments := []whisper.Segment{
		{Start: ms(0), End: ms(1000), Text: "Hello there."},
		{Start: ms(1000), End: ms(2000), Text: " More from same."},
		{Start: ms(2000), End: ms(3000), Text: "Different speaker now."},
	}
	turns := []diarize.Turn{
		{Start: ms(0), End: ms(1900), Speaker: 0},
		{Start: ms(1900), End: ms(3000), Speaker: 1},
	}
	got := AssignFromSegments(segments, turns)
	want := []SpeakerLine{
		{Start: ms(0), End: ms(2000), Speaker: 0, Text: "Hello there. More from same."},
		{Start: ms(2000), End: ms(3000), Speaker: 1, Text: "Different speaker now."},
	}
	equalLines(t, got, want)
}

func TestAssignFromSegments_Empty(t *testing.T) {
	if AssignFromSegments(nil, []diarize.Turn{{}}) != nil {
		t.Error("expected nil for empty segments")
	}
	if AssignFromSegments([]whisper.Segment{{}}, nil) != nil {
		t.Error("expected nil for empty turns")
	}
}

func equalLines(t *testing.T, got, want []SpeakerLine) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("line count: got %d want %d\ngot=%+v\nwant=%+v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("line[%d]:\n  got=%+v\n want=%+v", i, got[i], want[i])
		}
	}
}
