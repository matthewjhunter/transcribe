package output

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/align"
)

var fixture = []align.SpeakerLine{
	{Start: 0 * time.Second, End: 2 * time.Second, Speaker: 0, Text: "Hello there."},
	{Start: 2 * time.Second, End: 4 * time.Second, Speaker: 1, Text: "General Kenobi."},
	{Start: time.Hour + 2*time.Minute + 3*time.Second, End: time.Hour + 2*time.Minute + 5*time.Second, Speaker: 0, Text: "You're a bold one."},
}

func TestRender_TimestampedGolden(t *testing.T) {
	want, err := os.ReadFile("testdata/timestamped.golden.txt")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var buf bytes.Buffer
	if err := Render(fixture, &buf, FormatTimestampedTXT); err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Errorf("output mismatch:\ngot:  %q\nwant: %q", buf.String(), string(want))
	}
}

func TestRender_WhisperXGolden(t *testing.T) {
	want, err := os.ReadFile("testdata/whisperx.golden.txt")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var buf bytes.Buffer
	if err := Render(fixture, &buf, FormatWhisperXTXT); err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Errorf("output mismatch:\ngot:  %q\nwant: %q", buf.String(), string(want))
	}
}

func TestRender_JSON(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(fixture, &buf, FormatJSON); err != nil {
		t.Fatalf("Render: %v", err)
	}
	var got []jsonLine
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if len(got) != len(fixture) {
		t.Fatalf("length: got %d want %d", len(got), len(fixture))
	}
	if got[0].Speaker != 0 || got[0].Text != "Hello there." {
		t.Errorf("first line: got %+v", got[0])
	}
	if got[2].Start < 3722.0 || got[2].Start > 3724.0 {
		t.Errorf("third line start: got %v want ~3723.0", got[2].Start)
	}
}

func TestRender_UnknownFormatRejected(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(fixture, &buf, Format(999)); err == nil {
		t.Error("expected error for unknown format")
	}
}

var labeledFixture = []align.SpeakerLine{
	{Start: 0 * time.Second, End: 2 * time.Second, Speaker: 0, Label: "M", Text: "Hello there."},
	{Start: 2 * time.Second, End: 4 * time.Second, Speaker: 1, Label: "F", Text: "General Kenobi."},
	{Start: 4 * time.Second, End: 6 * time.Second, Speaker: 2, Label: "?", Text: "Mystery voice."},
}

func TestRender_TimestampedWithLabels(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(labeledFixture, &buf, FormatTimestampedTXT); err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := "[00:00:00] [SPEAKER_00 (M)]: Hello there.\n" +
		"[00:00:02] [SPEAKER_01 (F)]: General Kenobi.\n" +
		"[00:00:04] [SPEAKER_02 (?)]: Mystery voice.\n"
	if got := buf.String(); got != want {
		t.Errorf("labeled tstxt mismatch:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestRender_WhisperXIgnoresLabels(t *testing.T) {
	// wxtxt is mandated to be byte-for-byte WhisperX-compatible; labels
	// must never leak into it even when present on the SpeakerLine.
	var buf bytes.Buffer
	if err := Render(labeledFixture, &buf, FormatWhisperXTXT); err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := "[SPEAKER_00]: Hello there.\n" +
		"[SPEAKER_01]: General Kenobi.\n" +
		"[SPEAKER_02]: Mystery voice.\n"
	if got := buf.String(); got != want {
		t.Errorf("wxtxt label leakage:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestRender_JSONIncludesLabel(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(labeledFixture, &buf, FormatJSON); err != nil {
		t.Fatalf("Render: %v", err)
	}
	var got []jsonLine
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if got[0].Label != "M" || got[1].Label != "F" || got[2].Label != "?" {
		t.Errorf("labels: got %q/%q/%q, want M/F/?", got[0].Label, got[1].Label, got[2].Label)
	}
}

func TestRender_JSONOmitsEmptyLabel(t *testing.T) {
	// label,omitempty: lines with no label should not include the field.
	var buf bytes.Buffer
	if err := Render(fixture, &buf, FormatJSON); err != nil {
		t.Fatalf("Render: %v", err)
	}
	if bytes.Contains(buf.Bytes(), []byte(`"label"`)) {
		t.Errorf("expected no \"label\" key for unlabeled fixture, got: %s", buf.String())
	}
}

func TestRender_EmptyInput(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(nil, &buf, FormatTimestampedTXT); err != nil {
		t.Fatalf("Render(nil): %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("expected empty output, got %q", buf.String())
	}
}
