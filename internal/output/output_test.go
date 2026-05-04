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

func TestRender_EmptyInput(t *testing.T) {
	var buf bytes.Buffer
	if err := Render(nil, &buf, FormatTimestampedTXT); err != nil {
		t.Fatalf("Render(nil): %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("expected empty output, got %q", buf.String())
	}
}
