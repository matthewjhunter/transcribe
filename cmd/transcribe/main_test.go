package main

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSiblingAudioPath(t *testing.T) {
	cases := []struct {
		name  string
		cfg   config
		codec string
		want  string
	}{
		{
			name:  "default output, video input",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "aac",
			want:  "/home/m/clip.m4a",
		},
		{
			name:  "explicit output redirects sibling too",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "/tmp/foo.txt"},
			codec: "aac",
			want:  "/tmp/foo.m4a",
		},
		{
			name:  "explicit output without extension",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "/tmp/transcript"},
			codec: "opus",
			want:  "/tmp/transcript.opus",
		},
		{
			name:  "stdout output skips sibling",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "-"},
			codec: "aac",
			want:  "",
		},
		{
			name:  "empty codec skips sibling",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "",
			want:  "",
		},
		{
			name:  "unknown codec falls back to codec-name extension",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "weird",
			want:  "/home/m/clip.weird",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := siblingAudioPath(tc.cfg, tc.codec)
			if got != tc.want {
				t.Errorf("siblingAudioPath() = %q, want %q", got, tc.want)
			}
		})
	}
}

// "relative" (and the empty string, for anyone passing --start-time=) is the
// opt-out now that "auto" is the default.
func TestResolveStartTime_RelativeOptOut(t *testing.T) {
	for _, v := range []string{"", "relative", "Relative", "none"} {
		got, err := resolveStartTime(discardLogger(), config{startTime: v}, time.Hour)
		if err != nil {
			t.Fatalf("resolveStartTime(%q): %v", v, err)
		}
		if !got.IsZero() {
			t.Errorf("resolveStartTime(%q) = %v, want zero time", v, got)
		}
	}
}

// The default is "auto", so parsing no flags at all must yield absolute
// timestamps -- that is the whole point of the change.
func TestParseFlags_StartTimeDefaultsToAuto(t *testing.T) {
	cfg, err := parseFlags([]string{"--no-diarize", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.startTime != "auto" {
		t.Errorf("cfg.startTime = %q, want %q", cfg.startTime, "auto")
	}
	if cfg.startTimeSet {
		t.Error("startTimeSet should be false when the flag was not passed")
	}
}

func TestParseFlags_StartTimeSetIsTracked(t *testing.T) {
	cfg, err := parseFlags([]string{"--no-diarize", "--start-time", "auto", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if !cfg.startTimeSet {
		t.Error("startTimeSet should be true when the flag was passed explicitly")
	}
}

func TestResolveStartTime_Explicit(t *testing.T) {
	got, err := resolveStartTime(discardLogger(), config{startTime: "2026-07-28T20:35:45"}, time.Hour)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	want := time.Date(2026, 7, 28, 20, 35, 45, 0, time.Local)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestResolveStartTime_Invalid(t *testing.T) {
	if _, err := resolveStartTime(discardLogger(), config{startTime: "half past eight"}, time.Hour); err == nil {
		t.Error("want an error for an unparseable start time")
	}
}

// "auto" derives the start as mtime minus duration: the file's last write is
// when the recording stopped, so backing off its length lands on the start.
func TestResolveStartTime_Auto(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "session.mkv")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	stopped := time.Date(2026, 7, 28, 23, 50, 41, 0, time.Local)
	if err := os.Chtimes(f, stopped, stopped); err != nil {
		t.Fatal(err)
	}

	dur := 3*time.Hour + 14*time.Minute + 56*time.Second
	got, err := resolveStartTime(discardLogger(), config{input: f, startTime: "auto", startTimeSet: true}, dur)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	want := stopped.Add(-dur)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// Asked for explicitly, an underivable "auto" is an error: the caller said what
// they wanted and silently not doing it would be worse than stopping.
func TestResolveStartTime_ExplicitAutoNeedsDuration(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "session.mkv")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveStartTime(discardLogger(), config{input: f, startTime: "auto", startTimeSet: true}, 0); err == nil {
		t.Error("want an error when the duration is unknown and auto was explicit")
	}
}

// Arrived at by default, an underivable "auto" degrades to relative timestamps
// rather than failing a transcription run that would have worked before.
func TestResolveStartTime_DefaultAutoDegradesToRelative(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "session.mkv")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := resolveStartTime(discardLogger(), config{input: f, startTime: "auto"}, 0)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	if !got.IsZero() {
		t.Errorf("got %v, want zero time (fall back to relative)", got)
	}
}

func TestResolveStartTime_DefaultAutoDegradesOnMissingInput(t *testing.T) {
	f := filepath.Join(t.TempDir(), "not-there.mkv")
	got, err := resolveStartTime(discardLogger(), config{input: f, startTime: "auto"}, time.Hour)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	if !got.IsZero() {
		t.Errorf("got %v, want zero time (fall back to relative)", got)
	}
}

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
