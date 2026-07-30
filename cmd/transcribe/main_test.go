package main

import (
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

func TestResolveStartTime_EmptyMeansRelative(t *testing.T) {
	got, err := resolveStartTime(config{}, time.Hour)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	if !got.IsZero() {
		t.Errorf("want zero time for an unset flag, got %v", got)
	}
}

func TestResolveStartTime_Explicit(t *testing.T) {
	got, err := resolveStartTime(config{startTime: "2026-07-28T20:35:45"}, time.Hour)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	want := time.Date(2026, 7, 28, 20, 35, 45, 0, time.Local)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestResolveStartTime_Invalid(t *testing.T) {
	if _, err := resolveStartTime(config{startTime: "half past eight"}, time.Hour); err == nil {
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
	got, err := resolveStartTime(config{input: f, startTime: "auto"}, dur)
	if err != nil {
		t.Fatalf("resolveStartTime: %v", err)
	}
	want := stopped.Add(-dur)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestResolveStartTime_AutoNeedsDuration(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "session.mkv")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveStartTime(config{input: f, startTime: "auto"}, 0); err == nil {
		t.Error("want an error when the duration is unknown")
	}
}
