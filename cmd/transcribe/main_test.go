package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/whisper"
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

// --whisper-url and --whisper-model default from the environment, mirroring
// --whisper-api-key. Which host in the fleet serves ASR is a property of the
// machine, not of the invocation, so it belongs in the shell profile rather
// than in every command line.
func TestParseFlags_WhisperEndpointFromEnv(t *testing.T) {
	t.Setenv("WHISPER_URL", "http://192.0.2.10:13305/api/v1")
	t.Setenv("WHISPER_MODEL", "Whisper-Large-v3-Turbo")

	cfg, err := parseFlags([]string{"--no-diarize", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.whisperURL != "http://192.0.2.10:13305/api/v1" {
		t.Errorf("cfg.whisperURL = %q, want the WHISPER_URL value", cfg.whisperURL)
	}
	if cfg.whisperModel != "Whisper-Large-v3-Turbo" {
		t.Errorf("cfg.whisperModel = %q, want the WHISPER_MODEL value", cfg.whisperModel)
	}
}

// An explicit flag beats the environment, or the env var becomes impossible to
// override for a one-off run against a different backend.
func TestParseFlags_WhisperFlagsBeatEnv(t *testing.T) {
	t.Setenv("WHISPER_URL", "http://192.0.2.10:13305/api/v1")
	t.Setenv("WHISPER_MODEL", "Whisper-Large-v3-Turbo")

	cfg, err := parseFlags([]string{
		"--no-diarize",
		"--whisper-url", "http://192.0.2.99:13305/api/v1",
		"--whisper-model", "Whisper-Tiny",
		"session.mkv",
	})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.whisperURL != "http://192.0.2.99:13305/api/v1" {
		t.Errorf("cfg.whisperURL = %q, want the flag value", cfg.whisperURL)
	}
	if cfg.whisperModel != "Whisper-Tiny" {
		t.Errorf("cfg.whisperModel = %q, want the flag value", cfg.whisperModel)
	}
}

// With nothing set anywhere, the built-in defaults still apply.
func TestParseFlags_WhisperDefaultsWithoutEnv(t *testing.T) {
	t.Setenv("WHISPER_URL", "")
	t.Setenv("WHISPER_MODEL", "")

	cfg, err := parseFlags([]string{"--no-diarize", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.whisperURL != whisper.DefaultEndpoint {
		t.Errorf("cfg.whisperURL = %q, want %q", cfg.whisperURL, whisper.DefaultEndpoint)
	}
	if cfg.whisperModel != whisper.DefaultModel {
		t.Errorf("cfg.whisperModel = %q, want %q", cfg.whisperModel, whisper.DefaultModel)
	}
}
