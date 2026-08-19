package main

import (
	"io"
	"log/slog"
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

// --vad-threads is separate from --diarize-threads. They were once the same
// knob, which hid the VAD's thread penalty behind a flag named for the other
// stage: tuning diarization silently retuned VAD in the opposite direction.
func TestParseFlags_VADThreadsIsSeparateFromDiarizeThreads(t *testing.T) {
	cfg, err := parseFlags([]string{"--no-diarize", "--diarize-threads", "8", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.vadThreads != 0 {
		t.Errorf("cfg.vadThreads = %d, want 0 -- --diarize-threads must not set it", cfg.vadThreads)
	}

	cfg, err = parseFlags([]string{"--no-diarize", "--vad-threads", "4", "session.mkv"})
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}
	if cfg.vadThreads != 4 || cfg.diarizeThreads != 0 {
		t.Errorf("vadThreads=%d diarizeThreads=%d, want 4 and 0", cfg.vadThreads, cfg.diarizeThreads)
	}
}

// The sibling audio file must inherit the source's modification time. It is a
// re-container of the same recording, and --start-time auto derives the
// recording start from mtime minus duration -- so a sibling stamped with the
// time transcribe happened to run dates the recording to the wrong hour, and
// transcribing it later yields silently wrong wall-clock timestamps.
func TestCopyModTime(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "session.mkv")
	dst := filepath.Join(dir, "session.m4a")
	for _, p := range []string{src, dst} {
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want := time.Date(2026, 8, 18, 20, 50, 17, 0, time.Local)
	if err := os.Chtimes(src, want, want); err != nil {
		t.Fatal(err)
	}

	if err := copyModTime(src, dst); err != nil {
		t.Fatalf("copyModTime: %v", err)
	}
	fi, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !fi.ModTime().Equal(want) {
		t.Errorf("dst mtime = %v, want %v", fi.ModTime(), want)
	}
}

func TestCopyModTime_MissingSource(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "session.m4a")
	if err := os.WriteFile(dst, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := copyModTime(filepath.Join(dir, "not-there.mkv"), dst); err == nil {
		t.Error("expected an error when the source does not exist")
	}
}
