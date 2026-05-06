package vad

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNew_MissingModel(t *testing.T) {
	_, err := New(Config{})
	if err == nil || !strings.Contains(err.Error(), "Model is required") {
		t.Errorf("got %v, want Model-required error", err)
	}
}

func TestNew_FileMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := New(Config{Model: filepath.Join(dir, "nope.onnx")})
	if err == nil || !strings.Contains(err.Error(), "model:") {
		t.Errorf("got %v, want model-not-found error", err)
	}
}

func TestSamplesToDuration(t *testing.T) {
	cases := []struct {
		name       string
		n          int
		sampleRate int
		want       time.Duration
	}{
		{"zero", 0, 16000, 0},
		{"one second", 16000, 16000, time.Second},
		{"half second", 8000, 16000, 500 * time.Millisecond},
		{"32ms window", 512, 16000, 32 * time.Millisecond},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := samplesToDuration(tc.n, tc.sampleRate)
			if got != tc.want {
				t.Errorf("samplesToDuration(%d, %d) = %v, want %v",
					tc.n, tc.sampleRate, got, tc.want)
			}
		})
	}
}
