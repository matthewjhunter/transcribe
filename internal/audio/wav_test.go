package audio

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestEncodePCM16WAV_RoundTrip(t *testing.T) {
	in := []float32{0, 0.25, -0.25, 1.0, -1.0, 0.5}
	const sr = 16000

	var buf bytes.Buffer
	if err := EncodePCM16WAV(in, sr, &buf); err != nil {
		t.Fatalf("EncodePCM16WAV: %v", err)
	}

	// 44-byte header + 12 bytes of PCM (6 samples * 2)
	if got, want := buf.Len(), 44+12; got != want {
		t.Errorf("wav size: got %d want %d", got, want)
	}

	// Round-trip through ReadFloat32 to verify the file is parseable
	// and samples are recovered to within 16-bit quantization tolerance.
	tmp := filepath.Join(t.TempDir(), "round.wav")
	if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write tmp: %v", err)
	}
	got, gotSR, err := ReadFloat32(tmp)
	if err != nil {
		t.Fatalf("ReadFloat32: %v", err)
	}
	if gotSR != sr {
		t.Errorf("sample rate: got %d want %d", gotSR, sr)
	}
	if len(got) != len(in) {
		t.Fatalf("sample count: got %d want %d", len(got), len(in))
	}
	const tol = 1.0 / 32767.0 * 2
	for i, v := range got {
		if d := v - in[i]; d > tol || d < -tol {
			t.Errorf("sample %d: got %v want %v (tol %v)", i, v, in[i], tol)
		}
	}
}

func TestEncodePCM16WAV_Clips(t *testing.T) {
	in := []float32{2.0, -2.0, 0}
	var buf bytes.Buffer
	if err := EncodePCM16WAV(in, 16000, &buf); err != nil {
		t.Fatalf("EncodePCM16WAV: %v", err)
	}
	// PCM data starts at byte 44.
	pcm := buf.Bytes()[44:]
	gotPos := int16(uint16(pcm[0]) | uint16(pcm[1])<<8)
	gotNeg := int16(uint16(pcm[2]) | uint16(pcm[3])<<8)
	if gotPos != 32767 {
		t.Errorf("clipped +2.0: got %d want 32767", gotPos)
	}
	if gotNeg != -32768 {
		t.Errorf("clipped -2.0: got %d want -32768", gotNeg)
	}
}

func TestEncodePCM16WAV_BadSampleRate(t *testing.T) {
	var buf bytes.Buffer
	if err := EncodePCM16WAV([]float32{0}, 0, &buf); err == nil {
		t.Error("expected error for sample rate 0, got nil")
	}
}

func TestEncodePCM16WAV_Empty(t *testing.T) {
	var buf bytes.Buffer
	if err := EncodePCM16WAV(nil, 16000, &buf); err != nil {
		t.Fatalf("EncodePCM16WAV(nil): %v", err)
	}
	if buf.Len() != 44 {
		t.Errorf("empty WAV size: got %d want 44 (header only)", buf.Len())
	}
}
