package audio

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	goaudio "github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

func TestReadFloat32_Mono16kS16(t *testing.T) {
	const sampleRate = 16000
	// 100 ms of a 1 kHz square-ish tone alternating between two values.
	samples := make([]int, sampleRate/10)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = 16384
		} else {
			samples[i] = -16384
		}
	}

	wavPath := filepath.Join(t.TempDir(), "tone.wav")
	writeMonoWAV(t, wavPath, samples, sampleRate)

	got, sr, err := ReadFloat32(wavPath)
	if err != nil {
		t.Fatalf("ReadFloat32: %v", err)
	}
	if sr != sampleRate {
		t.Errorf("sample rate: got %d want %d", sr, sampleRate)
	}
	if len(got) != len(samples) {
		t.Fatalf("sample count: got %d want %d", len(got), len(samples))
	}

	// 16384 / 32768 = 0.5; tolerate small float drift.
	for i, want := range []float32{0.5, -0.5, 0.5, -0.5} {
		if math.Abs(float64(got[i]-want)) > 1e-3 {
			t.Errorf("sample[%d]: got %v want %v", i, got[i], want)
		}
	}
}

func TestReadFloat32_RejectsStereo(t *testing.T) {
	wavPath := filepath.Join(t.TempDir(), "stereo.wav")
	writeStereoWAV(t, wavPath, []int{0, 0, 100, -100}, 16000)

	_, _, err := ReadFloat32(wavPath)
	if err == nil {
		t.Fatal("expected error on stereo input, got nil")
	}
}

func TestProbeAndExtract(t *testing.T) {
	requireFFmpeg(t)

	// Make a stereo 22050 Hz source so ExtractWAV is forced to actually
	// resample + downmix to 16 kHz mono.
	src := filepath.Join(t.TempDir(), "src.wav")
	cmd := exec.Command("ffmpeg",
		"-y", "-loglevel", "error",
		"-f", "lavfi", "-i", "sine=f=440:duration=1:sample_rate=22050",
		"-ac", "2",
		src,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg generate: %v: %s", err, out)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	p, err := ProbeFile(ctx, src)
	if err != nil {
		t.Fatalf("ProbeFile: %v", err)
	}
	if !p.HasAudio {
		t.Errorf("HasAudio: got false, want true")
	}
	if p.HasVideo {
		t.Errorf("HasVideo: got true, want false")
	}
	if p.AudioCodec != "pcm_s16le" {
		t.Errorf("AudioCodec: got %q, want pcm_s16le", p.AudioCodec)
	}
	if p.Duration < 900*time.Millisecond || p.Duration > 1100*time.Millisecond {
		t.Errorf("Duration: got %v, want ~1s", p.Duration)
	}

	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ExtractWAV(ctx, src, out); err != nil {
		t.Fatalf("ExtractWAV: %v", err)
	}
	got, sr, err := ReadFloat32(out)
	if err != nil {
		t.Fatalf("ReadFloat32(extracted): %v", err)
	}
	if sr != 16000 {
		t.Errorf("extracted sample rate: got %d want 16000", sr)
	}
	if len(got) < 15000 || len(got) > 17000 {
		t.Errorf("extracted sample count: got %d want ~16000", len(got))
	}
}

func TestExtractAudioStream_Lossless(t *testing.T) {
	requireFFmpeg(t)

	// Generate a video with an AAC audio stream so the stream copy
	// exercises a real codec→m4a path (not WAV-into-WAV).
	src := filepath.Join(t.TempDir(), "src.mp4")
	cmd := exec.Command("ffmpeg",
		"-y", "-loglevel", "error",
		"-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
		"-f", "lavfi", "-i", "sine=f=440:duration=1",
		"-c:v", "libx264", "-c:a", "aac", "-shortest",
		src,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("ffmpeg generate (encoders unavailable?): %v: %s", err, out)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	p, err := ProbeFile(ctx, src)
	if err != nil {
		t.Fatalf("ProbeFile: %v", err)
	}
	if !p.HasVideo || !p.HasAudio {
		t.Fatalf("source missing streams: %+v", p)
	}
	if p.AudioCodec != "aac" {
		t.Errorf("AudioCodec: got %q, want aac", p.AudioCodec)
	}
	if got := AudioStreamExt(p.AudioCodec); got != "m4a" {
		t.Errorf("AudioStreamExt(aac) = %q, want m4a", got)
	}

	out := filepath.Join(t.TempDir(), "out.m4a")
	if err := ExtractAudioStream(ctx, src, out); err != nil {
		t.Fatalf("ExtractAudioStream: %v", err)
	}

	// Confirm the extracted file has audio, no video, and the same codec.
	op, err := ProbeFile(ctx, out)
	if err != nil {
		t.Fatalf("ProbeFile(out): %v", err)
	}
	if op.HasVideo {
		t.Errorf("output has video stream; expected audio-only")
	}
	if !op.HasAudio || op.AudioCodec != "aac" {
		t.Errorf("output codec: got %q, want aac", op.AudioCodec)
	}
}

func TestAudioStreamExt(t *testing.T) {
	cases := []struct {
		codec, want string
	}{
		{"aac", "m4a"},
		{"alac", "m4a"},
		{"mp3", "mp3"},
		{"opus", "opus"},
		{"vorbis", "ogg"},
		{"flac", "flac"},
		{"ac3", "ac3"},
		{"eac3", "eac3"},
		{"wmav2", "wma"},
		{"pcm_s16le", "wav"},
		{"pcm_f32le", "wav"},
		{"unknown_codec", "unknown_codec"}, // fallback
	}
	for _, tc := range cases {
		if got := AudioStreamExt(tc.codec); got != tc.want {
			t.Errorf("AudioStreamExt(%q) = %q, want %q", tc.codec, got, tc.want)
		}
	}
}

func requireFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH; skipping")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not on PATH; skipping")
	}
}

func writeMonoWAV(t *testing.T, path string, samples []int, sampleRate int) {
	t.Helper()
	writeWAV(t, path, samples, sampleRate, 1)
}

func writeStereoWAV(t *testing.T, path string, samples []int, sampleRate int) {
	t.Helper()
	writeWAV(t, path, samples, sampleRate, 2)
}

func writeWAV(t *testing.T, path string, samples []int, sampleRate, channels int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %q: %v", path, err)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, channels, 1) // audioFormat=1 → PCM
	defer func() {
		if err := enc.Close(); err != nil {
			t.Fatalf("close encoder: %v", err)
		}
	}()

	buf := &goaudio.IntBuffer{
		Format:         &goaudio.Format{NumChannels: channels, SampleRate: sampleRate},
		Data:           samples,
		SourceBitDepth: 16,
	}
	if err := enc.Write(buf); err != nil {
		t.Fatalf("encode: %v", err)
	}
}
