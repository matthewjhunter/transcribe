// Package audio shells out to ffmpeg/ffprobe for stream inspection and
// WAV extraction, and decodes 16-bit signed PCM WAV files into float32
// samples for downstream consumers (the speaker-diarization stage in
// particular).
//
// ExtractWAV always produces 16 kHz mono int16 LE WAV regardless of
// input format. That keeps the rest of the pipeline simple: one decode
// path, one buffer reused for both the Whisper upload and the sherpa
// diarization input.
package audio

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"time"

	"github.com/go-audio/wav"
)

// Probe is the subset of ffprobe metadata we care about.
type Probe struct {
	HasVideo bool
	HasAudio bool
	Duration time.Duration
}

// ProbeFile inspects path with ffprobe and returns the streams summary.
//
// ffprobe is required on PATH. The returned error wraps the ffprobe
// stderr output when the probe fails.
func ProbeFile(ctx context.Context, path string) (Probe, error) {
	cmd := exec.CommandContext(ctx,
		"ffprobe",
		"-v", "error",
		"-of", "json",
		"-show_streams",
		"-show_format",
		path,
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return Probe{}, fmt.Errorf("audio: ffprobe %q: %w: %s", path, err, stderr.String())
	}

	var raw struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &raw); err != nil {
		return Probe{}, fmt.Errorf("audio: parse ffprobe output: %w", err)
	}

	var p Probe
	for _, s := range raw.Streams {
		switch s.CodecType {
		case "video":
			p.HasVideo = true
		case "audio":
			p.HasAudio = true
		}
	}
	if raw.Format.Duration != "" {
		if secs, err := strconv.ParseFloat(raw.Format.Duration, 64); err == nil {
			p.Duration = time.Duration(secs * float64(time.Second))
		}
	}
	return p, nil
}

// ExtractWAV transcodes input into a 16 kHz mono signed-16-bit WAV at output.
// Existing files at output are overwritten.
//
// ffmpeg is required on PATH. We always re-encode; the cost is small
// even for 2-hour recordings, and it gives the rest of the pipeline a
// single canonical input format.
func ExtractWAV(ctx context.Context, input, output string) error {
	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-y",
		"-loglevel", "error",
		"-i", input,
		"-vn",
		"-ac", "1",
		"-ar", "16000",
		"-sample_fmt", "s16",
		"-f", "wav",
		output,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("audio: ffmpeg %q -> %q: %w: %s",
			input, output, err, stderr.String())
	}
	return nil
}

// ReadFloat32 decodes the WAV file at path and returns its samples
// normalized to [-1.0, 1.0] along with the sample rate.
//
// Multi-channel inputs are not supported — call ExtractWAV first to
// guarantee a mono WAV. Returns an error rather than silently downmixing
// to make any caller mistake explicit.
func ReadFloat32(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("audio: open %q: %w", path, err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("audio: decode %q: %w", path, err)
	}
	if buf == nil || buf.Format == nil {
		return nil, 0, errors.New("audio: empty WAV decode result")
	}
	if buf.Format.NumChannels != 1 {
		return nil, 0, fmt.Errorf("audio: WAV is %d-channel; want mono",
			buf.Format.NumChannels)
	}

	fb := buf.AsFloat32Buffer()
	return fb.Data, buf.Format.SampleRate, nil
}
