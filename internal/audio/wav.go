package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// EncodePCM16WAV writes mono float32 samples to w as a 16 kHz canonical
// 16-bit signed PCM WAV. Samples outside [-1.0, 1.0] are clipped before
// quantization with the standard f32 * 32767 mapping.
//
// This is the in-memory counterpart of ExtractWAV: where ExtractWAV
// shells out to ffmpeg to produce a file on disk, EncodePCM16WAV
// produces the exact same bytes from samples already in memory. Use it
// to feed a chunk of audio to a Whisper backend without round-tripping
// through the filesystem.
func EncodePCM16WAV(samples []float32, sampleRate int, w io.Writer) error {
	if sampleRate <= 0 {
		return fmt.Errorf("audio: sample rate must be positive, got %d", sampleRate)
	}
	const (
		numChannels   = 1
		bitsPerSample = 16
	)
	dataSize := uint32(len(samples) * 2)
	byteRate := uint32(sampleRate * numChannels * bitsPerSample / 8)
	blockAlign := uint16(numChannels * bitsPerSample / 8)

	buf := make([]byte, 44)
	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], 36+dataSize)
	copy(buf[8:12], "WAVE")
	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(buf[22:24], uint16(numChannels))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:32], byteRate)
	binary.LittleEndian.PutUint16(buf[32:34], blockAlign)
	binary.LittleEndian.PutUint16(buf[34:36], bitsPerSample)
	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], dataSize)
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("audio: write WAV header: %w", err)
	}

	// Stream samples in modest batches to keep allocation bounded
	// regardless of input length.
	const batch = 4096
	out := make([]byte, batch*2)
	for start := 0; start < len(samples); start += batch {
		end := min(start+batch, len(samples))
		for i, s := range samples[start:end] {
			binary.LittleEndian.PutUint16(out[i*2:], uint16(quantizePCM16(s)))
		}
		if _, err := w.Write(out[:(end-start)*2]); err != nil {
			return fmt.Errorf("audio: write WAV samples: %w", err)
		}
	}
	return nil
}

func quantizePCM16(f float32) int16 {
	scaled := float64(f) * 32767
	if scaled > math.MaxInt16 {
		return math.MaxInt16
	}
	if scaled < math.MinInt16 {
		return math.MinInt16
	}
	return int16(scaled)
}
