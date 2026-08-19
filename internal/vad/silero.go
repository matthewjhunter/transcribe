// Package vad wraps sherpa-onnx's Silero VAD for offline speech-region
// detection. The detector takes a slice of 16 kHz mono float32 samples
// and returns time-bounded speech segments.
//
// One ONNX model file is required at runtime: silero_vad.onnx. Use
// EnsureModel from models.go to fetch it into the user's cache directory.
package vad

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

const defaultSampleRate = 16000

// defaultThreads is the ONNX threadpool size used when Config.NumThreads is
// unset. Silero is a ~2 MB model stepped 512 samples at a time, so there is
// almost nothing in a window to parallelise and the pool's per-window
// synchronisation dominates the arithmetic. Handing it every core makes it
// dramatically slower, not faster: measured over 300 s of audio on a
// 32-thread Ryzen AI MAX+ 395, 1 thread took 0.5 s and 32 took 12.1 s, a 24x
// penalty. Callers who know better can still override it.
const defaultThreads = 1

// Config controls a Detector. Empty fields fall back to package defaults.
type Config struct {
	// Model is the path to the silero_vad.onnx file. Required.
	Model string

	// SampleRate is the rate of the samples passed to Detect. Defaults
	// to 16000, which is the rate the canonical Silero export expects.
	SampleRate int

	// Threshold is the per-frame speech probability cutoff (0..1).
	// Sherpa default is 0.5.
	Threshold float32

	// MinSilenceDuration is the silence span (seconds) needed to end a
	// speech segment. Sherpa default is 0.5.
	MinSilenceDuration float32

	// MinSpeechDuration drops segments shorter than this many seconds.
	// Sherpa default is 0.25.
	MinSpeechDuration float32

	// MaxSpeechDuration is the soft cap on segment length (seconds).
	// When a segment exceeds this, sherpa internally raises the
	// threshold to 0.9 to force a cut. Useful for keeping chunks
	// inside the Whisper 30 s window.
	MaxSpeechDuration float32

	// WindowSize is the per-step sample count fed to the VAD. Silero's
	// canonical export expects 512 samples at 16 kHz.
	WindowSize int

	// BufferSeconds is the size of sherpa's internal segment buffer.
	// Should be at least MaxSpeechDuration. Defaults to MaxSpeechDuration+10.
	BufferSeconds float32

	// NumThreads is the threadpool size for the ONNX runtime.
	// Zero defaults to defaultThreads (1) -- see the note there before
	// raising it; more threads measurably slow this model down.
	NumThreads int

	// Provider selects the ONNX execution provider ("cpu", "cuda", ...).
	// Empty leaves the sherpa default ("cpu").
	Provider string

	// Debug enables sherpa's debug logging when non-zero.
	Debug bool
}

// Segment is one detected speech region with absolute time bounds in
// the input audio.
type Segment struct {
	Start time.Duration
	End   time.Duration
}

// Detector wraps a sherpa VoiceActivityDetector. Close it when done.
type Detector struct {
	impl       *sherpa.VoiceActivityDetector
	sampleRate int
	windowSize int
}

// New constructs a Detector. Returns an error if the model file is
// missing, unreadable, or sherpa-onnx fails to initialize.
func New(cfg Config) (*Detector, error) {
	if cfg.Model == "" {
		return nil, errors.New("vad: Model is required")
	}
	if _, err := os.Stat(cfg.Model); err != nil {
		return nil, fmt.Errorf("vad: model: %w", err)
	}

	sampleRate := cfg.SampleRate
	if sampleRate == 0 {
		sampleRate = defaultSampleRate
	}
	threshold := cfg.Threshold
	if threshold == 0 {
		threshold = 0.5
	}
	minSilence := cfg.MinSilenceDuration
	if minSilence == 0 {
		minSilence = 0.5
	}
	minSpeech := cfg.MinSpeechDuration
	if minSpeech == 0 {
		minSpeech = 0.25
	}
	maxSpeech := cfg.MaxSpeechDuration
	if maxSpeech == 0 {
		maxSpeech = 28
	}
	windowSize := cfg.WindowSize
	if windowSize == 0 {
		windowSize = 512
	}
	bufferSeconds := cfg.BufferSeconds
	if bufferSeconds == 0 {
		bufferSeconds = maxSpeech + 10
	}
	threads := cfg.NumThreads
	if threads <= 0 {
		threads = defaultThreads
	}
	debug := 0
	if cfg.Debug {
		debug = 1
	}

	sc := &sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              cfg.Model,
			Threshold:          threshold,
			MinSilenceDuration: minSilence,
			MinSpeechDuration:  minSpeech,
			MaxSpeechDuration:  maxSpeech,
			WindowSize:         windowSize,
		},
		SampleRate: sampleRate,
		NumThreads: threads,
		Provider:   cfg.Provider,
		Debug:      debug,
	}

	impl := sherpa.NewVoiceActivityDetector(sc, bufferSeconds)
	if impl == nil {
		return nil, errors.New("vad: NewVoiceActivityDetector returned nil (check model file)")
	}
	d := &Detector{impl: impl, sampleRate: sampleRate, windowSize: windowSize}
	runtime.SetFinalizer(d, func(d *Detector) { _ = d.Close() })
	return d, nil
}

// SampleRate returns the rate the detector expects from Detect input.
func (d *Detector) SampleRate() int { return d.sampleRate }

// Detect runs Silero VAD over a slice of mono float32 PCM samples and
// returns the detected speech segments in input time order.
//
// inputRate must match SampleRate(); resample upstream if it doesn't.
func (d *Detector) Detect(samples []float32, inputRate int) ([]Segment, error) {
	if d.impl == nil {
		return nil, errors.New("vad: Detector has been closed")
	}
	if inputRate != d.sampleRate {
		return nil, fmt.Errorf("vad: input sample rate %d does not match expected %d",
			inputRate, d.sampleRate)
	}
	if len(samples) == 0 {
		return nil, errors.New("vad: empty samples slice")
	}

	out := make([]Segment, 0)
	drain := func() {
		for !d.impl.IsEmpty() {
			seg := d.impl.Front()
			out = append(out, Segment{
				Start: samplesToDuration(seg.Start, d.sampleRate),
				End:   samplesToDuration(seg.Start+len(seg.Samples), d.sampleRate),
			})
			d.impl.Pop()
		}
	}

	for i := 0; i+d.windowSize <= len(samples); i += d.windowSize {
		d.impl.AcceptWaveform(samples[i : i+d.windowSize])
		drain()
	}
	d.impl.Flush()
	drain()
	return out, nil
}

// Close releases the underlying sherpa instance. Safe to call multiple times.
func (d *Detector) Close() error {
	if d.impl != nil {
		sherpa.DeleteVoiceActivityDetector(d.impl)
		d.impl = nil
	}
	return nil
}

func samplesToDuration(n, sampleRate int) time.Duration {
	return time.Duration(float64(n) / float64(sampleRate) * float64(time.Second))
}
