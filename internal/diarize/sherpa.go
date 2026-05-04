// Package diarize wraps sherpa-onnx's offline speaker diarization for
// in-process use from Go. The diarizer takes a slice of 16 kHz mono
// float32 samples and returns time-bounded turns labeled with a speaker
// identifier.
//
// Two ONNX model files are required at runtime: a pyannote segmentation
// model and a speaker embedding model. Use the helpers in models.go to
// fetch the canonical files into the user's cache directory.
package diarize

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Config controls a Diarizer. Empty fields fall back to package defaults.
type Config struct {
	// SegmentationModel is the path to the pyannote-segmentation-3.0
	// ONNX file. Required.
	SegmentationModel string

	// EmbeddingModel is the path to a speaker-embedding ONNX file
	// (NeMo TitaNet, 3D-Speaker, WeSpeaker, etc.). Required.
	EmbeddingModel string

	// NumSpeakers is the number of distinct speakers to assume. Zero
	// means auto-discover via clustering with Threshold below.
	NumSpeakers int

	// Threshold is the clustering threshold used when NumSpeakers == 0.
	// Smaller values produce more speakers; larger values merge similar
	// voices. Sherpa default is 0.5.
	Threshold float32

	// NumThreads is the threadpool size for sherpa's segmentation and
	// embedding stages. Zero defaults to runtime.NumCPU().
	NumThreads int

	// Debug enables sherpa's debug logging when non-zero.
	Debug bool
}

// Diarizer wraps a sherpa OfflineSpeakerDiarization. Close it when done.
type Diarizer struct {
	impl       *sherpa.OfflineSpeakerDiarization
	sampleRate int
}

// New constructs a Diarizer. Returns an error if the model files are
// missing, unreadable, or sherpa-onnx fails to initialize.
func New(cfg Config) (*Diarizer, error) {
	if cfg.SegmentationModel == "" {
		return nil, errors.New("diarize: SegmentationModel is required")
	}
	if cfg.EmbeddingModel == "" {
		return nil, errors.New("diarize: EmbeddingModel is required")
	}
	if _, err := os.Stat(cfg.SegmentationModel); err != nil {
		return nil, fmt.Errorf("diarize: segmentation model: %w", err)
	}
	if _, err := os.Stat(cfg.EmbeddingModel); err != nil {
		return nil, fmt.Errorf("diarize: embedding model: %w", err)
	}

	threads := cfg.NumThreads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	threshold := cfg.Threshold
	if threshold == 0 && cfg.NumSpeakers == 0 {
		threshold = 0.5
	}
	debug := 0
	if cfg.Debug {
		debug = 1
	}

	sc := &sherpa.OfflineSpeakerDiarizationConfig{
		Segmentation: sherpa.OfflineSpeakerSegmentationModelConfig{
			Pyannote:   sherpa.OfflineSpeakerSegmentationPyannoteModelConfig{Model: cfg.SegmentationModel},
			NumThreads: threads,
			Debug:      debug,
		},
		Embedding: sherpa.SpeakerEmbeddingExtractorConfig{
			Model:      cfg.EmbeddingModel,
			NumThreads: threads,
			Debug:      debug,
		},
		Clustering: sherpa.FastClusteringConfig{
			NumClusters: cfg.NumSpeakers,
			Threshold:   threshold,
		},
	}

	impl := sherpa.NewOfflineSpeakerDiarization(sc)
	if impl == nil {
		return nil, errors.New("diarize: sherpa.NewOfflineSpeakerDiarization returned nil (check model files)")
	}
	d := &Diarizer{impl: impl, sampleRate: impl.SampleRate()}
	runtime.SetFinalizer(d, func(d *Diarizer) { _ = d.Close() })
	return d, nil
}

// SampleRate returns the rate the diarizer expects from Process input.
// For pyannote-segmentation-3.0 this is 16000.
func (d *Diarizer) SampleRate() int { return d.sampleRate }

// Process runs diarization over a slice of mono float32 PCM samples.
// inputRate must match SampleRate(); resample upstream if it doesn't.
func (d *Diarizer) Process(samples []float32, inputRate int) ([]Turn, error) {
	if d.impl == nil {
		return nil, errors.New("diarize: Diarizer has been closed")
	}
	if inputRate != d.sampleRate {
		return nil, fmt.Errorf("diarize: input sample rate %d does not match expected %d",
			inputRate, d.sampleRate)
	}
	if len(samples) == 0 {
		return nil, errors.New("diarize: empty samples slice")
	}

	segs := d.impl.Process(samples)
	turns := make([]Turn, len(segs))
	for i, s := range segs {
		turns[i] = Turn{
			Start:   secondsFloat(float64(s.Start)),
			End:     secondsFloat(float64(s.End)),
			Speaker: s.Speaker,
		}
	}
	return turns, nil
}

// Close releases the underlying sherpa instance. Safe to call multiple times.
func (d *Diarizer) Close() error {
	if d.impl != nil {
		sherpa.DeleteOfflineSpeakerDiarization(d.impl)
		d.impl = nil
	}
	return nil
}

func secondsFloat(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}
