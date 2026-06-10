package voicelabel

import (
	"sort"
	"time"

	"github.com/matthewjhunter/transcribe/internal/diarize"
)

// Label is a coarse voice-characterization tag attached to a
// diarization cluster. Empty string means no attempt was made.
type Label string

const (
	LabelMale    Label = "M"
	LabelFemale  Label = "F"
	LabelUnknown Label = "?"
)

// ClassifyOptions controls cluster-level F0 aggregation and the
// median-F0 → Label decision.
type ClassifyOptions struct {
	// MaleMaxHz is the upper bound for assigning LabelMale. Median F0
	// strictly below this maps to M. Default 155.
	MaleMaxHz float64

	// FemaleMinHz is the lower bound for assigning LabelFemale. Median
	// F0 strictly above this maps to F. Default 180.
	FemaleMinHz float64

	// MinConfidence is the YIN per-frame confidence threshold a frame
	// must clear to count as voiced. Default 0.7.
	MinConfidence float64

	// MinF0Hz / MaxF0Hz bound the F0 values we'll trust per frame.
	// Defaults: 60 / 400. Catches octave errors and unvoiced noise.
	MinF0Hz float64
	MaxF0Hz float64

	// MaxClusterAudio caps the per-cluster audio analyzed. The aim is
	// a stable median over 5-30 s of voiced speech; more is wasteful.
	// Default 30 s.
	MaxClusterAudio time.Duration

	// MinFrames is the minimum number of voiced frames required to
	// commit to M or F. Below this, the cluster gets LabelUnknown.
	// Default 30 frames (~0.5 s of voiced speech at 16 ms hop).
	MinFrames int
}

// DefaultClassifyOptions are the defaults applied to zero-valued fields.
func DefaultClassifyOptions() ClassifyOptions {
	return ClassifyOptions{
		MaleMaxHz:       155,
		FemaleMinHz:     180,
		MinConfidence:   0.7,
		MinF0Hz:         60,
		MaxF0Hz:         400,
		MaxClusterAudio: 30 * time.Second,
		MinFrames:       30,
	}
}

// ClusterStats reports the per-cluster F0 estimate and the assigned label.
type ClusterStats struct {
	MedianF0Hz float64
	Frames     int
	Label      Label
}

// ClassifyClusters walks every diarization cluster (turn.Speaker), pulls
// up to MaxClusterAudio of its audio out of samples, runs YIN, and
// returns a per-cluster stats map keyed by speaker ID.
//
// Inputs are not mutated. samples must be 16 kHz mono float32 in [-1, 1]
// (the canonical pipeline format). turns must be ordered by Start; the
// canonical diarize.Process output already is.
func ClassifyClusters(samples []float32, sampleRate int, turns []diarize.Turn, opts ClassifyOptions) map[int]ClusterStats {
	opts = applyDefaults(opts)
	out := map[int]ClusterStats{}
	if len(samples) == 0 || len(turns) == 0 || sampleRate <= 0 {
		return out
	}
	yinCfg := DefaultYINConfig(sampleRate)
	clusters := groupBySpeaker(turns)
	for speaker, clusterTurns := range clusters {
		audio := collectClusterAudio(samples, sampleRate, clusterTurns, opts.MaxClusterAudio)
		if len(audio) < yinCfg.FrameSize {
			out[speaker] = ClusterStats{Label: LabelUnknown}
			continue
		}
		out[speaker] = classifyOne(audio, yinCfg, opts)
	}
	return out
}

func applyDefaults(opts ClassifyOptions) ClassifyOptions {
	d := DefaultClassifyOptions()
	if opts.MaleMaxHz <= 0 {
		opts.MaleMaxHz = d.MaleMaxHz
	}
	if opts.FemaleMinHz <= 0 {
		opts.FemaleMinHz = d.FemaleMinHz
	}
	if opts.MinConfidence <= 0 {
		opts.MinConfidence = d.MinConfidence
	}
	if opts.MinF0Hz <= 0 {
		opts.MinF0Hz = d.MinF0Hz
	}
	if opts.MaxF0Hz <= 0 {
		opts.MaxF0Hz = d.MaxF0Hz
	}
	if opts.MaxClusterAudio <= 0 {
		opts.MaxClusterAudio = d.MaxClusterAudio
	}
	if opts.MinFrames <= 0 {
		opts.MinFrames = d.MinFrames
	}
	return opts
}

func groupBySpeaker(turns []diarize.Turn) map[int][]diarize.Turn {
	out := map[int][]diarize.Turn{}
	for _, t := range turns {
		out[t.Speaker] = append(out[t.Speaker], t)
	}
	return out
}

// collectClusterAudio concatenates samples from the cluster's turns up
// to maxDuration of cumulative speech. Out-of-range turns are clamped.
func collectClusterAudio(samples []float32, sampleRate int, turns []diarize.Turn, maxDuration time.Duration) []float32 {
	maxSamples := durationToSamples(maxDuration, sampleRate)
	var out []float32
	for _, t := range turns {
		startSample := durationToSamples(t.Start, sampleRate)
		endSample := durationToSamples(t.End, sampleRate)
		startSample = max(startSample, 0)
		endSample = min(endSample, len(samples))
		if startSample >= endSample {
			continue
		}
		out = append(out, samples[startSample:endSample]...)
		if len(out) >= maxSamples {
			out = out[:maxSamples]
			break
		}
	}
	return out
}

func classifyOne(audio []float32, yinCfg YINConfig, opts ClassifyOptions) ClusterStats {
	frames := EstimateF0(audio, yinCfg)
	voiced := make([]float64, 0, len(frames))
	for _, f := range frames {
		if f.Confidence < opts.MinConfidence {
			continue
		}
		if f.F0Hz < opts.MinF0Hz || f.F0Hz > opts.MaxF0Hz {
			continue
		}
		voiced = append(voiced, f.F0Hz)
	}
	if len(voiced) < opts.MinFrames {
		return ClusterStats{Frames: len(voiced), Label: LabelUnknown}
	}
	sort.Float64s(voiced)
	median := voiced[len(voiced)/2]
	label := LabelUnknown
	switch {
	case median < opts.MaleMaxHz:
		label = LabelMale
	case median > opts.FemaleMinHz:
		label = LabelFemale
	}
	return ClusterStats{
		MedianF0Hz: median,
		Frames:     len(voiced),
		Label:      label,
	}
}

func durationToSamples(d time.Duration, sampleRate int) int {
	return int(float64(d) / float64(time.Second) * float64(sampleRate))
}
