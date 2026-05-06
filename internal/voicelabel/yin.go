// Package voicelabel attaches a coarse voice-characterization label
// (M/F/?) to each diarization cluster, using YIN-based fundamental
// frequency estimation over the cluster's clean speech regions.
//
// This is intentionally pure DSP — no ML model, no network. Adult-male
// vs adult-female F0 distributions barely overlap (males ~85–180 Hz
// median, females ~165–255 Hz median), so a thresholded median F0 is
// enough to disambiguate roughly 92–96% of speakers on clean
// conversational audio.
package voicelabel

// YINConfig controls the F0 estimator. Defaults assume 16 kHz mono input.
type YINConfig struct {
	// SampleRate is the input sample rate. Required.
	SampleRate int

	// FrameSize is the analysis-window length in samples. 50–80 ms is
	// the YIN sweet spot; 1024 samples = 64 ms at 16 kHz works well.
	// Must exceed TauMax by enough samples for a stable autocorrelation.
	FrameSize int

	// HopSize is the step between consecutive frames in samples.
	HopSize int

	// TauMin is the smallest lag considered, in samples. Bounds the
	// max F0 we'll return: F0_max = SampleRate / TauMin.
	TauMin int

	// TauMax is the largest lag considered, in samples. Bounds the
	// min F0 we'll return: F0_min = SampleRate / TauMax. Must be
	// smaller than FrameSize.
	TauMax int

	// Threshold is the YIN absolute-threshold value. Lags whose
	// cumulative-mean-normalized difference dips below this are
	// candidate periods. 0.10–0.15 is the standard range.
	Threshold float64
}

// DefaultYINConfig returns a tuning suited to 16 kHz speech: 64 ms
// windows, 16 ms hop, F0 search range 60–400 Hz, threshold 0.15.
func DefaultYINConfig(sampleRate int) YINConfig {
	return YINConfig{
		SampleRate: sampleRate,
		FrameSize:  1024,
		HopSize:    256,
		TauMin:     sampleRate / 400, // 60 Hz < F0 < 400 Hz
		TauMax:     sampleRate / 60,
		Threshold:  0.15,
	}
}

// F0Frame is one YIN estimate. F0Hz is 0 when the frame is unvoiced
// (no lag dipped below the threshold). Confidence is 1 - d'(τ) at the
// chosen lag; higher values indicate a stronger periodicity match.
type F0Frame struct {
	F0Hz       float64
	Confidence float64
}

// EstimateF0 walks frame-by-frame across samples and returns one
// F0Frame per analysis window. Returns nil when the input is shorter
// than one frame or when the config is invalid.
func EstimateF0(samples []float32, cfg YINConfig) []F0Frame {
	if cfg.SampleRate <= 0 || cfg.FrameSize <= 0 || cfg.HopSize <= 0 {
		return nil
	}
	if cfg.TauMax <= 0 || cfg.TauMin <= 0 || cfg.TauMin >= cfg.TauMax {
		return nil
	}
	if cfg.FrameSize <= cfg.TauMax {
		return nil
	}
	if len(samples) < cfg.FrameSize {
		return nil
	}
	nFrames := (len(samples)-cfg.FrameSize)/cfg.HopSize + 1
	out := make([]F0Frame, nFrames)
	diff := make([]float64, cfg.TauMax+1)
	cmnd := make([]float64, cfg.TauMax+1)
	for i := range nFrames {
		start := i * cfg.HopSize
		out[i] = yinFrame(samples[start:start+cfg.FrameSize], cfg, diff, cmnd)
	}
	return out
}

// yinFrame is the per-frame YIN estimator. diff and cmnd are scratch
// buffers reused across frames to keep allocation flat.
func yinFrame(frame []float32, cfg YINConfig, diff, cmnd []float64) F0Frame {
	W := cfg.FrameSize

	// Step 1+2: difference function d(τ) = Σ (x[j] - x[j+τ])^2
	for tau := 1; tau <= cfg.TauMax; tau++ {
		var sum float64
		end := W - tau
		for j := range end {
			d := float64(frame[j]) - float64(frame[j+tau])
			sum += d * d
		}
		diff[tau] = sum
	}

	// Step 3: cumulative mean normalized difference
	//   d'(τ) = d(τ) * τ / Σ_{j=1}^{τ} d(j)
	cmnd[0] = 1
	var running float64
	for tau := 1; tau <= cfg.TauMax; tau++ {
		running += diff[tau]
		if running > 0 {
			cmnd[tau] = diff[tau] * float64(tau) / running
		} else {
			cmnd[tau] = 1
		}
	}

	// Step 4: absolute threshold — first τ ≥ TauMin whose d' dips below
	// Threshold; walk forward to the bottom of that local minimum so the
	// parabolic refinement in step 5 sits on a true minimum.
	tau := -1
	for t := cfg.TauMin; t <= cfg.TauMax; t++ {
		if cmnd[t] < cfg.Threshold {
			for t+1 <= cfg.TauMax && cmnd[t+1] < cmnd[t] {
				t++
			}
			tau = t
			break
		}
	}
	if tau < 0 {
		return F0Frame{}
	}

	// Step 5: parabolic interpolation around the chosen minimum for
	// sub-sample period precision. Skip when at the edge of the search
	// range — there's no left/right neighbor to interpolate against.
	refined := float64(tau)
	if tau > cfg.TauMin && tau < cfg.TauMax {
		a, b, c := cmnd[tau-1], cmnd[tau], cmnd[tau+1]
		denom := 2 * (a - 2*b + c)
		if denom != 0 {
			refined = float64(tau) + (a-c)/denom
		}
	}
	if refined <= 0 {
		return F0Frame{}
	}
	return F0Frame{
		F0Hz:       float64(cfg.SampleRate) / refined,
		Confidence: 1 - cmnd[tau],
	}
}
