package voicelabel

import (
	"math"
	"testing"
)

func sineWave(freq float64, sampleRate int, duration float64) []float32 {
	n := int(float64(sampleRate) * duration)
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(sampleRate)))
	}
	return out
}

func medianF0(t *testing.T, frames []F0Frame, minConfidence float64) float64 {
	t.Helper()
	var voiced []float64
	for _, f := range frames {
		if f.Confidence >= minConfidence && f.F0Hz > 0 {
			voiced = append(voiced, f.F0Hz)
		}
	}
	if len(voiced) == 0 {
		t.Fatal("no voiced frames in YIN output")
	}
	for i := 1; i < len(voiced); i++ {
		j := i
		for j > 0 && voiced[j-1] > voiced[j] {
			voiced[j-1], voiced[j] = voiced[j], voiced[j-1]
			j--
		}
	}
	return voiced[len(voiced)/2]
}

func TestEstimateF0_KnownSine(t *testing.T) {
	const sr = 16000
	cfg := DefaultYINConfig(sr)

	cases := []struct {
		name string
		freq float64
	}{
		{"male-low", 100},
		{"male-typical", 130},
		{"crossover", 165},
		{"female-typical", 220},
		{"female-high", 280},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			samples := sineWave(tc.freq, sr, 1.0)
			frames := EstimateF0(samples, cfg)
			if len(frames) == 0 {
				t.Fatalf("no frames produced for %.0f Hz", tc.freq)
			}
			got := medianF0(t, frames, 0.7)
			// Sub-sample precision via parabolic interpolation should
			// hit within 1 Hz on a clean sinusoid.
			if math.Abs(got-tc.freq) > 1.0 {
				t.Errorf("median F0 = %.2f Hz, want %.0f Hz (±1)", got, tc.freq)
			}
		})
	}
}

func TestEstimateF0_SilenceUnvoiced(t *testing.T) {
	const sr = 16000
	silence := make([]float32, sr) // 1 s of zeros
	frames := EstimateF0(silence, DefaultYINConfig(sr))
	if len(frames) == 0 {
		t.Fatal("expected frames for 1 s of silence")
	}
	for i, f := range frames {
		if f.F0Hz != 0 {
			t.Errorf("frame %d on silence: F0 = %.2f, want 0", i, f.F0Hz)
		}
	}
}

func TestEstimateF0_RejectsTooShort(t *testing.T) {
	cfg := DefaultYINConfig(16000)
	samples := make([]float32, cfg.FrameSize-1)
	if frames := EstimateF0(samples, cfg); frames != nil {
		t.Errorf("expected nil for sub-frame input, got %d frames", len(frames))
	}
}

func TestEstimateF0_RejectsBadConfig(t *testing.T) {
	samples := sineWave(200, 16000, 0.5)
	cases := []YINConfig{
		{}, // zero everywhere
		{SampleRate: 16000, FrameSize: 100, HopSize: 50, TauMin: 200, TauMax: 200, Threshold: 0.15}, // TauMin >= TauMax
		{SampleRate: 16000, FrameSize: 100, HopSize: 50, TauMin: 40, TauMax: 200, Threshold: 0.15},  // FrameSize <= TauMax
	}
	for i, cfg := range cases {
		if frames := EstimateF0(samples, cfg); frames != nil {
			t.Errorf("case %d: expected nil for bad config, got %d frames", i, len(frames))
		}
	}
}
